#!/usr/bin/env python3
"""
Near-duplicate detection and removal for R2 custom training samples.

Two-pass deduplication:
  1. Exact fingerprint dedup: (category, first_user_msg[:100], first_tool_name)
  2. TF-IDF cosine similarity within each category (threshold > 0.85)

Input:  /Users/aiqarus/Desktop/Projects/aiqarus-agent-4b/data/upload/custom_all.jsonl
Output: /Users/aiqarus/Desktop/Projects/aiqarus-agent-4b/aiqarus-agent-4b-2603/data/v3/custom_deduped.jsonl
        /Users/aiqarus/Desktop/Projects/aiqarus-agent-4b/aiqarus-agent-4b-2603/data/v3/custom_dupes_removed.jsonl
"""

import hashlib
import json
import os
import sys
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH = "/Users/aiqarus/Desktop/Projects/aiqarus-agent-4b/data/upload/custom_all.jsonl"
OUTPUT_DIR = "/Users/aiqarus/Desktop/Projects/aiqarus-agent-4b/aiqarus-agent-4b-2603/data/v3"
DEDUPED_PATH = os.path.join(OUTPUT_DIR, "custom_deduped.jsonl")
REMOVED_PATH = os.path.join(OUTPUT_DIR, "custom_dupes_removed.jsonl")

SIMILARITY_THRESHOLD = 0.85


def extract_first_user_message(messages: list) -> str:
    """Return the content of the first user message, or empty string."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_first_tool_name(tools: list) -> str:
    """Return the name of the first tool, or empty string."""
    if tools and isinstance(tools, list) and len(tools) > 0:
        tool = tools[0]
        # Tools have "name" key directly (not nested in "function")
        return tool.get("name", "")
    return ""


def fingerprint(category: str, user_msg: str, tool_name: str) -> str:
    """Create a hash fingerprint from (category, user_msg[:100], first_tool_name)."""
    key = f"{category}||{user_msg[:100]}||{tool_name}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def main():
    # ------------------------------------------------------------------
    # Load all samples
    # ------------------------------------------------------------------
    samples = []
    with open(INPUT_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    total_input = len(samples)
    print(f"Loaded {total_input} samples from {INPUT_PATH}\n")

    # ------------------------------------------------------------------
    # Pass 1: Exact fingerprint dedup
    # ------------------------------------------------------------------
    seen_fps = {}  # fingerprint -> index of first occurrence
    fp_dupes = []  # list of (sample, reason)
    fp_kept = []   # samples surviving pass 1

    for i, sample in enumerate(samples):
        cat = sample.get("category", "unknown")
        user_msg = extract_first_user_message(sample.get("messages", []))
        tool_name = extract_first_tool_name(sample.get("tools", []))
        fp = fingerprint(cat, user_msg, tool_name)

        if fp in seen_fps:
            first_idx = seen_fps[fp]
            first_id = samples[first_idx].get("id", f"idx-{first_idx}")
            fp_dupes.append((sample, f"exact_fingerprint_dupe_of:{first_id}"))
        else:
            seen_fps[fp] = i
            fp_kept.append(sample)

    print(f"Pass 1 (fingerprint): {len(fp_dupes)} exact dupes removed, {len(fp_kept)} kept\n")

    # Tally fingerprint dupes by category
    fp_dupe_cats = defaultdict(int)
    for sample, _ in fp_dupes:
        fp_dupe_cats[sample.get("category", "unknown")] += 1

    if fp_dupe_cats:
        print("  Fingerprint dupes by category:")
        for cat in sorted(fp_dupe_cats, key=lambda c: -fp_dupe_cats[c]):
            print(f"    {cat}: {fp_dupe_cats[cat]}")
        print()

    # ------------------------------------------------------------------
    # Pass 2: TF-IDF cosine similarity within each category
    # ------------------------------------------------------------------
    # Group surviving samples by category
    cat_groups = defaultdict(list)
    for idx, sample in enumerate(fp_kept):
        cat_groups[sample.get("category", "unknown")].append((idx, sample))

    sim_dupes_set = set()  # indices into fp_kept to remove
    sim_dupe_details = []  # (sample, reason)
    similarity_scores = []  # all pairwise scores > threshold (for distribution)

    print(f"Pass 2 (TF-IDF cosine similarity > {SIMILARITY_THRESHOLD}):")
    for cat in sorted(cat_groups):
        group = cat_groups[cat]
        if len(group) < 2:
            continue

        # Extract user messages for this category
        indices = [idx for idx, _ in group]
        user_msgs = []
        for _, sample in group:
            user_msgs.append(extract_first_user_message(sample.get("messages", [])))

        # Filter out empty messages
        valid = [(i, msg) for i, msg in zip(indices, user_msgs) if msg.strip()]
        if len(valid) < 2:
            continue

        valid_indices = [v[0] for v in valid]
        valid_msgs = [v[1] for v in valid]

        # TF-IDF + cosine similarity
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(valid_msgs)
        except ValueError:
            # All messages too short or empty after stop words removal
            continue

        sim_matrix = cosine_similarity(tfidf_matrix)

        # Find pairs above threshold (upper triangle only)
        cat_flagged = 0
        for i in range(len(valid_indices)):
            if valid_indices[i] in sim_dupes_set:
                continue  # already marked for removal
            for j in range(i + 1, len(valid_indices)):
                if valid_indices[j] in sim_dupes_set:
                    continue
                score = sim_matrix[i, j]
                if score > SIMILARITY_THRESHOLD:
                    similarity_scores.append(score)
                    # Remove the second sample (keep first occurrence)
                    remove_idx = valid_indices[j]
                    keep_idx = valid_indices[i]
                    sim_dupes_set.add(remove_idx)
                    keep_id = fp_kept[keep_idx].get("id", f"idx-{keep_idx}")
                    sim_dupe_details.append(
                        (fp_kept[remove_idx],
                         f"tfidf_similarity={score:.4f}_similar_to:{keep_id}")
                    )
                    cat_flagged += 1

        if cat_flagged > 0:
            print(f"  {cat}: {cat_flagged} near-dupes flagged")

    print(f"\nPass 2 total: {len(sim_dupes_set)} near-dupes removed\n")

    # ------------------------------------------------------------------
    # Build final outputs
    # ------------------------------------------------------------------
    final_kept = []
    for idx, sample in enumerate(fp_kept):
        if idx not in sim_dupes_set:
            final_kept.append(sample)

    all_removed = []
    for sample, reason in fp_dupes:
        entry = dict(sample)
        entry["_removal_reason"] = reason
        all_removed.append(entry)
    for sample, reason in sim_dupe_details:
        entry = dict(sample)
        entry["_removal_reason"] = reason
        all_removed.append(entry)

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(DEDUPED_PATH, "w") as f:
        for sample in final_kept:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    with open(REMOVED_PATH, "w") as f:
        for entry in all_removed:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    total_removed = len(all_removed)
    total_after = len(final_kept)

    # Category-level before/after
    before_cats = defaultdict(int)
    after_cats = defaultdict(int)
    removed_cats = defaultdict(int)

    for s in samples:
        before_cats[s.get("category", "unknown")] += 1
    for s in final_kept:
        after_cats[s.get("category", "unknown")] += 1
    for entry in all_removed:
        removed_cats[entry.get("category", "unknown")] += 1

    print("=" * 72)
    print("DEDUPLICATION REPORT")
    print("=" * 72)
    print(f"  Input:   {total_input} samples")
    print(f"  Output:  {total_after} samples")
    print(f"  Removed: {total_removed} ({total_removed/total_input*100:.1f}%)")
    print(f"    - Exact fingerprint dupes: {len(fp_dupes)}")
    print(f"    - TF-IDF near-dupes (>{SIMILARITY_THRESHOLD}): {len(sim_dupe_details)}")
    print()

    print("  Per-category breakdown:")
    print(f"  {'Category':<30s} {'Before':>7s} {'After':>7s} {'Removed':>8s} {'%':>6s}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")
    for cat in sorted(before_cats):
        b = before_cats[cat]
        a = after_cats.get(cat, 0)
        r = removed_cats.get(cat, 0)
        pct = r / b * 100 if b > 0 else 0
        print(f"  {cat:<30s} {b:>7d} {a:>7d} {r:>8d} {pct:>5.1f}%")
    print()

    # Similarity score distribution
    if similarity_scores:
        import statistics
        print("  TF-IDF similarity distribution (flagged pairs):")
        print(f"    Count:  {len(similarity_scores)}")
        print(f"    Min:    {min(similarity_scores):.4f}")
        print(f"    Max:    {max(similarity_scores):.4f}")
        print(f"    Mean:   {statistics.mean(similarity_scores):.4f}")
        print(f"    Median: {statistics.median(similarity_scores):.4f}")

        # Histogram buckets
        buckets = {
            "0.85-0.90": 0, "0.90-0.95": 0, "0.95-0.99": 0, "0.99-1.00": 0
        }
        for s in similarity_scores:
            if s >= 0.99:
                buckets["0.99-1.00"] += 1
            elif s >= 0.95:
                buckets["0.95-0.99"] += 1
            elif s >= 0.90:
                buckets["0.90-0.95"] += 1
            else:
                buckets["0.85-0.90"] += 1
        print("    Distribution:")
        for bucket, count in buckets.items():
            bar = "#" * min(count, 60)
            print(f"      {bucket}: {count:>4d} {bar}")
    else:
        print("  No TF-IDF near-dupes found above threshold.")

    print()
    print(f"  Deduped file:  {DEDUPED_PATH}")
    print(f"  Removed file:  {REMOVED_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
