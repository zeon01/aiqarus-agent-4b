#!/usr/bin/env python3
"""
prepare_dataset_v3.py — V3 Dataset Assembly Script (~70K effective samples)
===========================================================================
Assembles all V3 training data sources into final train/val files for SFT.

Sources:
  1. R2 customs (synonym-replaced, 3x upsampled): custom_upsampled_synonymed.jsonl
  2. New behavioral failure data: data/v3/behavioral/*.jsonl
  3. New category data: data/v3/categories/*.jsonl
  4. Fresh foundation: data/v3/foundation/all.jsonl or shard_*.jsonl

Upsample strategy:
  - R2 customs: already 3x from synonym replacement — NO additional upsampling
  - New behavioral: 4x (simple duplication)
  - New categories: 4x (simple duplication)
  - Foundation: 1x (no upsampling)

Output:
  data/v3/train_v3.jsonl — training set (95%)
  data/v3/val_v3.jsonl   — validation set (5%)

Each output line: {"messages": [...], "tools": [...]}

Usage:
  python scripts/prepare_dataset_v3.py
  python scripts/prepare_dataset_v3.py --dry-run
"""

import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # aiqarus-agent-4b-2603/
DATA_V3 = PROJECT_ROOT / "data" / "v3"

CUSTOMS_FILE = DATA_V3 / "custom_upsampled_synonymed.jsonl"
BEHAVIORAL_DIR = DATA_V3 / "behavioral"
CATEGORIES_DIR = DATA_V3 / "categories"
FOUNDATION_DIR = DATA_V3 / "foundation"

OUTPUT_TRAIN = DATA_V3 / "train_v3.jsonl"
OUTPUT_VAL = DATA_V3 / "val_v3.jsonl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
VAL_RATIO = 0.05

# Upsample multipliers
UPSAMPLE_BEHAVIORAL = 4
UPSAMPLE_CATEGORIES = 4
UPSAMPLE_CUSTOMS = 1      # already 3x from synonym replacement
UPSAMPLE_FOUNDATION = 1   # no upsampling

# Allowed message keys — strip all metadata from individual messages
ALLOWED_MSG_KEYS = {"role", "content", "name", "tool_calls"}

# Valid roles
VALID_ROLES = {"system", "user", "assistant", "tool"}

# Flush batch size (fsync every N writes)
FLUSH_BATCH = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file, skipping malformed lines."""
    samples = []
    if not path.exists():
        return samples
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return samples


def load_jsonl_dir(directory: Path) -> list[dict]:
    """Load all .jsonl files from a directory."""
    samples = []
    if not directory.exists() or not directory.is_dir():
        return samples
    for p in sorted(directory.glob("*.jsonl")):
        samples.extend(load_jsonl(p))
    return samples


def load_foundation(directory: Path) -> list[dict]:
    """Load foundation data: prefer all.jsonl, fall back to shard_*.jsonl."""
    all_file = directory / "all.jsonl"
    if all_file.exists():
        return load_jsonl(all_file)
    # Fall back to shards
    shards = sorted(directory.glob("shard_*.jsonl"))
    if shards:
        samples = []
        for p in shards:
            samples.extend(load_jsonl(p))
        return samples
    # Fall back to any .jsonl
    return load_jsonl_dir(directory)


def validate_sample(sample: dict) -> bool:
    """Validate a single sample has required structure."""
    messages = sample.get("messages")
    if not messages or not isinstance(messages, list):
        return False

    tools = sample.get("tools")
    if tools is not None and not isinstance(tools, list):
        return False

    # Must have at least: 1 user + 1 assistant message
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    if "user" not in roles or "assistant" not in roles:
        return False

    # All roles must be valid
    for m in messages:
        if not isinstance(m, dict):
            return False
        if m.get("role") not in VALID_ROLES:
            return False
        # Content must exist (can be empty string for some tool messages)
        if "content" not in m:
            return False

    return True


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Strip metadata keys from messages, keeping only allowed keys."""
    cleaned = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        clean_msg = {}
        for key in ALLOWED_MSG_KEYS:
            if key in m:
                clean_msg[key] = m[key]
        if "role" in clean_msg:
            cleaned.append(clean_msg)
    return cleaned


def normalize_tools(tools) -> list[dict]:
    """Normalize tools array to standard format."""
    if not tools or not isinstance(tools, list):
        return []
    normalized = []
    for t in tools:
        if isinstance(t, dict):
            normalized.append(t)
        elif isinstance(t, str):
            try:
                parsed = json.loads(t)
                if isinstance(parsed, dict):
                    normalized.append(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
    return normalized


def normalize_sample(sample: dict) -> dict | None:
    """Normalize a sample to output format: {messages, tools} only."""
    if not validate_sample(sample):
        return None

    messages = normalize_messages(sample.get("messages", []))
    tools = normalize_tools(sample.get("tools"))

    # Re-validate after normalization
    roles = [m.get("role") for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return None

    result = {"messages": messages, "tools": tools}
    return result


def infer_action_type(sample: dict) -> str:
    """Infer action type from sample.

    If the sample has an explicit action_type field, use it.
    Otherwise, heuristically determine from message content.
    """
    # Explicit field
    explicit = sample.get("action_type", "").strip()
    if explicit:
        return explicit

    # Heuristic: scan assistant messages for tool-call indicators
    for msg in sample.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = str(msg.get("content", ""))

        if role == "tool":
            return "call_tool"

        if role == "assistant":
            # <tool_call> tag
            if "<tool_call>" in content:
                return "call_tool"
            # JSON tool call pattern
            if re.search(r'\{"name"\s*:\s*"[^"]+"', content):
                return "call_tool"
            # tool_calls field
            if msg.get("tool_calls"):
                return "call_tool"

    # No tool indicators found — classify from content
    for msg in sample.get("messages", []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content", "")).lower()
        # Strip think block for classification
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()

        if any(w in content for w in ["escalat", "human agent", "transfer to",
                                       "handoff", "hand off", "routing to"]):
            return "escalate"
        if any(w in content for w in ["clarif", "could you specify",
                                       "can you clarify", "which one",
                                       "more details", "please specify"]):
            return "clarify"
        if any(w in content for w in ["cannot", "can't", "unable to",
                                       "not able to", "refuse", "decline",
                                       "not authorized", "not appropriate"]):
            return "refuse"

    return "answer_directly"


def get_category(sample: dict) -> str:
    """Get category from sample, defaulting to 'unknown'."""
    return sample.get("category", "unknown")


def get_source_label(sample: dict) -> str:
    """Get a human-readable source label."""
    return sample.get("source", "unknown")


def approx_token_count(sample: dict) -> int:
    """Approximate token count: characters / 4."""
    total_chars = 0
    for msg in sample.get("messages", []):
        total_chars += len(str(msg.get("content", "")))
    for tool in sample.get("tools", []):
        total_chars += len(json.dumps(tool))
    return total_chars // 4


# ---------------------------------------------------------------------------
# Box-drawing table printer
# ---------------------------------------------------------------------------

def print_table(title: str, headers: list[str], rows: list[list[str]],
                col_widths: list[int] | None = None):
    """Print a table using box-drawing characters."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(h)
            for row in rows:
                if i < len(row):
                    max_w = max(max_w, len(row[i]))
            col_widths.append(max_w + 2)

    def make_line(left, mid, right, fill):
        parts = [fill * w for w in col_widths]
        return left + mid.join(parts) + right

    def make_row(cells):
        parts = []
        for i, cell in enumerate(cells):
            w = col_widths[i] - 2
            if i == 0:
                parts.append(" " + cell.ljust(w) + " ")
            else:
                # Right-align numeric-looking values
                stripped = cell.strip().replace(",", "").replace("%", "").replace(".", "")
                if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
                    parts.append(" " + cell.rjust(w) + " ")
                else:
                    parts.append(" " + cell.ljust(w) + " ")
        return "\u2502" + "\u2502".join(parts) + "\u2502"

    print(f"\n{title}")
    print(make_line("\u250c", "\u252c", "\u2510", "\u2500"))
    print(make_row(headers))
    print(make_line("\u251c", "\u253c", "\u2524", "\u2500"))
    for row in rows:
        print(make_row(row))
    print(make_line("\u2514", "\u2534", "\u2518", "\u2500"))


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(samples: list[dict], source_tags: list[str],
                     category_tags: list[str], val_ratio: float,
                     seed: int) -> tuple[list[dict], list[dict]]:
    """Split samples into train/val, stratified by source x category.

    Args:
        samples: list of normalized samples
        source_tags: parallel list of source labels
        category_tags: parallel list of category labels
        val_ratio: fraction for validation
        seed: random seed

    Returns:
        (train_samples, val_samples)
    """
    rng = random.Random(seed)

    # Group indices by stratum (source x category)
    strata = defaultdict(list)
    for idx, (src, cat) in enumerate(zip(source_tags, category_tags)):
        strata[(src, cat)].append(idx)

    val_indices = set()

    for key, indices in strata.items():
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_indices.update(indices[:n_val])

    train = [samples[i] for i in range(len(samples)) if i not in val_indices]
    val = [samples[i] for i in range(len(samples)) if i in val_indices]

    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


# ---------------------------------------------------------------------------
# File writer with fsync
# ---------------------------------------------------------------------------

def write_jsonl_safe(samples: list[dict], path: Path):
    """Write JSONL with periodic fsync for safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if (i + 1) % FLUSH_BATCH == 0:
                f.flush()
                os.fsync(f.fileno())
        # Final flush
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False):
    random.seed(SEED)

    print("=" * 70)
    print("V3 DATASET ASSEMBLY")
    print("=" * 70)

    # ── 1. Load all sources ──────────────────────────────────────────────

    # R2 customs (synonym-replaced, already 3x)
    print(f"\nLoading R2 customs from {CUSTOMS_FILE.name}...")
    customs_raw = load_jsonl(CUSTOMS_FILE)
    print(f"  Loaded: {len(customs_raw):,} samples")

    # New behavioral failure data
    print(f"\nLoading behavioral data from {BEHAVIORAL_DIR.name}/...")
    behavioral_raw = load_jsonl_dir(BEHAVIORAL_DIR)
    print(f"  Loaded: {len(behavioral_raw):,} samples")

    # New categories
    print(f"\nLoading category data from {CATEGORIES_DIR.name}/...")
    categories_raw = load_jsonl_dir(CATEGORIES_DIR)
    print(f"  Loaded: {len(categories_raw):,} samples")

    # Foundation
    print(f"\nLoading foundation data from {FOUNDATION_DIR.name}/...")
    foundation_raw = load_foundation(FOUNDATION_DIR)
    print(f"  Loaded: {len(foundation_raw):,} samples")

    total_raw = len(customs_raw) + len(behavioral_raw) + len(categories_raw) + len(foundation_raw)
    print(f"\nTotal raw loaded: {total_raw:,}")

    if total_raw == 0:
        print("\nERROR: No data found. Check that data files exist under data/v3/.")
        return

    # ── 2. Validate and normalize ────────────────────────────────────────

    print("\nValidating and normalizing...")

    def process_source(raw_samples, source_name):
        """Validate, normalize, and return (samples, source_tags, category_tags, action_types, reject_count)."""
        valid = []
        sources = []
        categories = []
        actions = []
        rejected = 0
        for s in raw_samples:
            # Capture metadata BEFORE normalization strips it
            action = infer_action_type(s)
            category = get_category(s)
            src = source_name

            normalized = normalize_sample(s)
            if normalized is None:
                rejected += 1
                continue
            valid.append(normalized)
            sources.append(src)
            categories.append(category)
            actions.append(action)
        return valid, sources, categories, actions, rejected

    customs_valid, customs_src, customs_cat, customs_act, customs_rej = process_source(customs_raw, "r2_customs")
    behavioral_valid, behavioral_src, behavioral_cat, behavioral_act, behavioral_rej = process_source(behavioral_raw, "v3_behavioral")
    categories_valid, categories_src, categories_cat, categories_act, categories_rej = process_source(categories_raw, "v3_categories")
    foundation_valid, foundation_src, foundation_cat, foundation_act, foundation_rej = process_source(foundation_raw, "v3_foundation")

    print(f"  R2 customs:    {len(customs_valid):,} valid, {customs_rej:,} rejected")
    print(f"  Behavioral:    {len(behavioral_valid):,} valid, {behavioral_rej:,} rejected")
    print(f"  Categories:    {len(categories_valid):,} valid, {categories_rej:,} rejected")
    print(f"  Foundation:    {len(foundation_valid):,} valid, {foundation_rej:,} rejected")

    # ── 3. Upsample ─────────────────────────────────────────────────────

    print("\nApplying upsampling...")

    def upsample(samples, sources, categories, actions, factor):
        if factor <= 1:
            return samples, sources, categories, actions
        return (
            samples * factor,
            sources * factor,
            categories * factor,
            actions * factor,
        )

    customs_up, customs_src_up, customs_cat_up, customs_act_up = upsample(
        customs_valid, customs_src, customs_cat, customs_act, UPSAMPLE_CUSTOMS
    )
    behavioral_up, behavioral_src_up, behavioral_cat_up, behavioral_act_up = upsample(
        behavioral_valid, behavioral_src, behavioral_cat, behavioral_act, UPSAMPLE_BEHAVIORAL
    )
    categories_up, categories_src_up, categories_cat_up, categories_act_up = upsample(
        categories_valid, categories_src, categories_cat, categories_act, UPSAMPLE_CATEGORIES
    )
    foundation_up, foundation_src_up, foundation_cat_up, foundation_act_up = upsample(
        foundation_valid, foundation_src, foundation_cat, foundation_act, UPSAMPLE_FOUNDATION
    )

    print(f"  R2 customs:    {len(customs_valid):,} x {UPSAMPLE_CUSTOMS} = {len(customs_up):,}")
    print(f"  Behavioral:    {len(behavioral_valid):,} x {UPSAMPLE_BEHAVIORAL} = {len(behavioral_up):,}")
    print(f"  Categories:    {len(categories_valid):,} x {UPSAMPLE_CATEGORIES} = {len(categories_up):,}")
    print(f"  Foundation:    {len(foundation_valid):,} x {UPSAMPLE_FOUNDATION} = {len(foundation_up):,}")

    # ── 4. Merge all ─────────────────────────────────────────────────────

    all_samples = customs_up + behavioral_up + categories_up + foundation_up
    all_sources = customs_src_up + behavioral_src_up + categories_src_up + foundation_src_up
    all_categories = customs_cat_up + behavioral_cat_up + categories_cat_up + foundation_cat_up
    all_actions = customs_act_up + behavioral_act_up + categories_act_up + foundation_act_up

    total = len(all_samples)
    print(f"\nTotal effective samples: {total:,}")

    # ── 5. Shuffle (flattened curriculum — all mixed) ────────────────────

    print("\nShuffling (flattened curriculum)...")
    combined = list(zip(all_samples, all_sources, all_categories, all_actions))
    random.shuffle(combined)
    all_samples = [c[0] for c in combined]
    all_sources = [c[1] for c in combined]
    all_categories = [c[2] for c in combined]
    all_actions = [c[3] for c in combined]

    # ── 6. Stratified train/val split ────────────────────────────────────

    print(f"\nStratified train/val split ({100 - int(VAL_RATIO * 100)}/{int(VAL_RATIO * 100)})...")
    train_samples, val_samples = stratified_split(
        all_samples, all_sources, all_categories, VAL_RATIO, SEED
    )

    # Reconstruct metadata for val/train for reporting
    # (since stratified_split returns samples only, we need to re-derive)
    # Use index-based approach instead
    val_set = set(id(s) for s in val_samples)

    train_sources = [src for s, src in zip(all_samples, all_sources) if id(s) not in val_set]
    train_categories = [cat for s, cat in zip(all_samples, all_categories) if id(s) not in val_set]
    train_actions = [act for s, act in zip(all_samples, all_actions) if id(s) not in val_set]

    print(f"  Train: {len(train_samples):,}")
    print(f"  Val:   {len(val_samples):,}")

    # ── 7. Compute statistics ────────────────────────────────────────────

    # Source distribution
    source_counts = Counter(all_sources)

    # Category distribution
    category_counts = Counter(all_categories)

    # Action-type distribution
    action_counts = Counter(all_actions)

    # Token statistics
    token_counts = [approx_token_count(s) for s in all_samples]
    if token_counts:
        sorted_tokens = sorted(token_counts)
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / len(token_counts)
        median_tokens = sorted_tokens[len(sorted_tokens) // 2]
        p5_tokens = sorted_tokens[int(len(sorted_tokens) * 0.05)]
        p95_tokens = sorted_tokens[int(len(sorted_tokens) * 0.95)]
        min_tokens = sorted_tokens[0]
        max_tokens = sorted_tokens[-1]
    else:
        total_tokens = avg_tokens = median_tokens = 0
        p5_tokens = p95_tokens = min_tokens = max_tokens = 0

    # ── 8. Print report ──────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("V3 DATASET ASSEMBLY REPORT")
    print("=" * 70)

    # Summary table
    print_table(
        "Dataset Summary",
        ["Metric", "Value"],
        [
            ["Total samples (effective)", f"{total:,}"],
            ["Train samples", f"{len(train_samples):,}"],
            ["Val samples", f"{len(val_samples):,}"],
            ["Val ratio", f"{len(val_samples)/total*100:.1f}%"],
            ["Approx total tokens", f"{total_tokens:,}"],
        ],
    )

    # Source distribution
    source_rows = []
    for src in ["r2_customs", "v3_behavioral", "v3_categories", "v3_foundation"]:
        count = source_counts.get(src, 0)
        pct = count / total * 100 if total else 0
        source_rows.append([src, f"{count:,}", f"{pct:.1f}%"])
    source_rows.append(["TOTAL", f"{total:,}", "100.0%"])

    print_table(
        "Source Distribution",
        ["Source", "Count", "Pct"],
        source_rows,
    )

    # Upsample detail
    print_table(
        "Upsample Detail",
        ["Source", "Raw Valid", "Multiplier", "Effective"],
        [
            ["R2 customs (synonym-replaced)", f"{len(customs_valid):,}", f"{UPSAMPLE_CUSTOMS}x", f"{len(customs_up):,}"],
            ["New behavioral", f"{len(behavioral_valid):,}", f"{UPSAMPLE_BEHAVIORAL}x", f"{len(behavioral_up):,}"],
            ["New categories", f"{len(categories_valid):,}", f"{UPSAMPLE_CATEGORIES}x", f"{len(categories_up):,}"],
            ["Foundation", f"{len(foundation_valid):,}", f"{UPSAMPLE_FOUNDATION}x", f"{len(foundation_up):,}"],
            ["TOTAL", f"{len(customs_valid)+len(behavioral_valid)+len(categories_valid)+len(foundation_valid):,}", "", f"{total:,}"],
        ],
    )

    # Action-type distribution
    action_order = ["call_tool", "clarify", "escalate", "answer_directly", "refuse"]
    action_rows = []
    for act in action_order:
        count = action_counts.get(act, 0)
        pct = count / total * 100 if total else 0
        action_rows.append([act, f"{count:,}", f"{pct:.1f}%"])
    # Other actions not in the standard list
    other_count = sum(c for a, c in action_counts.items() if a not in action_order)
    if other_count > 0:
        other_pct = other_count / total * 100 if total else 0
        action_rows.append(["other", f"{other_count:,}", f"{other_pct:.1f}%"])
    action_rows.append(["TOTAL", f"{total:,}", "100.0%"])

    print_table(
        "Action-Type Distribution",
        ["Action Type", "Count", "Pct"],
        action_rows,
    )

    # Category distribution (top 25 + others)
    cat_sorted = category_counts.most_common()
    cat_rows = []
    shown = 0
    other_cat_count = 0
    for cat, count in cat_sorted:
        if shown < 25:
            pct = count / total * 100 if total else 0
            cat_rows.append([cat, f"{count:,}", f"{pct:.1f}%"])
            shown += 1
        else:
            other_cat_count += count
    if other_cat_count > 0:
        pct = other_cat_count / total * 100 if total else 0
        cat_rows.append([f"... ({len(cat_sorted) - 25} more)", f"{other_cat_count:,}", f"{pct:.1f}%"])

    print_table(
        "Category Distribution (top 25)",
        ["Category", "Count", "Pct"],
        cat_rows,
    )

    # Token statistics
    print_table(
        "Token Statistics (approx, chars/4)",
        ["Metric", "Value"],
        [
            ["Mean tokens/sample", f"{avg_tokens:,.0f}"],
            ["Median tokens/sample", f"{median_tokens:,}"],
            ["P5 tokens/sample", f"{p5_tokens:,}"],
            ["P95 tokens/sample", f"{p95_tokens:,}"],
            ["Min tokens/sample", f"{min_tokens:,}"],
            ["Max tokens/sample", f"{max_tokens:,}"],
            ["Total tokens (all)", f"{total_tokens:,}"],
        ],
    )

    # ── 9. Write output ──────────────────────────────────────────────────

    if dry_run:
        print("\n[DRY RUN] Skipping file writes.")
        return

    print(f"\nWriting train set to {OUTPUT_TRAIN.name}...")
    write_jsonl_safe(train_samples, OUTPUT_TRAIN)
    print(f"  Written: {len(train_samples):,} samples")

    print(f"Writing val set to {OUTPUT_VAL.name}...")
    write_jsonl_safe(val_samples, OUTPUT_VAL)
    print(f"  Written: {len(val_samples):,} samples")

    print(f"\n{'=' * 70}")
    print("V3 DATASET ASSEMBLY COMPLETE")
    print(f"  Train: {OUTPUT_TRAIN}  ({len(train_samples):,} samples)")
    print(f"  Val:   {OUTPUT_VAL}  ({len(val_samples):,} samples)")
    print(f"  Total: {total:,} effective samples")
    print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="V3 Dataset Assembly — assemble all training data sources"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and report statistics without writing output files",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
