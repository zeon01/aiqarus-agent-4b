#!/usr/bin/env python3
"""
Synonym Replacement Pipeline for R2 Custom Training Data.

Creates 3 copies per sample (3x upsample) with different tool name synonyms
to teach the model to read tool descriptions rather than memorize names.

Copy 0: Original tool names (no replacement)
Copy 1: Synonyms from index 0-3 (randomly selected per tool)
Copy 2: Synonyms from index 4-7 (randomly selected per tool)

Also slightly rephrases tool descriptions for synonym copies.

Usage:
    python scripts/synonym_replace.py
"""

import json
import re
import random
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "v3" / "custom_deduped.jsonl"
SYNONYM_FILE = PROJECT_ROOT / "data" / "v3" / "tool_synonyms.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "v3" / "custom_upsampled_synonymed.jsonl"

# Seed for reproducibility
random.seed(42)

# Description rephrase templates — prepend or append clauses to vary descriptions
DESCRIPTION_PREFIXES = [
    "Use this to ",
    "Call this endpoint to ",
    "Invoke to ",
    "This function will ",
    "Triggers an action to ",
    "Allows you to ",
    "Provides the ability to ",
    "Executes a request to ",
]

DESCRIPTION_SUFFIXES = [
    " and return the result",
    " from the connected system",
    " via the platform API",
    " using the integrated service",
    " in the target environment",
    " as requested",
    " through the backend service",
    " from the data store",
]


def load_synonyms(path):
    """Load the synonym mapping from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_samples(path):
    """Load JSONL samples."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_mapping(synonyms, copy_index, sample_seed):
    """
    Build a tool name -> synonym mapping for a given copy.

    copy_index 0: return identity mapping (original names)
    copy_index 1: pick from synonyms[0:4] for each tool
    copy_index 2: pick from synonyms[4:8] for each tool

    Uses sample_seed for deterministic but varied selection per sample.
    """
    if copy_index == 0:
        return {}  # No replacements

    rng = random.Random(sample_seed + copy_index * 1000)

    mapping = {}
    if copy_index == 1:
        start, end = 0, 4
    elif copy_index == 2:
        start, end = 4, 8
    else:
        return {}

    for orig_name, syns in synonyms.items():
        available = syns[start:end]
        if available:
            mapping[orig_name] = rng.choice(available)
        # If not enough synonyms, fall back to any available
        elif syns:
            mapping[orig_name] = rng.choice(syns)

    return mapping


def rephrase_description(original_desc, rng):
    """
    Slightly rephrase a tool description by prepending or appending a clause,
    or swapping common adjectives/verbs.
    """
    if not original_desc:
        return original_desc

    # 50% chance to modify
    action = rng.choice(["prefix", "suffix", "swap_verb", "none"])

    if action == "prefix":
        # Lowercase the first letter of original and prepend
        prefix = rng.choice(DESCRIPTION_PREFIXES)
        lower_desc = original_desc[0].lower() + original_desc[1:] if original_desc else original_desc
        # Remove trailing period if present before prepending
        lower_desc = lower_desc.rstrip(".")
        return prefix + lower_desc

    elif action == "suffix":
        desc = original_desc.rstrip(".")
        suffix = rng.choice(DESCRIPTION_SUFFIXES)
        return desc + suffix

    elif action == "swap_verb":
        # Swap common verbs at the start of descriptions
        swaps = {
            "Look up": "Search for",
            "Search": "Find",
            "Get": "Retrieve",
            "Retrieve": "Fetch",
            "Fetch": "Pull",
            "List": "Enumerate",
            "Execute": "Run",
            "Send": "Dispatch",
            "Create": "Generate",
            "Update": "Modify",
            "Check": "Verify",
            "Submit": "File",
        }
        for old_verb, new_verb in swaps.items():
            if original_desc.startswith(old_verb + " "):
                return new_verb + " " + original_desc[len(old_verb) + 1:]
            elif original_desc.startswith(old_verb.lower() + " "):
                return new_verb.lower() + " " + original_desc[len(old_verb) + 1:]
        return original_desc  # No swap matched

    else:
        return original_desc


def replace_tool_name_in_text(text, mapping):
    """
    Replace tool names in free text (e.g., assistant reasoning).
    Uses word-boundary-aware replacement to avoid partial matches.

    Sorts by length descending to prevent shorter names matching inside longer ones.
    """
    if not mapping or not text:
        return text

    # Sort by original name length descending to replace longer names first
    sorted_names = sorted(mapping.keys(), key=len, reverse=True)

    for orig_name in sorted_names:
        new_name = mapping[orig_name]
        # Use word boundary matching — tool names contain underscores and letters
        # Match the exact name surrounded by non-alphanumeric-underscore chars
        # or at string boundaries
        pattern = r'(?<![a-zA-Z0-9_])' + re.escape(orig_name) + r'(?![a-zA-Z0-9_])'
        text = re.sub(pattern, new_name, text)

    return text


def replace_tool_name_in_json_block(text, mapping):
    """
    Replace tool names specifically inside JSON blocks in assistant messages.
    Handles both pretty-printed and compact JSON containing "name": "tool_name".
    Also handles <tool_call> tags if present.
    """
    if not mapping or not text:
        return text

    # Replace "name": "tool_name" patterns in JSON (both styles of quotes/spacing)
    for orig_name, new_name in mapping.items():
        # Match "name": "tool_name" with various whitespace
        pattern = r'("name"\s*:\s*")' + re.escape(orig_name) + r'(")'
        text = re.sub(pattern, r'\g<1>' + new_name + r'\g<2>', text)

    return text


def apply_synonym_to_sample(sample, mapping, desc_rng):
    """
    Apply tool name synonym replacements to a single sample.
    Returns a new sample dict with replacements applied.
    """
    if not mapping:
        return sample  # Copy 0 — return as-is

    sample = json.loads(json.dumps(sample))  # Deep copy

    # 1. Replace in tools array
    if "tools" in sample:
        for tool in sample["tools"]:
            orig_name = tool.get("name", "")
            if orig_name in mapping:
                tool["name"] = mapping[orig_name]
                # Also rephrase the description
                if "description" in tool:
                    # Replace any self-references in description
                    tool["description"] = replace_tool_name_in_text(
                        tool["description"], mapping
                    )
                    tool["description"] = rephrase_description(
                        tool["description"], desc_rng
                    )

    # 2. Replace in messages
    if "messages" in sample:
        for msg in sample["messages"]:
            role = msg.get("role", "")

            if role == "tool":
                # Replace tool name in the "name" field
                if "name" in msg and msg["name"] in mapping:
                    msg["name"] = mapping[msg["name"]]
                # Tool content may reference tool names too
                if "content" in msg and isinstance(msg["content"], str):
                    msg["content"] = replace_tool_name_in_text(msg["content"], mapping)

            elif role == "assistant":
                content = msg.get("content", "")
                if not content:
                    continue

                # First: replace tool names in JSON "name" fields
                content = replace_tool_name_in_json_block(content, mapping)

                # Second: replace tool name references in reasoning/text
                # (e.g., "I will use `crm_lookup_customer`")
                content = replace_tool_name_in_text(content, mapping)

                msg["content"] = content

            elif role == "system":
                # System prompts may mention tool names
                content = msg.get("content", "")
                if content:
                    msg["content"] = replace_tool_name_in_text(content, mapping)

            elif role == "user":
                # User messages generally don't contain tool names,
                # but replace if present (rare edge case)
                content = msg.get("content", "")
                if content:
                    msg["content"] = replace_tool_name_in_text(content, mapping)

    return sample


def validate_sample(sample):
    """
    Validate that a sample has no orphaned tool references.
    Returns (is_valid, issues_list).
    """
    issues = []

    # Collect tool names from the tools array
    available_tools = set()
    if "tools" in sample:
        for tool in sample["tools"]:
            name = tool.get("name", "")
            if name:
                available_tools.add(name)

    if not available_tools:
        return True, []  # No tools to validate against

    # Check tool messages reference valid tools
    if "messages" in sample:
        for i, msg in enumerate(sample["messages"]):
            if msg.get("role") == "tool":
                tool_name = msg.get("name", "")
                if tool_name and tool_name not in available_tools:
                    issues.append(
                        f"Message {i}: tool role references '{tool_name}' "
                        f"not in tools array {available_tools}"
                    )

            # Check assistant messages for tool call JSON
            if msg.get("role") == "assistant":
                content = msg.get("content", "") or ""
                if not content:
                    continue
                # Find all "name": "..." patterns that look like tool calls
                # (inside JSON objects with "arguments")
                tool_call_names = re.findall(
                    r'"name"\s*:\s*"([^"]+)"', content
                )
                for tc_name in tool_call_names:
                    if tc_name in available_tools:
                        continue
                    # Check if this looks like a tool call (near "arguments")
                    # vs a regular data field (like a person's name)
                    # Heuristic: if "arguments" appears nearby, it's a tool call
                    idx = content.find(f'"{tc_name}"')
                    if idx >= 0:
                        context_window = content[max(0, idx - 200):idx + 200]
                        if '"arguments"' in context_window:
                            issues.append(
                                f"Message {i}: tool call references '{tc_name}' "
                                f"not in tools array"
                            )

    return len(issues) == 0, issues


def main():
    print("=" * 70)
    print("Synonym Replacement Pipeline")
    print("=" * 70)

    # Load data
    print(f"\nLoading samples from: {INPUT_FILE}")
    samples = load_samples(INPUT_FILE)
    print(f"  Loaded {len(samples)} samples")

    print(f"\nLoading synonyms from: {SYNONYM_FILE}")
    synonyms = load_synonyms(SYNONYM_FILE)
    print(f"  Loaded synonyms for {len(synonyms)} tool names")

    # Verify all tools in data have synonyms
    all_tools_in_data = set()
    for sample in samples:
        for tool in sample.get("tools", []):
            name = tool.get("name", "")
            if name:
                all_tools_in_data.add(name)

    missing_synonyms = all_tools_in_data - set(synonyms.keys())
    if missing_synonyms:
        print(f"\n  WARNING: {len(missing_synonyms)} tools have no synonyms:")
        for t in sorted(missing_synonyms):
            print(f"    - {t}")
    else:
        print(f"  All {len(all_tools_in_data)} tools in data have synonym entries")

    extra_synonyms = set(synonyms.keys()) - all_tools_in_data
    if extra_synonyms:
        print(f"  {len(extra_synonyms)} synonym entries not in data (unused): "
              f"{sorted(extra_synonyms)[:5]}...")

    # Check for synonym collisions
    print("\nChecking for synonym collisions...")
    all_syns = {}
    collision_count = 0
    for orig, syns in synonyms.items():
        for syn in syns:
            if syn in all_syns:
                print(f"  COLLISION: '{syn}' used for both '{orig}' and '{all_syns[syn]}'")
                collision_count += 1
            all_syns[syn] = orig
    if collision_count == 0:
        print("  No collisions found")

    # Generate 3 copies per sample
    print(f"\nGenerating 3x upsampled output...")
    output_samples = []
    category_counts = Counter()
    copy_counts = Counter()
    validation_failures = 0
    total_replacements_made = Counter()

    for sample_idx, sample in enumerate(samples):
        # Deterministic seed based on sample ID
        sample_id = sample.get("id", str(sample_idx))
        seed = int(hashlib.md5(sample_id.encode()).hexdigest()[:8], 16)

        for copy_idx in range(3):
            # Build mapping for this copy
            mapping = build_mapping(synonyms, copy_idx, seed)

            # Create description RNG
            desc_rng = random.Random(seed + copy_idx * 7777)

            # Apply synonyms
            new_sample = apply_synonym_to_sample(sample, mapping, desc_rng)

            # Update ID to distinguish copies
            if copy_idx > 0:
                new_sample["id"] = f"{sample_id}__syn{copy_idx}"
            # Copy 0 keeps original ID

            # Track which copy
            new_sample["synonym_copy"] = copy_idx

            # Validate
            is_valid, issues = validate_sample(new_sample)
            if not is_valid:
                validation_failures += 1
                if validation_failures <= 10:
                    print(f"  Validation issue in {new_sample['id']}:")
                    for issue in issues[:3]:
                        print(f"    {issue}")

            output_samples.append(new_sample)
            category_counts[new_sample.get("category", "unknown")] += 1
            copy_counts[copy_idx] += 1

            # Count replacements for stats
            if mapping:
                for orig_name in mapping:
                    # Check if this tool was actually in the sample
                    sample_tools = {t.get("name", "") for t in sample.get("tools", [])}
                    if orig_name in sample_tools:
                        total_replacements_made[orig_name] += 1

    # Write output
    print(f"\nWriting output to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        for sample in output_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Report
    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)

    print(f"\nInput:  {len(samples)} samples")
    print(f"Output: {len(output_samples)} samples (3x upsample)")
    print(f"Validation failures: {validation_failures}")

    print(f"\nBy copy:")
    for copy_idx in sorted(copy_counts.keys()):
        label = "original" if copy_idx == 0 else f"synonym set {copy_idx}"
        print(f"  Copy {copy_idx} ({label}): {copy_counts[copy_idx]}")

    print(f"\nBy category:")
    for cat in sorted(category_counts.keys()):
        orig = category_counts[cat] // 3  # Approximate original count
        print(f"  {cat:40s} {category_counts[cat]:6d}  (orig: {orig})")

    print(f"\nSynonym coverage (tools replaced across copy 1+2):")
    covered = sum(1 for v in total_replacements_made.values() if v > 0)
    print(f"  {covered}/{len(all_tools_in_data)} tools had at least one replacement applied")

    # Show top 10 most-replaced tools
    top_replaced = total_replacements_made.most_common(10)
    if top_replaced:
        print(f"\n  Top replaced tools:")
        for name, count in top_replaced:
            print(f"    {name:40s} {count:5d} replacements")

    # Spot-check: show a sample mapping
    print(f"\nSpot check — sample mapping for copy 1 (seed=42):")
    check_mapping = build_mapping(synonyms, 1, 42)
    for orig, syn in list(check_mapping.items())[:5]:
        print(f"  {orig:30s} -> {syn}")

    print(f"\nSpot check — sample mapping for copy 2 (seed=42):")
    check_mapping = build_mapping(synonyms, 2, 42)
    for orig, syn in list(check_mapping.items())[:5]:
        print(f"  {orig:30s} -> {syn}")

    print(f"\nDone. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
