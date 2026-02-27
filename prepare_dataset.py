"""
aiqarus-agent-4b: Dataset Preparation
======================================
Downloads, filters, normalizes, and merges all three training layers.

Layer 1a — Foundation (~15K samples)
  Source: vericava/sft-tool-calling-structured-output-v1
  Content: Basic tool-calling mechanics (OpenAI messages format)
  Filter:  English-only, must have tool calls, diverse difficulty

Layer 1b — Verified function calls (~5K samples)
  Source: Salesforce/xlam-function-calling-60k (gated, requires HF_TOKEN)
  Content: 60K verified function calls (format + execution + semantic check)
  Filter:  English-only, single query → tool call mapping

Layer 2 — Reasoning (~5K samples)
  Source: interstellarninja/hermes_reasoning_tool_use
  Content: Tool calls with <think> reasoning blocks (ShareGPT format)
  Filter:  Must have <think> block, must have tool calls

Layer 3 — Aiqarus-specific (~1,642 samples)
  Source: data/layer3_scored.jsonl (local)
  Content: Enterprise scenarios, all 8 agent categories
  Filter:  quality_score.avg >= 4.0

Output:
  data/layer1_filtered.jsonl    — normalized Layer 1a (vericava)
  data/layer1b_filtered.jsonl   — normalized Layer 1b (Salesforce)
  data/layer2_filtered.jsonl    — normalized Layer 2 (hermes)
  data/train.jsonl              — 95% merged & shuffled
  data/val.jsonl                — 5% held-out

Normalized message format (all layers):
  [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "<think>...</think>\\n\\n{JSON tool call}"},
    {"role": "tool",      "content": "{JSON result}"},
    {"role": "assistant", "content": "Final response to user"}
  ]

Usage:
  python prepare_dataset.py
  python prepare_dataset.py --layer1-count 10000  # fewer Layer 1 samples
  python prepare_dataset.py --layer2-count 3000   # fewer Layer 2 samples
  python prepare_dataset.py --skip-layer1         # skip Layer 1a (use cached)
  python prepare_dataset.py --skip-layer2         # skip Layer 2 (use cached)
  python prepare_dataset.py --salesforce-only     # only re-run Salesforce step
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).parent / ".env")

OUTPUT_DIR = Path("data")
L1_FILE    = OUTPUT_DIR / "layer1_filtered.jsonl"
L1B_FILE   = OUTPUT_DIR / "layer1b_filtered.jsonl"
L2_FILE    = OUTPUT_DIR / "layer2_filtered.jsonl"
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE   = OUTPUT_DIR / "val.jsonl"
L3_FILE    = OUTPUT_DIR / "layer3_scored.jsonl"

SEED = 42
VAL_RATIO = 0.05

DEFAULT_L1_COUNT  = 15_000
DEFAULT_L1B_COUNT = 5_000     # Salesforce xlam
DEFAULT_L2_COUNT  = 5_000

SALESFORCE_SYSTEM = (
    "You are an AI assistant with access to tools. "
    "Use the available tools to answer the user's request accurately."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def strip_tool_response_tags(text: str) -> str:
    """Remove <tool_response> / </tool_response> wrapper from hermes tool turns."""
    text = text.strip()
    text = re.sub(r"^<tool_response>\s*", "", text)
    text = re.sub(r"\s*</tool_response>\s*$", "", text)
    return text.strip()


def strip_tool_call_tags(text: str) -> str:
    """Convert <tool_call>{...}</tool_call> → plain {JSON} to match Layer 3 format."""
    text = re.sub(r"<tool_call>\s*", "", text)
    text = re.sub(r"\s*</tool_call>", "", text)
    return text.strip()


def has_think_block(text: str) -> bool:
    return "<think>" in text and "</think>" in text


def think_block_length(text: str) -> int:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return len(m.group(1).strip()) if m else 0


def is_likely_english(text: str) -> bool:
    """Reject if ANY CJK (Japanese/Chinese/Korean) characters are present."""
    return not any(
        '\u3000' <= c <= '\u9fff'   # CJK unified, hiragana, katakana, etc.
        or '\uf900' <= c <= '\ufaff'  # CJK compatibility
        for c in text
    )


def convert_tool_calls_to_text(tool_calls: list) -> str:
    """Convert OpenAI-style tool_calls list to plain JSON text (Layer 3 format)."""
    if not tool_calls:
        return ""
    tc = tool_calls[0]  # take first call; multi-tool handled per-message
    fn = tc.get("function", {})
    name = fn.get("name", "")
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    except (json.JSONDecodeError, TypeError):
        args = args_raw
    return json.dumps({"name": name, "arguments": args})


# ---------------------------------------------------------------------------
# Layer 1: vericava — normalize OpenAI messages + target → messages list
# ---------------------------------------------------------------------------
def normalize_l1(row: dict, idx: int) -> dict | None:
    messages_raw = row.get("messages") or []
    target = row.get("target") or {}

    # Must be English
    all_text = " ".join(
        str(m.get("content") or "") for m in messages_raw
    ) + str(target.get("content") or "")
    if not is_likely_english(all_text):
        return None

    # Must have at least one tool call (in messages or target)
    has_tool = bool(target.get("tool_calls")) or any(
        bool(m.get("tool_calls")) for m in messages_raw
    )
    if not has_tool:
        return None

    messages = []
    for msg in messages_raw:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []

        if role == "assistant" and tool_calls:
            content = convert_tool_calls_to_text(tool_calls)
        elif content == "None" or content is None:
            content = ""

        if role and content.strip():
            messages.append({"role": role, "content": str(content).strip()})
        elif role == "tool":
            # Tool result messages may have empty content — skip
            if content.strip():
                messages.append({"role": role, "content": str(content).strip()})

    # Add target as final assistant turn
    t_content = str(target.get("content") or "").strip()
    t_calls = target.get("tool_calls") or []
    if t_calls:
        t_content = convert_tool_calls_to_text(t_calls)

    if t_content:
        messages.append({"role": target.get("role", "assistant"), "content": t_content})

    # Minimum viable conversation: system/user + assistant
    roles = [m["role"] for m in messages]
    if "user" not in roles or "assistant" not in roles:
        return None
    if len(messages) < 2:
        return None

    return {
        "id": f"l1_{idx:05d}",
        "layer": 1,
        "source": "vericava/sft-tool-calling-structured-output-v1",
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Layer 2: hermes — normalize ShareGPT format → messages list
# ---------------------------------------------------------------------------
def normalize_l2(row: dict, idx: int) -> dict | None:
    conversations = row.get("conversations") or []

    role_map = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}
    messages = []

    has_think = False
    has_tool_call = False

    for turn in conversations:
        from_role = turn.get("from", "")
        value = turn.get("value", "") or ""
        role = role_map.get(from_role, from_role)

        if role == "assistant":
            if has_think_block(value):
                has_think = True
            # Normalize <tool_call> tags → plain JSON
            if "<tool_call>" in value:
                has_tool_call = True
                value = strip_tool_call_tags(value)
        elif role == "tool":
            # Strip <tool_response> wrapper
            value = strip_tool_response_tags(value)

        if value.strip():
            messages.append({"role": role, "content": value.strip()})

    # Quality filters
    if not has_think:
        return None
    if not has_tool_call:
        return None
    if len(messages) < 3:
        return None

    # Check think block is substantive (>100 chars)
    for msg in messages:
        if msg["role"] == "assistant" and has_think_block(msg["content"]):
            if think_block_length(msg["content"]) < 100:
                return None
            break

    return {
        "id": f"l2_{idx:05d}",
        "layer": 2,
        "source": "interstellarninja/hermes_reasoning_tool_use",
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Layer 1b: Salesforce xlam — single-turn verified function calls
# ---------------------------------------------------------------------------
def normalize_salesforce(row: dict, idx: int) -> dict | None:
    query = str(row.get("query") or "").strip()
    answers_raw = row.get("answers") or "[]"
    tools_raw   = row.get("tools")   or "[]"

    if not query or not is_likely_english(query):
        return None

    # Parse answers (list of {name, arguments} dicts)
    try:
        answers = json.loads(answers_raw) if isinstance(answers_raw, str) else answers_raw
    except (json.JSONDecodeError, TypeError):
        return None
    if not answers:
        return None

    # Parse tools
    try:
        tools = json.loads(tools_raw) if isinstance(tools_raw, str) else tools_raw
    except (json.JSONDecodeError, TypeError):
        tools = []

    # Build assistant content — single call or parallel calls as JSON array
    if len(answers) == 1:
        call = answers[0]
        assistant_content = json.dumps({
            "name": call.get("name", ""),
            "arguments": call.get("arguments", {})
        })
    else:
        # Parallel tool calls — format as array
        assistant_content = json.dumps([
            {"name": c.get("name", ""), "arguments": c.get("arguments", {})}
            for c in answers
        ])

    if not assistant_content:
        return None

    messages = [
        {"role": "system",    "content": SALESFORCE_SYSTEM},
        {"role": "user",      "content": query},
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "id":      f"l1b_{idx:05d}",
        "layer":   1,
        "source":  "Salesforce/xlam-function-calling-60k",
        "tools":   tools,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Layer 3: local scored JSONL — filter by quality score
# ---------------------------------------------------------------------------
def load_layer3(min_avg: float = 4.0) -> list[dict]:
    if not L3_FILE.exists():
        print(f"WARNING: {L3_FILE} not found — skipping Layer 3")
        return []

    samples = []
    skipped = 0
    with open(L3_FILE) as f:
        for line in f:
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = s.get("quality_score") or {}
            avg = score.get("avg", 0)
            if avg < min_avg:
                skipped += 1
                continue
            # Normalize: ensure layer field present, strip extra fields for training
            out = {
                "id": s.get("id", f"l3_{len(samples):04d}"),
                "layer": 3,
                "source": "aiqarus_layer3",
                "messages": s.get("messages", []),
            }
            if out["messages"]:
                samples.append(out)

    print(f"Layer 3: loaded {len(samples)}, skipped {skipped} (avg < {min_avg})")
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare merged training dataset")
    parser.add_argument("--layer1-count", type=int, default=DEFAULT_L1_COUNT,
                        help=f"Max Layer 1 samples (default: {DEFAULT_L1_COUNT})")
    parser.add_argument("--layer2-count", type=int, default=DEFAULT_L2_COUNT,
                        help=f"Max Layer 2 samples (default: {DEFAULT_L2_COUNT})")
    parser.add_argument("--salesforce-count", type=int, default=DEFAULT_L1B_COUNT,
                        help=f"Max Salesforce xlam samples (default: {DEFAULT_L1B_COUNT})")
    parser.add_argument("--skip-layer1", action="store_true",
                        help="Skip Layer 1a download (use existing layer1_filtered.jsonl)")
    parser.add_argument("--skip-layer1b", action="store_true",
                        help="Skip Layer 1b Salesforce download (use existing layer1b_filtered.jsonl)")
    parser.add_argument("--skip-layer2", action="store_true",
                        help="Skip Layer 2 download (use existing layer2_filtered.jsonl)")
    parser.add_argument("--salesforce-only", action="store_true",
                        help="Only run the Salesforce step (implies --skip-layer1 --skip-layer2)")
    parser.add_argument("--min-quality", type=float, default=4.0,
                        help="Minimum Layer 3 quality score avg (default: 4.0)")
    args = parser.parse_args()

    if args.salesforce_only:
        args.skip_layer1 = True
        args.skip_layer2 = True

    random.seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Set HF token for gated datasets
    hf_token = os.environ.get("HF_TOKEN")

    all_samples = []

    # -------------------------------------------------------------------
    # Layer 1a — vericava
    # -------------------------------------------------------------------
    if args.skip_layer1 and L1_FILE.exists():
        print(f"Layer 1: loading from {L1_FILE}")
        l1_samples = []
        with open(L1_FILE) as f:
            for line in f:
                try:
                    l1_samples.append(json.loads(line))
                except Exception:
                    pass
        print(f"Layer 1: {len(l1_samples)} samples loaded")
    else:
        print(f"\n--- Layer 1: vericava (downloading ~490K rows, sampling {args.layer1_count}) ---")
        ds1 = load_dataset("vericava/sft-tool-calling-structured-output-v1", split="train")
        print(f"Downloaded {len(ds1)} rows. Filtering and normalizing...")

        l1_samples = []
        idx = 0
        # Shuffle indices for diversity
        indices = list(range(len(ds1)))
        random.shuffle(indices)

        for i in tqdm(indices, desc="Layer 1"):
            row = ds1[i]
            normalized = normalize_l1(row, idx)
            if normalized:
                l1_samples.append(normalized)
                idx += 1
                if len(l1_samples) >= args.layer1_count:
                    break

        print(f"Layer 1: {len(l1_samples)} valid samples from {len(ds1)} total")
        with open(L1_FILE, "w") as f:
            for s in l1_samples:
                f.write(json.dumps(s) + "\n")
        print(f"Saved → {L1_FILE}")

    all_samples.extend(l1_samples)

    # -------------------------------------------------------------------
    # Layer 1b — Salesforce xlam (verified function calls, gated)
    # -------------------------------------------------------------------
    if args.skip_layer1b and L1B_FILE.exists():
        print(f"\nLayer 1b (Salesforce): loading from {L1B_FILE}")
        l1b_samples = []
        with open(L1B_FILE) as f:
            for line in f:
                try:
                    l1b_samples.append(json.loads(line))
                except Exception:
                    pass
        print(f"Layer 1b: {len(l1b_samples)} samples loaded")
    else:
        print(f"\n--- Layer 1b: Salesforce/xlam-function-calling-60k (60K rows, sampling {args.salesforce_count}) ---")
        if not hf_token:
            print("WARNING: HF_TOKEN not set — Salesforce dataset is gated. Skipping.")
            l1b_samples = []
        else:
            ds1b = load_dataset("Salesforce/xlam-function-calling-60k", split="train",
                                token=hf_token)
            print(f"Downloaded {len(ds1b)} rows. Filtering and normalizing...")

            l1b_candidates = []
            for i, row in enumerate(tqdm(ds1b, desc="Layer 1b")):
                normalized = normalize_salesforce(row, i)
                if normalized:
                    l1b_candidates.append(normalized)

            print(f"Layer 1b: {len(l1b_candidates)} valid from {len(ds1b)} total")
            random.shuffle(l1b_candidates)
            l1b_samples = l1b_candidates[:args.salesforce_count]
            for i, s in enumerate(l1b_samples):
                s["id"] = f"l1b_{i:05d}"

            print(f"Layer 1b: using {len(l1b_samples)} samples")
            with open(L1B_FILE, "w") as f:
                for s in l1b_samples:
                    f.write(json.dumps(s) + "\n")
            print(f"Saved → {L1B_FILE}")

    all_samples.extend(l1b_samples)

    # -------------------------------------------------------------------
    # Layer 2
    # -------------------------------------------------------------------
    if args.skip_layer2 and L2_FILE.exists():
        print(f"\nLayer 2: loading from {L2_FILE}")
        l2_samples = []
        with open(L2_FILE) as f:
            for line in f:
                try:
                    l2_samples.append(json.loads(line))
                except Exception:
                    pass
        print(f"Layer 2: {len(l2_samples)} samples loaded")
    else:
        print(f"\n--- Layer 2: hermes_reasoning_tool_use (downloading ~51K rows, sampling {args.layer2_count}) ---")
        ds2 = load_dataset("interstellarninja/hermes_reasoning_tool_use", split="train")
        print(f"Downloaded {len(ds2)} rows. Filtering and normalizing...")

        l2_candidates = []
        for i, row in enumerate(tqdm(ds2, desc="Layer 2")):
            normalized = normalize_l2(row, i)
            if normalized:
                l2_candidates.append(normalized)

        print(f"Layer 2: {len(l2_candidates)} valid samples from {len(ds2)} total")

        # Sample down to target count
        random.shuffle(l2_candidates)
        l2_samples = l2_candidates[:args.layer2_count]
        # Re-index
        for i, s in enumerate(l2_samples):
            s["id"] = f"l2_{i:05d}"

        print(f"Layer 2: using {len(l2_samples)} samples")
        with open(L2_FILE, "w") as f:
            for s in l2_samples:
                f.write(json.dumps(s) + "\n")
        print(f"Saved → {L2_FILE}")

    all_samples.extend(l2_samples)

    # -------------------------------------------------------------------
    # Layer 3
    # -------------------------------------------------------------------
    print(f"\n--- Layer 3: aiqarus custom (local, min_quality={args.min_quality}) ---")
    l3_samples = load_layer3(min_avg=args.min_quality)
    all_samples.extend(l3_samples)

    # -------------------------------------------------------------------
    # Merge, shuffle, split
    # -------------------------------------------------------------------
    print(f"\n--- Merging ---")
    print(f"  Layer 1a (vericava):    {len(l1_samples):,}")
    print(f"  Layer 1b (Salesforce):  {len(l1b_samples):,}")
    print(f"  Layer 2  (hermes):      {len(l2_samples):,}")
    print(f"  Layer 3  (aiqarus):     {len(l3_samples):,}")
    print(f"  Total:                  {len(all_samples):,}")

    random.shuffle(all_samples)

    n_val = max(1, int(len(all_samples) * VAL_RATIO))
    n_train = len(all_samples) - n_val
    train_samples = all_samples[:n_train]
    val_samples   = all_samples[n_train:]

    with open(TRAIN_FILE, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(VAL_FILE, "w") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    print(f"\n{'='*50}")
    print(f"Train: {len(train_samples):,} samples → {TRAIN_FILE}")
    print(f"Val:   {len(val_samples):,} samples  → {VAL_FILE}")

    # Layer distribution in train set
    layer_dist = {}
    for s in train_samples:
        l = s.get("layer", "?")
        layer_dist[l] = layer_dist.get(l, 0) + 1
    print(f"\nTrain layer distribution:")
    for l, n in sorted(layer_dist.items()):
        pct = n / len(train_samples) * 100
        print(f"  Layer {l}: {n:,} ({pct:.1f}%)")

    print(f"\nDataset ready. Next: push to Modal volume with:")
    print(f"  modal volume put aiqarus-data data/train.jsonl dataset/")
    print(f"  modal volume put aiqarus-data data/val.jsonl dataset/")


if __name__ == "__main__":
    main()
