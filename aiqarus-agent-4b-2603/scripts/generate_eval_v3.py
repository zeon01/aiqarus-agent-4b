#!/usr/bin/env python3
"""
V3 Eval Case Generator
======================
Generates 1,400 multi-turn eval test cases across 14 categories
using Gemini 3.1 Pro Preview via CLI.

Features:
- Randomized tool schemas from the 799 training + 72 held-out library
- 70/30 known/novel schema split per category
- Batched generation (5 cases per CLI call)
- Shard support for parallel execution
- Resume-safe (tracks generated IDs)
- 40+ enterprise contexts, rotated per batch

Usage:
  # Full run (all 14 categories)
  python scripts/generate_eval_v3.py

  # Single category
  python scripts/generate_eval_v3.py --category over_execution

  # Parallel shards
  python scripts/generate_eval_v3.py --shard 1/3
  python scripts/generate_eval_v3.py --shard 2/3
  python scripts/generate_eval_v3.py --shard 3/3

  # Smoke test
  python scripts/generate_eval_v3.py --limit 5
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # aiqarus-agent-4b-2603/
SCHEMA_DIR = PROJECT_ROOT / "data" / "v3" / "tool_schemas"
EVAL_DIR = PROJECT_ROOT / "data" / "v3" / "eval_cases"
PROMPT_FILE = PROJECT_ROOT / "data" / "v3" / "eval_prompt.md"

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL = "gemini-3.1-pro-preview"
BATCH_SIZE = 5
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Categories: 100 cases each = 1,400 total
# ---------------------------------------------------------------------------
CATEGORIES = [
    "multi_step_chaining",
    "scope_creep",
    "error_recovery",
    "clarification_loop",
    "tool_result_injection",
    "over_execution",
    "tool_loop_prevention",
    "clarification_follow_through",
    "handoff_routing",
    "pii_data_sensitivity",
    "permission_verification",
    "correction_handling",
    "multi_turn_context",
    "tool_chain_trajectories",
]

CASES_PER_CATEGORY = 100
KNOWN_CASES = 70  # first 70 use training schemas
NOVEL_CASES = 30  # last 30 use held-out schemas

# ---------------------------------------------------------------------------
# Enterprise contexts (40+)
# ---------------------------------------------------------------------------
ENTERPRISE_CONTEXTS = [
    "a B2B SaaS company with 200 employees",
    "a fintech startup processing payments for SMBs",
    "a healthcare data platform operating under HIPAA compliance",
    "an e-commerce company managing peak season operations",
    "a consulting firm managing multiple client engagements",
    "a manufacturing company tracking supply chain and inventory",
    "a legal tech company managing contracts and compliance",
    "an HR tech platform serving enterprise clients",
    "a cybersecurity firm responding to client incidents",
    "a media company managing content workflows and licensing",
    "a logistics company coordinating freight and delivery",
    "a real estate brokerage managing listings and closings",
    "a pharmaceutical company tracking clinical trials",
    "an insurance company processing claims and policies",
    "a telecommunications provider managing network operations",
    "a government contractor handling procurement and grants",
    "an educational institution managing student enrollment",
    "a retail chain managing multi-store inventory",
    "an energy company monitoring utility infrastructure",
    "a nonprofit managing donor relations and grants",
    "a banking institution handling consumer accounts",
    "an automotive company managing dealership operations",
    "a hospitality chain managing reservations and guest services",
    "a construction company tracking projects and subcontractors",
    "an advertising agency managing campaigns and client budgets",
    "a food & beverage company managing supply chain compliance",
    "a defense contractor with strict clearance requirements",
    "a biotech startup managing research pipelines",
    "a transportation company managing fleet operations",
    "a professional services firm managing billing and engagements",
    "a venture capital firm managing portfolio companies",
    "a sports organization managing ticketing and fan engagement",
    "a fashion brand managing global supply chain and retail",
    "an agricultural company managing crop and equipment data",
    "a mining company managing safety compliance and operations",
    "a water utility managing infrastructure and billing",
    "a gaming company managing player accounts and transactions",
    "a recruitment firm managing candidate pipelines",
    "a coworking space managing memberships and bookings",
    "a waste management company tracking routes and compliance",
    "an aviation company managing flight operations and maintenance",
    "a dental practice management platform",
]

# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------
def load_schemas():
    """Load training and held-out schemas."""
    with open(SCHEMA_DIR / "schemas.json") as f:
        training = json.load(f)
    with open(SCHEMA_DIR / "held_out_schemas.json") as f:
        held_out = json.load(f)
    return training, held_out


def sample_tools(schemas: list, n: int = 3, rng: random.Random = None) -> list:
    """Sample n tools from schema list, ensuring domain diversity."""
    if rng is None:
        rng = random.Random()

    if len(schemas) <= n:
        return schemas[:n]

    # Try to pick from different domains
    by_domain = {}
    for s in schemas:
        d = s.get("domain", "unknown")
        by_domain.setdefault(d, []).append(s)

    selected = []
    domains = list(by_domain.keys())
    rng.shuffle(domains)

    for domain in domains:
        if len(selected) >= n:
            break
        tool = rng.choice(by_domain[domain])
        selected.append(tool)

    # Fill remaining if needed
    remaining = [s for s in schemas if s not in selected]
    while len(selected) < n and remaining:
        pick = rng.choice(remaining)
        selected.append(pick)
        remaining.remove(pick)

    return selected[:n]


def format_tools_for_prompt(tools: list) -> str:
    """Format tools as JSON for inclusion in the prompt."""
    # Strip domain/source metadata, keep only what model sees
    clean = []
    for t in tools:
        clean.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        })
    return json.dumps(clean, indent=2)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_prompt(category: str, batch_size: int, start_id: int,
                 tools: list, schema_type: str, context: str) -> str:
    """Build the full generation prompt."""
    template = PROMPT_FILE.read_text()

    tools_json = format_tools_for_prompt(tools)

    prompt = template.replace("{batch_size}", str(batch_size))
    prompt = prompt.replace("{category}", category)
    prompt = prompt.replace("{start_id:04d}", f"{start_id:04d}")
    prompt = prompt.replace("{schema_type}", schema_type)
    prompt = prompt.replace("{enterprise_context}", context)
    prompt = prompt.replace("{tools_json}", tools_json)

    return prompt


# ---------------------------------------------------------------------------
# Gemini CLI invocation
# ---------------------------------------------------------------------------
def call_gemini(prompt: str, expected_count: int) -> list[dict]:
    """Call Gemini CLI and parse JSON lines response."""
    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                ["gemini", "-m", MODEL, "-s", "false", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=300,  # 5 min for batch of 5
            )

            output = result.stdout.strip()
            stderr = result.stderr.strip()

            # Detect rate limit
            stderr_lower = stderr.lower()
            rate_limited = any(phrase in stderr_lower for phrase in [
                "rate limit", "rate_limit", "ratelimit",
                "quota exceeded", "quota_exceeded",
                "resource exhausted", "resource_exhausted",
                "too many requests", "429",
                "exhausted your capacity",
            ])
            if rate_limited:
                wait = 60 * (attempt + 1)
                print(f"\n    RATE LIMITED — waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue

            if not output:
                if attempt < MAX_RETRIES - 1:
                    print(f"    Empty output (attempt {attempt+1}), retrying...", flush=True)
                    time.sleep(5)
                    continue
                return []

            # Strip markdown fences if present
            if output.startswith("```"):
                output = output.split("\n", 1)[1] if "\n" in output else output[3:]
            if output.endswith("```"):
                output = output[:-3]
            output = output.strip()
            if output.startswith("json"):
                output = output[4:].strip()

            # Parse JSON objects — they may span multiple lines
            results = []
            # Try line-by-line first
            lines = output.split("\n")

            # Accumulate potential multi-line JSON
            buffer = ""
            brace_depth = 0
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue

                buffer += stripped + " "
                brace_depth += stripped.count("{") - stripped.count("}")

                if brace_depth == 0 and buffer.strip():
                    try:
                        start = buffer.find("{")
                        end = buffer.rfind("}") + 1
                        if start >= 0 and end > start:
                            obj = json.loads(buffer[start:end])
                            results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    buffer = ""
                elif brace_depth < 0:
                    # Reset on broken state
                    buffer = ""
                    brace_depth = 0

            if results:
                return results

            # Fallback: try parsing entire output as JSON array
            try:
                arr = json.loads(output)
                if isinstance(arr, list):
                    return arr
            except json.JSONDecodeError:
                pass

            if attempt < MAX_RETRIES - 1:
                print(f"    Parse failed (attempt {attempt+1}), retrying...", flush=True)
                time.sleep(5)
                continue

            # Save raw output for debugging and recovery
            raw_file = EVAL_DIR / "_raw_responses.jsonl"
            with open(raw_file, "a") as f:
                f.write(json.dumps({"_raw": output, "_parse_error": True,
                                    "_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return [{"_raw": output[:2000], "_parse_error": True}]

        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRIES - 1:
                print(f"    Timeout (attempt {attempt+1}), retrying...", flush=True)
                time.sleep(5)
                continue
            return []
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
                continue
            print(f"    Error: {e}", file=sys.stderr)
            return []

    return []


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def load_existing_ids(output_file: Path) -> set:
    """Load IDs of already-generated cases for resume."""
    ids = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "id" in obj:
                        ids.add(obj["id"])
                except json.JSONDecodeError:
                    continue
    return ids


# ---------------------------------------------------------------------------
# Shard support
# ---------------------------------------------------------------------------
def get_shard_categories(shard_str: str) -> list[str]:
    """Split categories across shards."""
    if not shard_str:
        return CATEGORIES

    parts = shard_str.split("/")
    shard_num = int(parts[0])
    total_shards = int(parts[1])

    # Distribute categories round-robin
    return [cat for i, cat in enumerate(CATEGORIES) if (i % total_shards) + 1 == shard_num]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_case(case: dict) -> tuple[bool, str]:
    """Basic validation of a generated eval case."""
    if "_parse_error" in case:
        return False, "parse_error"

    required = ["id", "category", "turns", "expected_action_type", "rubric_notes"]
    for field in required:
        if field not in case:
            return False, f"missing_{field}"

    if not isinstance(case.get("turns"), list) or len(case["turns"]) < 2:
        return False, "insufficient_turns"

    valid_actions = {"call_tool", "clarify", "escalate", "refuse", "answer_directly"}
    if case.get("expected_action_type") not in valid_actions:
        return False, f"invalid_action_type: {case.get('expected_action_type')}"

    if not case.get("rubric_notes") or len(case["rubric_notes"]) < 20:
        return False, "rubric_notes_too_short"

    return True, "ok"


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate_category(category: str, training_schemas: list, held_out_schemas: list,
                      limit: int = None):
    """Generate all eval cases for one category."""
    output_file = EVAL_DIR / f"{category}.jsonl"
    malformed_file = EVAL_DIR / f"{category}.malformed.jsonl"

    existing_ids = load_existing_ids(output_file)
    total_target = min(CASES_PER_CATEGORY, limit) if limit else CASES_PER_CATEGORY

    # Count existing valid cases
    existing_count = len(existing_ids)
    if existing_count >= total_target:
        print(f"  [{category}] Already have {existing_count}/{total_target}. Skipping.")
        return existing_count

    remaining = total_target - existing_count
    print(f"  [{category}] Have {existing_count}/{total_target}. Generating {remaining} more...")

    # Shuffle enterprise contexts for rotation
    rng = random.Random(hash(category) + 42)
    contexts = list(ENTERPRISE_CONTEXTS)
    rng.shuffle(contexts)
    context_idx = 0

    generated = existing_count
    batch_num = 0
    consecutive_empty = 0
    MAX_CONSECUTIVE_EMPTY = 5

    while generated < total_target:
        batch_size = min(BATCH_SIZE, total_target - generated)
        start_id = generated + 1

        # Determine schema type: 70% known, 30% novel (scaled to total_target)
        known_boundary = int(total_target * 0.7)
        if generated < known_boundary:
            schema_type = "known"
            schema_pool = training_schemas
        else:
            schema_type = "novel"
            schema_pool = held_out_schemas

        # Sample tools for this batch (2-4 per case, but same set for the batch prompt)
        n_tools = rng.randint(2, 4)
        tools = sample_tools(schema_pool, n=n_tools, rng=rng)

        # Rotate enterprise context
        context = contexts[context_idx % len(contexts)]
        context_idx += 1

        # Build and send prompt
        prompt = build_prompt(category, batch_size, start_id, tools, schema_type, context)

        print(f"    Batch {batch_num+1}: {category} #{start_id:04d}-{start_id+batch_size-1:04d} "
              f"({schema_type}, {n_tools} tools, {context[:40]}...)", flush=True)

        results = call_gemini(prompt, batch_size)

        if not results:
            consecutive_empty += 1
            print(f"    WARNING: Empty results for batch {batch_num+1} "
                  f"({consecutive_empty}/{MAX_CONSECUTIVE_EMPTY} consecutive)", flush=True)
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                print(f"    ERROR: {MAX_CONSECUTIVE_EMPTY} consecutive empty batches. "
                      f"Aborting category.", flush=True)
                break
            batch_num += 1
            time.sleep(2)
            continue

        valid_count = 0
        for case in results:
            is_valid, reason = validate_case(case)

            if is_valid:
                case_id = case.get("id", f"{category}_{start_id + valid_count:04d}")
                if case_id in existing_ids:
                    # Assign new unique ID
                    case_id = f"{category}_{generated + valid_count + 1:04d}"
                    case["id"] = case_id

                # Ensure metadata
                case["category"] = category
                case["schema_type"] = schema_type
                if "tools" not in case:
                    case["tools"] = [{"name": t["name"], "description": t["description"],
                                      "parameters": t["parameters"]} for t in tools]

                with open(output_file, "a") as f:
                    f.write(json.dumps(case, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                existing_ids.add(case_id)
                valid_count += 1
            else:
                case["_rejection_reason"] = reason
                case["_category"] = category
                with open(malformed_file, "a") as f:
                    f.write(json.dumps(case, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

        generated += valid_count
        batch_num += 1

        if valid_count > 0:
            consecutive_empty = 0
        else:
            consecutive_empty += 1
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                print(f"    ERROR: {MAX_CONSECUTIVE_EMPTY} consecutive empty batches. "
                      f"Aborting category.", flush=True)
                break

        if valid_count < batch_size:
            print(f"    Got {valid_count}/{batch_size} valid cases", flush=True)

        # Brief pause between batches
        time.sleep(1)

    print(f"  [{category}] Done: {generated} cases generated.")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate V3 eval cases")
    parser.add_argument("--shard", type=str, default=None, help="Shard spec: '1/3', '2/3', '3/3'")
    parser.add_argument("--category", type=str, default=None, help="Generate for single category")
    parser.add_argument("--limit", type=int, default=None, help="Max cases per category")
    args = parser.parse_args()

    # Load schemas
    print("Loading schema library...", flush=True)
    training_schemas, held_out_schemas = load_schemas()
    print(f"  Training: {len(training_schemas)} schemas, Held-out: {len(held_out_schemas)} schemas")

    # Determine categories to generate
    if args.category:
        if args.category not in CATEGORIES:
            print(f"ERROR: Unknown category '{args.category}'. Valid: {CATEGORIES}")
            sys.exit(1)
        categories = [args.category]
    else:
        categories = get_shard_categories(args.shard)

    print(f"\nCategories to generate: {categories}")
    if args.shard:
        print(f"Shard: {args.shard}")
    if args.limit:
        print(f"Limit: {args.limit} per category")

    # Create output directory
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Generate
    total = 0
    for category in categories:
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")
        count = generate_category(category, training_schemas, held_out_schemas, args.limit)
        total += count

    # Summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total cases: {total}")
    print(f"Categories: {len(categories)}")
    print(f"Output: {EVAL_DIR}/")

    # Per-category breakdown
    print(f"\nPer-category counts:")
    for cat in categories:
        cat_file = EVAL_DIR / f"{cat}.jsonl"
        if cat_file.exists():
            count = len(load_existing_ids(cat_file))
            malformed = EVAL_DIR / f"{cat}.malformed.jsonl"
            mal_count = sum(1 for line in open(malformed) if line.strip()) if malformed.exists() else 0
            print(f"  {cat}: {count} valid, {mal_count} malformed")


if __name__ == "__main__":
    main()
