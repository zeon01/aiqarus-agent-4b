"""
qa_score_v3.py — Batched QA scoring for V3 generated training data
===================================================================
Scores V3 training samples (foundation/behavioral/categories) and eval
cases using Gemini Flash via CLI. Supports batching (5 per call) and
sharding for parallel execution.

Target model: Qwen3.5-4B enterprise agent (aiqarus-agent-4b-2603)

Usage:
  # Score a single file
  python3 scripts/qa_score_v3.py --input data/v3/foundation/shard_1.jsonl

  # With explicit output and type
  python3 scripts/qa_score_v3.py --input data/v3/eval_cases/eval.jsonl --type eval

  # Parallel shards (run 3 in separate terminals)
  python3 scripts/qa_score_v3.py --input data/v3/foundation/shard_1.jsonl --shard 1/3
  python3 scripts/qa_score_v3.py --input data/v3/foundation/shard_1.jsonl --shard 2/3
  python3 scripts/qa_score_v3.py --input data/v3/foundation/shard_1.jsonl --shard 3/3
  # Then merge: cat data/v3/foundation/shard_1_scored_shard*.jsonl > data/v3/foundation/shard_1_scored.jsonl

  # Score ALL V3 generated data
  python3 scripts/qa_score_v3.py --score-all-v3

  # Quick test run
  python3 scripts/qa_score_v3.py --input data/v3/foundation/shard_1.jsonl --limit 10 --summary
"""

import argparse
import json
import os
import subprocess
import sys
import time
from glob import glob
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_BATCH_SIZE = 5
DEFAULT_MIN_SCORE = 3
MAX_RETRIES = 3
MAX_CONSECUTIVE_EMPTY = 5
GEMINI_TIMEOUT = 120  # seconds per CLI call


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def extract_assistant_content(messages: list) -> str:
    """Extract assistant and tool turns for the judge to evaluate."""
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            parts.append(f"[Assistant turn {len(parts)+1}]:\n{content}")
        elif role == "tool":
            c = content if isinstance(content, str) else json.dumps(content)
            parts.append(f"[Tool response]:\n{c[:500]}")
    return "\n\n".join(parts)


def build_training_prompt(batch: list[dict]) -> str:
    """Build a scoring prompt for training data (foundation/behavioral/categories)."""
    header = f"""You are a strict quality evaluator for AI agent training data.
These samples will train a Qwen3.5-4B enterprise agent model for tool-calling,
planning, routing, escalation, and adversarial robustness.

Score each of the {len(batch)} samples below on the following dimensions (1-5 each).

## Scoring Rubric

1. **score** (overall quality): How good is this as a training sample?
   1=Broken/harmful, 2=Poor, 3=Acceptable, 4=Good, 5=Excellent

2. **reasoning_quality**: Does the <think> block show genuine reasoning?
   1=Missing/empty, 2=Token ("I'll call the tool"), 3=Adequate, 4=Good (weighs alternatives), 5=Excellent (considers edge cases, explains why)

3. **tool_usage_correctness**: Are tool calls appropriate and correct?
   1=Wrong tool/hallucinated, 2=Right tool wrong args, 3=Correct but suboptimal, 4=Good, 5=Optimal tool selection and arguments
   For no-tool samples: score whether correctly NOT calling tools (5=correct refusal/escalation, 1=should have called a tool)

4. **think_block_quality**: Structural quality of the <think> block.
   1=Missing, 2=Present but trivial (<50 chars), 3=Adequate (states intent), 4=Good (multi-step reasoning), 5=Excellent (weighs tradeoffs, anticipates failures)

5. **action_type_match**: Does the action type in the response match what the scenario requires?
   true=The response takes the correct type of action (call tool / refuse / escalate / clarify)
   false=The response takes the wrong type of action (e.g., calls a tool when it should refuse)

Also list any specific **issues** found (empty list if none).

---

"""
    sample_sections = []
    for i, sample in enumerate(batch):
        messages = sample.get("messages", [])
        category = sample.get("category", "unknown")
        sample_id = sample.get("id", f"unknown_{i}")
        tools = sample.get("tools", [])

        system_prompt = ""
        user_input = ""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")[:500]
            elif msg.get("role") == "user":
                if not user_input:
                    user_input = msg.get("content", "")[:500]

        assistant_content = extract_assistant_content(messages)
        tool_names = [t.get("name", t.get("function", {}).get("name", "?")) for t in tools] if tools else []

        sample_sections.append(f"""=== SAMPLE {i+1}: {sample_id} (category: {category}) ===
Tools available: {json.dumps(tool_names)}
System: {system_prompt}
User: {user_input}
Response:
{assistant_content}
""")

    footer = f"""
---

Return EXACTLY {len(batch)} JSON objects, one per line, in order. No markdown fences, no extra text.
Each line must be valid JSON with this exact schema:
{{"id": "sample_id", "score": N, "reasoning_quality": N, "tool_usage_correctness": N, "action_type_match": true/false, "think_block_quality": N, "issues": ["issue1", "issue2"], "pass": true/false}}

pass = true when: score >= 3 AND action_type_match == true AND think_block_quality >= 2
"""
    return header + "\n".join(sample_sections) + footer


def build_eval_prompt(batch: list[dict]) -> str:
    """Build a scoring prompt for eval cases."""
    header = f"""You are a strict quality evaluator for AI agent EVALUATION test cases.
These cases will be used to evaluate a Qwen3.5-4B enterprise agent model.
Each case should present a clear scenario with an unambiguous expected behavior.

Score each of the {len(batch)} cases below on the following dimensions (1-5 each).

## Scoring Rubric

1. **score** (overall quality): How good is this as an eval test case?
   1=Broken/ambiguous, 2=Poor, 3=Acceptable, 4=Good, 5=Excellent

2. **rubric_clarity**: Is the scoring rubric / expected answer clear?
   1=No rubric, 2=Vague, 3=Adequate, 4=Clear, 5=Unambiguous with edge cases covered

3. **scenario_realism**: Is this a realistic enterprise scenario?
   1=Nonsensical, 2=Contrived, 3=Plausible, 4=Realistic, 5=Drawn from real-world patterns

4. **expected_behavior_clear**: Is it obvious what the correct model behavior should be?
   true=Yes, false=No / ambiguous

Also list any specific **issues** found (empty list if none).

---

"""
    sample_sections = []
    for i, case in enumerate(batch):
        case_id = case.get("id", case.get("case_id", f"unknown_{i}"))
        category = case.get("category", "unknown")
        scenario = case.get("scenario", case.get("prompt", ""))
        expected = case.get("expected_behavior", case.get("expected", case.get("rubric", "")))
        tools = case.get("tools", [])
        tool_names = [t.get("name", t.get("function", {}).get("name", "?")) for t in tools] if tools else []

        # Handle various eval case formats
        if isinstance(scenario, dict):
            scenario = json.dumps(scenario, indent=2)[:800]
        elif isinstance(scenario, str):
            scenario = scenario[:800]
        else:
            scenario = str(scenario)[:800]

        if isinstance(expected, dict):
            expected = json.dumps(expected, indent=2)[:400]
        elif isinstance(expected, str):
            expected = expected[:400]
        else:
            expected = str(expected)[:400]

        sample_sections.append(f"""=== CASE {i+1}: {case_id} (category: {category}) ===
Tools available: {json.dumps(tool_names)}
Scenario: {scenario}
Expected behavior: {expected}
""")

    footer = f"""
---

Return EXACTLY {len(batch)} JSON objects, one per line, in order. No markdown fences, no extra text.
Each line must be valid JSON with this exact schema:
{{"id": "case_id", "score": N, "rubric_clarity": N, "scenario_realism": N, "expected_behavior_clear": true/false, "issues": ["issue1", "issue2"], "pass": true/false}}

pass = true when: score >= 3 AND expected_behavior_clear == true
"""
    return header + "\n".join(sample_sections) + footer


# ---------------------------------------------------------------------------
# Gemini CLI caller
# ---------------------------------------------------------------------------

def call_gemini(prompt: str, expected_count: int) -> tuple[list[dict], str]:
    """Call Gemini CLI to score a batch.

    Returns:
        (list of score dicts, raw output string)
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                ["gemini", "-m", GEMINI_MODEL, "-s", "false", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=GEMINI_TIMEOUT,
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
                    time.sleep(5)
                    continue
                return [], ""

            raw_output = output

            # Strip markdown fences if present
            if output.startswith("```"):
                output = output.split("\n", 1)[1] if "\n" in output else output[3:]
            if output.endswith("```"):
                output = output[:-3]
            output = output.strip()
            if output.startswith("json"):
                output = output[4:].strip()

            # Parse each line as a JSON object
            results = []
            for line in output.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    start = line.find("{")
                    end = line.rfind("}") + 1
                    if start >= 0 and end > start:
                        obj = json.loads(line[start:end])
                        results.append(obj)
                except json.JSONDecodeError:
                    results.append({"_raw": line, "_parse_error": True})

            return results, raw_output

        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRIES - 1:
                print(f"    Timeout (attempt {attempt+1}), retrying...", flush=True)
                time.sleep(5)
                continue
            return [], ""
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(5)
                continue
            print(f"    Error: {e}", file=sys.stderr)
            return [], ""

    return [], ""


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def validate_training_score(score: dict) -> dict:
    """Validate and normalize a training score dict."""
    dims = ["score", "reasoning_quality", "tool_usage_correctness", "think_block_quality"]
    for d in dims:
        val = score.get(d)
        if not isinstance(val, (int, float)):
            score[d] = 0
        else:
            score[d] = max(0, min(5, int(val)))

    if "action_type_match" not in score or not isinstance(score.get("action_type_match"), bool):
        score["action_type_match"] = False

    if "issues" not in score or not isinstance(score.get("issues"), list):
        raw_issues = score.get("issues", "")
        if isinstance(raw_issues, str) and raw_issues and raw_issues.lower() != "none":
            score["issues"] = [raw_issues]
        else:
            score["issues"] = []

    # Compute pass based on rubric
    score["pass"] = (
        score["score"] >= 3
        and score["action_type_match"] is True
        and score["think_block_quality"] >= 2
    )

    return score


def validate_eval_score(score: dict) -> dict:
    """Validate and normalize an eval score dict."""
    dims = ["score", "rubric_clarity", "scenario_realism"]
    for d in dims:
        val = score.get(d)
        if not isinstance(val, (int, float)):
            score[d] = 0
        else:
            score[d] = max(0, min(5, int(val)))

    if "expected_behavior_clear" not in score or not isinstance(score.get("expected_behavior_clear"), bool):
        score["expected_behavior_clear"] = False

    if "issues" not in score or not isinstance(score.get("issues"), list):
        raw_issues = score.get("issues", "")
        if isinstance(raw_issues, str) and raw_issues and raw_issues.lower() != "none":
            score["issues"] = [raw_issues]
        else:
            score["issues"] = []

    # Compute pass based on rubric
    score["pass"] = (
        score["score"] >= 3
        and score["expected_behavior_clear"] is True
    )

    return score


def detect_type(input_path: str) -> str:
    """Auto-detect whether input is training data or eval cases based on path."""
    path_lower = input_path.lower()
    if "eval_case" in path_lower or "eval_cases" in path_lower or "/eval/" in path_lower:
        return "eval"
    return "training"


# ---------------------------------------------------------------------------
# File scoring
# ---------------------------------------------------------------------------

def score_file(
    input_path: str,
    output_path: str | None,
    score_type: str,
    batch_size: int,
    shard_spec: str | None,
    limit: int,
    min_score: int,
    show_summary: bool,
):
    """Score a single JSONL file."""

    input_p = Path(input_path)
    if not input_p.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        return

    # Determine output path
    if output_path is None:
        if shard_spec:
            shard_num = shard_spec.split("/")[0]
            output_path = str(input_p.parent / f"{input_p.stem}_scored_shard{shard_num}.jsonl")
        else:
            output_path = str(input_p.parent / f"{input_p.stem}_scored.jsonl")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Raw response log and malformed output paths
    raw_response_path = f"{output_path}_raw_responses.jsonl"
    malformed_path = str(Path(output_path).with_suffix(".malformed.jsonl"))

    # Auto-detect type if needed
    if score_type == "auto":
        score_type = detect_type(input_path)

    # Load input samples
    samples = []
    skipped = 0
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
    if skipped:
        print(f"Skipped {skipped} corrupt input lines")

    # Apply sharding
    if shard_spec:
        shard_num, shard_total = map(int, shard_spec.split("/"))
        all_samples = samples
        samples = [s for i, s in enumerate(all_samples) if i % shard_total == (shard_num - 1)]
        print(f"Shard {shard_num}/{shard_total}: {len(samples)} samples (of {len(all_samples)} total)")

    if limit > 0:
        samples = samples[:limit]

    # Auto-resume: track already-scored IDs
    scored_ids = set()
    if Path(output_path).exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        qa = r.get("_qa")
                        sid = r.get("id", r.get("case_id", ""))
                        if qa is not None and sid:
                            scored_ids.add(sid)
                    except json.JSONDecodeError:
                        pass
        if scored_ids:
            print(f"Auto-resuming: {len(scored_ids)} already scored")

    remaining = [s for s in samples if s.get("id", s.get("case_id", "")) not in scored_ids]

    print(f"\nInput:       {input_path}")
    print(f"Output:      {output_path}")
    print(f"Type:        {score_type}")
    print(f"Model:       {GEMINI_MODEL}")
    print(f"Batch size:  {batch_size}")
    print(f"Total:       {len(samples)}")
    print(f"Remaining:   {len(remaining)}")
    print(f"Malformed:   {malformed_path}")
    print()

    if not remaining:
        print("Nothing to score!")
        if show_summary:
            _print_summary_from_file(output_path, score_type, min_score)
        return

    # Tracking
    pass_count = 0
    fail_count = 0
    error_count = 0
    malformed_count = 0
    total_scored = len(scored_ids)
    total_target = len(samples)
    consecutive_empty = 0
    overall_start = time.time()

    # Accumulators for summary
    all_scores = []

    # Choose prompt builder and validator
    if score_type == "eval":
        build_prompt = build_eval_prompt
        validate_score = validate_eval_score
    else:
        build_prompt = build_training_prompt
        validate_score = validate_training_score

    out_f = open(output_path, "a")
    mal_f = open(malformed_path, "a")
    raw_f = open(raw_response_path, "a")

    try:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            batch_ids = [s.get("id", s.get("case_id", "?")) for s in batch]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(remaining) + batch_size - 1) // batch_size

            print(
                f"  Batch {batch_num}/{total_batches} "
                f"({len(batch)} samples: {batch_ids[0]}..{batch_ids[-1]})...",
                end=" ", flush=True,
            )
            batch_t = time.time()

            prompt = build_prompt(batch)
            results, raw_output = call_gemini(prompt, len(batch))

            batch_elapsed = time.time() - batch_t

            # Save raw output on any parse issues
            if raw_output and (
                len(results) != len(batch)
                or any(r.get("_parse_error") for r in results)
            ):
                raw_f.write(json.dumps({
                    "batch_num": batch_num,
                    "expected": len(batch),
                    "got": len(results),
                    "raw": raw_output,
                }) + "\n")
                raw_f.flush()
                os.fsync(raw_f.fileno())

            # Circuit breaker: too many consecutive empty results
            if not results:
                consecutive_empty += 1
                if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                    print(f"\n\nCIRCUIT BREAKER: {MAX_CONSECUTIVE_EMPTY} consecutive empty results. Stopping.")
                    break
            else:
                consecutive_empty = 0

            # Match results to samples by position
            matched = 0
            for j, sample in enumerate(batch):
                sid = sample.get("id", sample.get("case_id", f"unknown_{batch_start+j}"))

                if j < len(results):
                    score = results[j]

                    if score.get("_parse_error"):
                        # Save malformed output
                        mal_f.write(json.dumps({
                            "id": sid,
                            "raw": score.get("_raw", ""),
                            "batch": batch_num,
                        }) + "\n")
                        mal_f.flush()
                        os.fsync(mal_f.fileno())
                        malformed_count += 1
                        sample["_qa"] = None
                        error_count += 1
                    else:
                        # Validate and normalize
                        score["id"] = sid
                        score = validate_score(score)
                        sample["_qa"] = score
                        matched += 1
                        all_scores.append(score)

                        if score.get("pass"):
                            pass_count += 1
                        else:
                            fail_count += 1
                else:
                    # Fewer results than samples — truncation
                    mal_f.write(json.dumps({
                        "id": sid,
                        "raw": "TRUNCATED — no result returned",
                        "batch": batch_num,
                    }) + "\n")
                    mal_f.flush()
                    os.fsync(mal_f.fileno())
                    malformed_count += 1
                    sample["_qa"] = None
                    error_count += 1

                out_line = json.dumps(sample) + "\n"
                out_f.write(out_line)
                out_f.flush()
                os.fsync(out_f.fileno())

            total_scored += matched

            # Timing stats
            per_sample = batch_elapsed / len(batch)
            elapsed_total = time.time() - overall_start
            scored_so_far = batch_start + len(batch)
            remaining_count = len(remaining) - scored_so_far
            eta_seconds = remaining_count * per_sample if per_sample > 0 else 0
            eta_min = eta_seconds / 60

            print(
                f"{batch_elapsed:.1f}s ({per_sample:.1f}s/ea) "
                f"matched={matched}/{len(batch)} "
                f"| scored={total_scored}/{total_target} "
                f"| pass={pass_count} fail={fail_count} "
                f"| elapsed={elapsed_total/60:.1f}m ETA={eta_min:.0f}m",
                flush=True,
            )

            if matched < len(batch):
                print(f"    WARNING: {len(batch)-matched} samples unmatched (truncation or parse error)")

            time.sleep(1)  # Brief pause between batches

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved — safe to resume.")
    finally:
        out_f.close()
        mal_f.close()
        raw_f.close()

    # Final summary
    elapsed_total = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"QA SCORING COMPLETE")
    print(f"  Scored:    {total_scored}/{total_target}")
    print(f"  Passed:    {pass_count}")
    print(f"  Failed:    {fail_count}")
    print(f"  Errors:    {error_count}")
    print(f"  Malformed: {malformed_count}")
    effective = max(1, total_scored - len(scored_ids))
    print(f"  Time:      {elapsed_total/60:.1f} min ({elapsed_total/effective:.1f}s/sample)")
    print(f"  Output:    {output_path}")
    if malformed_count:
        print(f"  Malformed: {malformed_path}")
    print(f"{'='*60}")

    if show_summary:
        _print_detailed_summary(all_scores, score_type, min_score, output_path)


def _print_summary_from_file(output_path: str, score_type: str, min_score: int):
    """Read an already-scored file and print summary stats."""
    if not Path(output_path).exists():
        print("No scored file found for summary.")
        return

    all_scores = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                qa = r.get("_qa")
                if qa and isinstance(qa, dict) and "score" in qa:
                    all_scores.append(qa)
            except json.JSONDecodeError:
                pass

    if not all_scores:
        print("No scored samples found in output file.")
        return

    _print_detailed_summary(all_scores, score_type, min_score, output_path)


def _print_detailed_summary(
    all_scores: list[dict],
    score_type: str,
    min_score: int,
    output_path: str,
):
    """Print a detailed scoring summary."""
    if not all_scores:
        print("\nNo scores to summarize.")
        return

    total = len(all_scores)
    passed = sum(1 for s in all_scores if s.get("pass"))
    failed = total - passed

    avg_score = sum(s.get("score", 0) for s in all_scores) / total

    print(f"\nQA Summary:")
    print(f"  Total scored: {total}")
    print(f"  Passed: {passed} ({100*passed/total:.1f}%)")
    print(f"  Failed: {failed} ({100*failed/total:.1f}%)")
    print(f"  Avg score: {avg_score:.1f}/5")

    if score_type == "training":
        dims = {
            "reasoning_quality": "Avg reasoning_quality",
            "tool_usage_correctness": "Avg tool_usage_correctness",
            "think_block_quality": "Avg think_block_quality",
        }
        for key, label in dims.items():
            vals = [s.get(key, 0) for s in all_scores if isinstance(s.get(key), (int, float)) and s.get(key) > 0]
            if vals:
                print(f"  {label}: {sum(vals)/len(vals):.1f}/5")

        # Action type match rate
        atm_count = sum(1 for s in all_scores if s.get("action_type_match") is True)
        print(f"  Action type match: {atm_count}/{total} ({100*atm_count/total:.1f}%)")

    elif score_type == "eval":
        dims = {
            "rubric_clarity": "Avg rubric_clarity",
            "scenario_realism": "Avg scenario_realism",
        }
        for key, label in dims.items():
            vals = [s.get(key, 0) for s in all_scores if isinstance(s.get(key), (int, float)) and s.get(key) > 0]
            if vals:
                print(f"  {label}: {sum(vals)/len(vals):.1f}/5")

        ebc_count = sum(1 for s in all_scores if s.get("expected_behavior_clear") is True)
        print(f"  Expected behavior clear: {ebc_count}/{total} ({100*ebc_count/total:.1f}%)")

    # Score distribution
    print(f"\n  Score distribution:")
    for s_val in range(1, 6):
        count = sum(1 for s in all_scores if s.get("score") == s_val)
        bar = "#" * count
        print(f"    {s_val}/5: {count:4d} {bar}")

    # Most common issues
    all_issues = []
    for s in all_scores:
        issues = s.get("issues", [])
        if isinstance(issues, list):
            all_issues.extend(issues)
    if all_issues:
        from collections import Counter
        issue_counts = Counter(all_issues).most_common(5)
        print(f"\n  Top issues:")
        for issue, count in issue_counts:
            print(f"    [{count}x] {issue}")

    print(f"\n  Output: {output_path}")


# ---------------------------------------------------------------------------
# Score-all-v3 convenience
# ---------------------------------------------------------------------------

def find_all_v3_files(base_dir: str = "data/v3") -> list[str]:
    """Find all V3 generated JSONL files that need scoring."""
    patterns = [
        f"{base_dir}/foundation/shard_*.jsonl",
        f"{base_dir}/behavioral/*.jsonl",
        f"{base_dir}/categories/*.jsonl",
        f"{base_dir}/eval_cases/*.jsonl",
    ]

    files = []
    for pattern in patterns:
        files.extend(glob(pattern))

    # Exclude already-scored, malformed, log, and raw response files
    exclude_suffixes = (
        "_scored.jsonl",
        ".malformed.jsonl",
        ".log",
        "_raw_responses.jsonl",
    )
    files = [
        f for f in files
        if not any(f.endswith(suffix) for suffix in exclude_suffixes)
        and "_scored_shard" not in f
    ]

    files.sort()
    return files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V3 QA scoring — batch Gemini Flash scorer with sharding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/qa_score_v3.py --input data/v3/foundation/shard_1.jsonl
  python3 scripts/qa_score_v3.py --input data/v3/foundation/shard_1.jsonl --shard 1/3
  python3 scripts/qa_score_v3.py --input data/v3/eval_cases/eval.jsonl --type eval
  python3 scripts/qa_score_v3.py --score-all-v3 --summary
        """,
    )
    parser.add_argument("--input", help="Input JSONL file")
    parser.add_argument("--output", default=None, help="Output JSONL file (default: {input_stem}_scored.jsonl)")
    parser.add_argument("--shard", default=None, help="Shard spec: N/M (e.g., 1/3)")
    parser.add_argument(
        "--type", default="auto", choices=["training", "eval", "auto"],
        help="Sample type (default: auto-detect from path)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Samples per Gemini call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--limit", type=int, default=0, help="Only score first N samples (for testing)")
    parser.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE, help=f"Minimum passing score (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--summary", action="store_true", help="Print detailed pass/fail/score summary at end")
    parser.add_argument("--score-all-v3", action="store_true", help="Find and score ALL V3 generated data")

    args = parser.parse_args()

    if args.score_all_v3:
        files = find_all_v3_files()
        if not files:
            print("No V3 files found to score. Expected files in data/v3/{foundation,behavioral,categories,eval_cases}/")
            sys.exit(1)

        print(f"Found {len(files)} V3 files to score:")
        for f in files:
            print(f"  {f}")
        print()

        for filepath in files:
            print(f"\n{'='*60}")
            print(f"Scoring: {filepath}")
            print(f"{'='*60}")
            score_file(
                input_path=filepath,
                output_path=None,
                score_type=args.type,
                batch_size=args.batch_size,
                shard_spec=args.shard,
                limit=args.limit,
                min_score=args.min_score,
                show_summary=args.summary,
            )

        print(f"\n\nAll V3 files scored.")
        return

    if not args.input:
        parser.error("--input is required (or use --score-all-v3)")

    score_file(
        input_path=args.input,
        output_path=args.output,
        score_type=args.type,
        batch_size=args.batch_size,
        shard_spec=args.shard,
        limit=args.limit,
        min_score=args.min_score,
        show_summary=args.summary,
    )


if __name__ == "__main__":
    main()
