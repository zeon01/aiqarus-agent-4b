"""
build_restraint_pairs_v3.py — Task 15: Tool Restraint Alignment
================================================================
Extracts false-positive tool calls from V3 eval results and builds
restraint-specific SimPO preference pairs.

False positive = model generated <tool_call> but expected action was
clarify / escalate / refuse / answer_directly.

For each false positive:
  1. Check on-policy completions for a correct non-tool response.
  2. If found, pair it (on-policy chosen vs. false-positive rejected).
  3. If not found, generate a correct response via Gemini 3.1 Pro CLI
     (unless --on-policy-only is set).

Output format (matches simpo_v3.py expectations):
  {
    "prompt": [{"role": "system", ...}, {"role": "user", ...}],
    "chosen": [{"role": "assistant", "content": "correct non-tool response"}],
    "rejected": [{"role": "assistant", "content": "incorrect <tool_call> response"}],
    "source": "on_policy|frontier_generated",
    "case_id": "original eval case ID",
    "expected_action": "clarify|escalate|refuse|answer_directly"
  }

Usage:
  # Build pairs from eval results + on-policy completions
  python3 scripts/build_restraint_pairs_v3.py

  # Skip frontier generation (only use on-policy pairs)
  python3 scripts/build_restraint_pairs_v3.py --on-policy-only

  # Custom paths
  python3 scripts/build_restraint_pairs_v3.py --eval-results PATH --onpolicy PATH --output PATH

  # Dry run (report stats only, no output written)
  python3 scripts/build_restraint_pairs_v3.py --dry-run
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EVAL_RESULTS = "data/v3/v3_eval.jsonl"
DEFAULT_ONPOLICY     = "data/v3/onpolicy_completions.jsonl"
DEFAULT_OUTPUT       = "data/v3/restraint_pairs.jsonl"
DEFAULT_MIN_SCORE    = 3.5

# Non-tool action types — these are the targets for restraint training.
# A false positive is a case where the model called a tool but the
# expected_action_type is one of these.
NON_TOOL_ACTIONS = {"clarify", "escalate", "refuse", "answer_directly"}

# Frontier generation model
FRONTIER_MODEL = "gemini-3.1-pro-preview"

# Rate limiting for frontier generation
FRONTIER_BASE_DELAY  = 2.0      # seconds between calls
FRONTIER_MAX_RETRIES = 3
FRONTIER_TIMEOUT     = 120      # seconds per call

# Action-type instructions used in frontier generation prompts
ACTION_INSTRUCTIONS = {
    "clarify": (
        "ask the user a clarifying question. Identify what information is "
        "missing or ambiguous and politely request it."
    ),
    "escalate": (
        "escalate to a human reviewer or supervisor. Explain why this "
        "situation requires human judgment and what specifically needs review."
    ),
    "refuse": (
        "politely refuse the request. Explain why you cannot or should not "
        "perform this action (e.g., unauthorized, out of scope, dangerous)."
    ),
    "answer_directly": (
        "answer the user's question directly using your own knowledge. "
        "Do NOT call any tool — the answer does not require tool access."
    ),
}


# ---------------------------------------------------------------------------
# Safe file I/O helpers
# ---------------------------------------------------------------------------

def safe_append_jsonl(filepath: str, obj: dict):
    """Append a single JSON object to a JSONL file with fsync."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(filepath: str) -> list[dict]:
    """Load JSONL file, skipping malformed lines."""
    items = []
    if not os.path.exists(filepath):
        return items
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def load_completed_case_ids(filepath: str) -> set:
    """Load case_ids already written to the output file (for resume)."""
    ids = set()
    if not os.path.exists(filepath):
        return ids
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "case_id" in obj:
                    ids.add(obj["case_id"])
            except json.JSONDecodeError:
                continue
    return ids


# ---------------------------------------------------------------------------
# Parsing helpers (same as eval harness / on-policy generator)
# ---------------------------------------------------------------------------

def extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output. Returns list of parsed dicts."""
    calls = []

    # Format 1: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            calls.append(json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    if calls:
        return calls

    # Format 2: ```json blocks
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            if "name" in obj or "function" in obj:
                calls.append(obj)
        except json.JSONDecodeError:
            pass
    if calls:
        return calls

    # Format 3: bare JSON with "name" and "arguments" keys
    for m in re.finditer(
        r'\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}', text
    ):
        try:
            calls.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            pass

    return calls


def has_tool_call(text: str) -> bool:
    """Check whether text contains any <tool_call> tag."""
    return bool(re.search(r"<tool_call>", text))


def classify_action(text: str, tool_calls: list[dict]) -> str:
    """Classify model output into action type."""
    if tool_calls:
        return "call_tool"

    lower = text.lower()

    escalation_kw = [
        "escalat", "human review", "manager", "supervisor",
        "manual review", "higher authority", "approval needed",
        "hand off", "handoff", "hand this off", "route to",
    ]
    if any(kw in lower for kw in escalation_kw):
        return "escalate"

    refusal_kw = [
        "cannot", "can't", "unable to", "not authorized",
        "not permitted", "refuse", "decline", "inappropriate",
        "outside my scope", "beyond my capabilities",
        "i'm not able", "i am not able",
    ]
    if any(kw in lower for kw in refusal_kw):
        return "refuse"

    clarify_kw = [
        "could you clarify", "can you specify", "which one",
        "what do you mean", "more information", "please provide",
        "could you provide", "i need to know", "can you confirm",
        "unclear", "ambiguous", "not sure which",
        "could you tell me", "do you mean",
    ]
    if any(kw in lower for kw in clarify_kw):
        return "clarify"

    return "answer_directly"


# ---------------------------------------------------------------------------
# Prompt / transcript extraction
# ---------------------------------------------------------------------------

def extract_prompt_from_transcript(transcript: list[dict]) -> list[dict]:
    """Extract the prompt portion (system + all messages up to and NOT
    including the last assistant turn that was the false positive) from
    a multi-turn transcript.

    This gives the full context leading up to the bad response.
    """
    prompt_msgs = []

    # Walk the transcript and collect everything up to the last assistant turn
    # that contains <tool_call>.  We want the full multi-turn context.
    last_fp_idx = None
    for i, turn in enumerate(transcript):
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "assistant" and has_tool_call(content):
            last_fp_idx = i

    if last_fp_idx is None:
        # No tool call found; shouldn't happen for a false positive,
        # but fall back to everything except last assistant turn
        last_fp_idx = len(transcript)
        for i in range(len(transcript) - 1, -1, -1):
            if transcript[i].get("role") == "assistant":
                last_fp_idx = i
                break

    # Collect all turns before the false-positive assistant turn
    for i in range(last_fp_idx):
        turn = transcript[i]
        role = turn.get("role", "")
        content = turn.get("content", "")

        # Skip meta turns and tool_response turns (re-role them as user
        # for the pair format, since that's how they were fed to the model)
        if role in ("system", "user", "assistant"):
            prompt_msgs.append({"role": role, "content": content})
        elif role == "tool_response":
            # Tool responses were injected as user messages during inference
            tool_content = content
            tool_name = turn.get("tool_name", "unknown")
            formatted = (
                f"<tool_response>\n{tool_content}\n</tool_response>"
            )
            prompt_msgs.append({"role": "user", "content": formatted})
        # Skip _meta turns

    return prompt_msgs


def extract_rejected_messages(transcript: list[dict]) -> list[dict]:
    """Extract the false-positive assistant response(s) from a transcript.

    Returns the assistant messages starting from the last one that contains
    a <tool_call>. For multi-turn false positives this captures the full
    bad generation.
    """
    # Find the last assistant turn with a tool call
    last_fp_idx = None
    for i, turn in enumerate(transcript):
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "assistant" and has_tool_call(content):
            last_fp_idx = i

    if last_fp_idx is None:
        # Fall back to last assistant turn
        for i in range(len(transcript) - 1, -1, -1):
            if transcript[i].get("role") == "assistant":
                last_fp_idx = i
                break

    if last_fp_idx is None:
        return []

    # Return all assistant messages from last_fp_idx onward
    rejected = []
    for i in range(last_fp_idx, len(transcript)):
        turn = transcript[i]
        if turn.get("role") == "assistant":
            rejected.append({
                "role": "assistant",
                "content": turn["content"],
            })

    return rejected


def extract_chosen_from_onpolicy(completion: dict) -> list[dict]:
    """Extract assistant messages from an on-policy completion transcript."""
    msgs = []
    for turn in completion.get("transcript", []):
        if turn.get("role") == "assistant":
            msgs.append({
                "role": "assistant",
                "content": turn["content"],
            })
    return msgs


# ---------------------------------------------------------------------------
# Step 1: Identify false positives from eval results
# ---------------------------------------------------------------------------

def find_false_positives(eval_results: list[dict]) -> list[dict]:
    """Find cases where the model called a tool but should not have.

    A false positive satisfies ALL of:
      1. expected_action_type is in NON_TOOL_ACTIONS
      2. final_action == "call_tool" (model called a tool)
      3. transcript contains at least one <tool_call> tag

    Returns the filtered eval result dicts.
    """
    false_positives = []

    for result in eval_results:
        expected = result.get("expected_action_type", "")
        final_action = result.get("final_action", "")

        if expected not in NON_TOOL_ACTIONS:
            continue

        if final_action != "call_tool":
            continue

        # Verify there's actually a <tool_call> in the transcript
        transcript = result.get("transcript", [])
        found_tool_call = False
        for turn in transcript:
            if turn.get("role") == "assistant" and has_tool_call(turn.get("content", "")):
                found_tool_call = True
                break

        if not found_tool_call:
            continue

        false_positives.append(result)

    return false_positives


# ---------------------------------------------------------------------------
# Step 2: Match on-policy completions to false positives
# ---------------------------------------------------------------------------

def index_onpolicy_completions(
    completions: list[dict],
    min_score: float,
) -> dict[str, list[dict]]:
    """Index on-policy completions by case_id.

    Only keeps completions that:
      1. Have a composite score >= min_score
      2. Do NOT contain any <tool_call> in assistant turns
      3. Have the correct action type matching expected_action_type

    Returns {case_id: [list of valid completions sorted by score desc]}.
    """
    by_case: dict[str, list[dict]] = defaultdict(list)

    for comp in completions:
        case_id = comp.get("case_id", "")
        if not case_id:
            continue

        # Check composite score
        composite = comp.get("composite", -1)
        if composite < min_score:
            continue

        # Check that no assistant turn contains <tool_call>
        transcript = comp.get("transcript", [])
        has_tc = False
        for turn in transcript:
            if turn.get("role") == "assistant" and has_tool_call(turn.get("content", "")):
                has_tc = True
                break
        if has_tc:
            continue

        # Check that the final action matches expected
        final_action = comp.get("final_action", "")
        expected = comp.get("expected_action_type", "")
        if expected in NON_TOOL_ACTIONS and final_action != expected:
            # Action type mismatch — skip
            continue

        by_case[case_id].append(comp)

    # Sort each case's completions by composite score descending
    for case_id in by_case:
        by_case[case_id].sort(
            key=lambda c: c.get("composite", 0), reverse=True
        )

    return dict(by_case)


# ---------------------------------------------------------------------------
# Step 3: Frontier generation for cases with no correct on-policy completion
# ---------------------------------------------------------------------------

def build_frontier_prompt(
    prompt_msgs: list[dict],
    expected_action: str,
    tools: list[dict],
) -> str:
    """Build a prompt for Gemini 3.1 Pro to generate a correct non-tool
    response for a false-positive case."""

    action_instruction = ACTION_INSTRUCTIONS.get(
        expected_action,
        "respond appropriately without calling any tool.",
    )

    # Extract system prompt and user messages from prompt_msgs
    system_prompt = ""
    user_messages = []
    assistant_messages = []
    for msg in prompt_msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_messages.append(content)
        elif role == "assistant":
            assistant_messages.append(content)

    # Build a readable conversation for the frontier model
    conversation_parts = []
    for msg in prompt_msgs:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            conversation_parts.append(f"[System prompt]: {content}")
        elif role == "user":
            conversation_parts.append(f"[User]: {content}")
        elif role == "assistant":
            conversation_parts.append(f"[Assistant (previous turn)]: {content}")

    conversation_text = "\n\n".join(conversation_parts)

    # Format tools for display
    clean_tools = []
    for t in tools:
        clean_tools.append({
            "name": t.get("name", "unknown"),
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {}),
        })
    tools_json = json.dumps(clean_tools, indent=2) if clean_tools else "[]"

    prompt = f"""You are an enterprise AI agent. Given this conversation, generate the correct assistant response.

The correct action type is: {expected_action}
Do NOT call any tool. Instead, {action_instruction}

## Conversation so far

{conversation_text}

## Available tools (for context only — do NOT call any of them)

{tools_json}

## Instructions

Generate ONLY the assistant response. It MUST include a <think>...</think> block with substantive reasoning (at least 100 characters) explaining WHY you are choosing to {expected_action} instead of calling a tool. After the </think> block, write the actual response to the user.

Do NOT generate <tool_call> tags under any circumstances.
Do NOT include any prefix like "Assistant:" — just output the raw response content.
Do NOT wrap output in markdown fences."""

    return prompt


def generate_frontier_response(
    prompt_msgs: list[dict],
    expected_action: str,
    tools: list[dict],
    raw_save_dir: str,
    case_id: str,
) -> str | None:
    """Generate a correct non-tool response using Gemini 3.1 Pro CLI.

    Returns the generated response text, or None on failure.
    Raw output is saved on parse failure for debugging.
    """
    prompt = build_frontier_prompt(prompt_msgs, expected_action, tools)

    for attempt in range(FRONTIER_MAX_RETRIES):
        try:
            result = subprocess.run(
                ["gemini", "-m", FRONTIER_MODEL, "-s", "false", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=FRONTIER_TIMEOUT,
            )

            output = result.stdout.strip()
            stderr = result.stderr.strip()

            # Check for rate limiting
            if stderr:
                stderr_lower = stderr.lower()
                if any(phrase in stderr_lower for phrase in [
                    "rate limit", "quota exceeded", "resource exhausted",
                    "too many requests", "429",
                ]):
                    wait = 30 * (attempt + 1)
                    print(f"      Rate limited -- waiting {wait}s...")
                    time.sleep(wait)
                    continue

            if not output:
                if attempt < FRONTIER_MAX_RETRIES - 1:
                    time.sleep(5)
                    continue
                return None

            # Strip markdown fences if present
            if output.startswith("```"):
                output = output.split("\n", 1)[1] if "\n" in output else output[3:]
            if output.endswith("```"):
                output = output[:-3]
            output = output.strip()

            # Validate: must NOT contain <tool_call>
            if has_tool_call(output):
                print(f"      WARNING: Frontier response contains <tool_call>, discarding.")
                # Save raw for debugging
                _save_raw_frontier(raw_save_dir, case_id, output, "contained_tool_call")
                if attempt < FRONTIER_MAX_RETRIES - 1:
                    time.sleep(FRONTIER_BASE_DELAY)
                    continue
                return None

            # Validate: should contain <think> block
            if "<think>" not in output:
                # Wrap with a minimal think block
                output = f"<think>I should {expected_action} here rather than call a tool.</think>\n{output}"

            return output

        except subprocess.TimeoutExpired:
            print(f"      Frontier generation timed out (attempt {attempt+1}).")
            if attempt < FRONTIER_MAX_RETRIES - 1:
                time.sleep(5)
                continue
            return None

        except FileNotFoundError:
            print("      ERROR: 'gemini' CLI not found. Install with: pip install google-genai")
            return None

        except Exception as e:
            print(f"      Frontier generation error: {e}")
            if attempt < FRONTIER_MAX_RETRIES - 1:
                time.sleep(5)
                continue
            return None

    return None


def _save_raw_frontier(raw_save_dir: str, case_id: str, output: str, reason: str):
    """Save raw frontier output on failure for debugging."""
    os.makedirs(raw_save_dir, exist_ok=True)
    raw_path = os.path.join(raw_save_dir, "frontier_raw_failures.jsonl")
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_id": case_id,
        "reason": reason,
        "raw_output": output,
    }
    safe_append_jsonl(raw_path, entry)


# ---------------------------------------------------------------------------
# Step 4: Extract tool schemas from eval case transcript
# ---------------------------------------------------------------------------

def extract_tools_from_transcript(transcript: list[dict]) -> list[dict]:
    """Try to extract tool schemas from the system prompt in a transcript.

    The system prompt typically contains a JSON array of tool definitions
    after the text "You have access to the following tools:".
    """
    for turn in transcript:
        if turn.get("role") != "system":
            continue
        content = turn.get("content", "")
        # Find the tool definitions block
        marker = "You have access to the following tools:"
        idx = content.find(marker)
        if idx < 0:
            continue
        after_marker = content[idx + len(marker):]
        # Find the JSON array
        try:
            # Try to find the start of the JSON array
            arr_start = after_marker.find("[")
            if arr_start < 0:
                continue
            # Find matching end
            brace_depth = 0
            arr_end = -1
            for i in range(arr_start, len(after_marker)):
                ch = after_marker[i]
                if ch == "[":
                    brace_depth += 1
                elif ch == "]":
                    brace_depth -= 1
                    if brace_depth == 0:
                        arr_end = i + 1
                        break
            if arr_end > arr_start:
                tools = json.loads(after_marker[arr_start:arr_end])
                if isinstance(tools, list):
                    return tools
        except (json.JSONDecodeError, ValueError):
            pass

    return []


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_restraint_pairs(
    eval_results_path: str,
    onpolicy_path: str,
    output_path: str,
    on_policy_only: bool = False,
    dry_run: bool = False,
    min_score: float = DEFAULT_MIN_SCORE,
):
    """Main pipeline: find false positives, match on-policy completions,
    optionally generate frontier responses, and output restraint pairs."""

    print("=" * 70)
    print("TOOL RESTRAINT ALIGNMENT — Build SimPO Pairs (Task 15)")
    print("=" * 70)

    # ── Load eval results ────────────────────────────────────────────
    print(f"\n  Loading eval results: {eval_results_path}")
    eval_results = load_jsonl(eval_results_path)
    if not eval_results:
        print(f"  ERROR: No eval results found at {eval_results_path}")
        sys.exit(1)
    print(f"  Loaded {len(eval_results)} eval results.")

    # ── Find false positives ─────────────────────────────────────────
    false_positives = find_false_positives(eval_results)
    print(f"\n  False positives found: {len(false_positives)}")

    if not false_positives:
        print("  No false positives to process. Exiting.")
        return

    # Report distribution by expected action
    fp_by_action: dict[str, int] = defaultdict(int)
    fp_by_category: dict[str, int] = defaultdict(int)
    for fp in false_positives:
        fp_by_action[fp.get("expected_action_type", "unknown")] += 1
        fp_by_category[fp.get("category", "unknown")] += 1

    print(f"\n  By expected action:")
    for action, count in sorted(fp_by_action.items(), key=lambda x: -x[1]):
        print(f"    {action:25s} {count}")

    print(f"\n  By category:")
    for cat, count in sorted(fp_by_category.items(), key=lambda x: -x[1]):
        print(f"    {cat:35s} {count}")

    if dry_run:
        print(f"\n  [DRY RUN] Would process {len(false_positives)} false positives.")
        print(f"  Exiting without writing output.")
        return

    # ── Load on-policy completions ───────────────────────────────────
    print(f"\n  Loading on-policy completions: {onpolicy_path}")
    onpolicy_raw = load_jsonl(onpolicy_path)
    print(f"  Loaded {len(onpolicy_raw)} on-policy completions.")

    onpolicy_index = index_onpolicy_completions(onpolicy_raw, min_score)
    n_cases_with_valid = len(onpolicy_index)
    n_total_valid = sum(len(v) for v in onpolicy_index.values())
    print(f"  Valid non-tool completions (score >= {min_score}): {n_total_valid} across {n_cases_with_valid} cases")

    # ── Resume support ───────────────────────────────────────────────
    completed_ids = load_completed_case_ids(output_path)
    if completed_ids:
        print(f"\n  Resuming: {len(completed_ids)} pairs already written, skipping those case_ids.")

    # ── Build pairs ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Building restraint pairs...")
    print(f"{'='*70}")

    stats = {
        "on_policy": 0,
        "frontier_generated": 0,
        "no_valid_chosen": 0,
        "skipped_resume": 0,
        "by_action": defaultdict(int),
    }

    raw_save_dir = os.path.join(os.path.dirname(output_path) or ".", "frontier_debug")
    total_fp = len(false_positives)

    for i, fp in enumerate(false_positives):
        case_id = fp.get("id", fp.get("case_id", f"unknown_{i}"))
        expected_action = fp.get("expected_action_type", "unknown")
        category = fp.get("category", "unknown")
        transcript = fp.get("transcript", [])

        # Skip if already completed (resume)
        if case_id in completed_ids:
            stats["skipped_resume"] += 1
            continue

        print(
            f"  [{i+1}/{total_fp}] {case_id} ({category}) "
            f"expected={expected_action}",
            end="",
            flush=True,
        )

        # Extract the prompt (everything before the false-positive response)
        prompt_msgs = extract_prompt_from_transcript(transcript)
        if not prompt_msgs:
            print("  SKIP (empty prompt)")
            stats["no_valid_chosen"] += 1
            continue

        # Extract the rejected response (the false-positive tool call)
        rejected_msgs = extract_rejected_messages(transcript)
        if not rejected_msgs:
            print("  SKIP (no rejected msgs)")
            stats["no_valid_chosen"] += 1
            continue

        # Verify rejected actually contains <tool_call>
        rejected_has_tc = False
        for msg in rejected_msgs:
            if has_tool_call(msg.get("content", "")):
                rejected_has_tc = True
                break
        if not rejected_has_tc:
            print("  SKIP (rejected has no <tool_call>)")
            stats["no_valid_chosen"] += 1
            continue

        # Try on-policy first
        valid_completions = onpolicy_index.get(case_id, [])
        chosen_msgs = None
        source = None

        if valid_completions:
            # Use the best-scoring on-policy completion
            best = valid_completions[0]
            chosen_msgs = extract_chosen_from_onpolicy(best)
            source = "on_policy"

            # Validate chosen does NOT contain <tool_call>
            chosen_has_tc = False
            for msg in chosen_msgs:
                if has_tool_call(msg.get("content", "")):
                    chosen_has_tc = True
                    break
            if chosen_has_tc:
                # This shouldn't happen due to indexing, but be safe
                chosen_msgs = None
                source = None

        if chosen_msgs and source == "on_policy":
            print(f"  on-policy (score={valid_completions[0].get('composite', '?')})")
            stats["on_policy"] += 1
        elif on_policy_only:
            print("  SKIP (no on-policy, --on-policy-only)")
            stats["no_valid_chosen"] += 1
            continue
        else:
            # Frontier generation
            print("  generating...", end="", flush=True)

            # Extract tools from transcript for context
            tools = extract_tools_from_transcript(transcript)

            generated_text = generate_frontier_response(
                prompt_msgs=prompt_msgs,
                expected_action=expected_action,
                tools=tools,
                raw_save_dir=raw_save_dir,
                case_id=case_id,
            )

            if generated_text:
                # Validate the generated response
                if has_tool_call(generated_text):
                    print(" FAIL (generated contains <tool_call>)")
                    stats["no_valid_chosen"] += 1
                    continue

                chosen_msgs = [{"role": "assistant", "content": generated_text}]
                source = "frontier_generated"
                print(f" OK")
                stats["frontier_generated"] += 1

                # Rate limit delay
                time.sleep(FRONTIER_BASE_DELAY)
            else:
                print(" FAIL (generation failed)")
                stats["no_valid_chosen"] += 1
                continue

        # Verify chosen does not contain <tool_call>
        if chosen_msgs is None:
            stats["no_valid_chosen"] += 1
            continue

        # Build the pair
        pair = {
            "prompt": prompt_msgs,
            "chosen": chosen_msgs,
            "rejected": rejected_msgs,
            "source": source,
            "case_id": case_id,
            "expected_action": expected_action,
        }

        # Write immediately (resume-safe)
        safe_append_jsonl(output_path, pair)
        stats["by_action"][expected_action] += 1

    # ── Summary ──────────────────────────────────────────────────────
    total_pairs = stats["on_policy"] + stats["frontier_generated"]

    print(f"\n{'='*70}")
    print("Restraint Pair Summary:")
    print(f"  False positives found: {total_fp}")
    if stats["skipped_resume"] > 0:
        print(f"  Skipped (already done): {stats['skipped_resume']}")
    if total_fp > 0:
        on_policy_pct = stats["on_policy"] / total_fp * 100 if total_fp else 0
        frontier_pct = stats["frontier_generated"] / total_fp * 100 if total_fp else 0
        no_valid_pct = stats["no_valid_chosen"] / total_fp * 100 if total_fp else 0
        print(f"  On-policy pairs built: {stats['on_policy']} ({on_policy_pct:.1f}%)")
        print(f"  Frontier-generated pairs: {stats['frontier_generated']} ({frontier_pct:.1f}%)")
        print(f"  No valid chosen found: {stats['no_valid_chosen']} ({no_valid_pct:.1f}%)")
    print(f"  Total pairs: {total_pairs}")

    print(f"\n  By expected action:")
    for action, count in sorted(stats["by_action"].items(), key=lambda x: -x[1]):
        print(f"    {action:25s} {count}")

    # Include resumed pairs in the count
    if completed_ids:
        total_on_disk = total_pairs + len(completed_ids)
        print(f"\n  Total pairs on disk (including resumed): {total_on_disk}")

    print(f"\n  Output: {output_path}")

    print(f"""
Next steps:
  # Run restraint-specific SimPO pass (higher beta for stronger signal)
  modal run --detach training/simpo_v3.py --pairs-path /data/dataset_v3/restraint_pairs.jsonl --beta 1.5""")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build restraint SimPO pairs from V3 eval false positives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Build pairs from eval results + on-policy completions
  python3 scripts/build_restraint_pairs_v3.py

  # Skip frontier generation (only use on-policy pairs)
  python3 scripts/build_restraint_pairs_v3.py --on-policy-only

  # Custom paths
  python3 scripts/build_restraint_pairs_v3.py --eval-results data/v3/v3_eval.jsonl --output data/v3/my_pairs.jsonl

  # Dry run (just report how many false positives found)
  python3 scripts/build_restraint_pairs_v3.py --dry-run
""",
    )

    parser.add_argument(
        "--eval-results",
        default=DEFAULT_EVAL_RESULTS,
        help=f"Path to eval results JSONL (default: {DEFAULT_EVAL_RESULTS})",
    )
    parser.add_argument(
        "--onpolicy",
        default=DEFAULT_ONPOLICY,
        help=f"Path to on-policy completions JSONL (default: {DEFAULT_ONPOLICY})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Path to output restraint pairs JSONL (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--on-policy-only",
        action="store_true",
        help="Skip frontier generation — only build pairs from on-policy completions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report false-positive stats only, do not write output.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Minimum composite score for chosen completions (default: {DEFAULT_MIN_SCORE})",
    )

    args = parser.parse_args()

    build_restraint_pairs(
        eval_results_path=args.eval_results,
        onpolicy_path=args.onpolicy,
        output_path=args.output,
        on_policy_only=args.on_policy_only,
        dry_run=args.dry_run,
        min_score=args.min_score,
    )


if __name__ == "__main__":
    main()
