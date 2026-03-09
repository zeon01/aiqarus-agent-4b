"""
llm_judge_multiturn.py — LLM-based judge for multi-turn eval transcripts
=========================================================================
Scores full multi-turn conversation transcripts using an LLM judge.

For each transcript, the judge scores:
  - decision_quality: 1-5 (did the model make correct decisions at each turn?)
  - reasoning_depth: 1-5 (quality of <think> blocks across all turns)
  - response_quality: 1-5 (quality of final response and intermediate responses)
  - composite: average of the three

Usage:
  python3 training/llm_judge_multiturn.py
  python3 training/llm_judge_multiturn.py --input data/multiturn_results_base.jsonl --output data/multiturn_scored_base.jsonl
"""

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def build_judge_prompt(tc: dict) -> str:
    """Build the prompt sent to the LLM judge for a multi-turn transcript."""
    category = tc.get("category", "unknown")
    difficulty = tc.get("difficulty", "unknown")
    tools_called = tc.get("tools_called", [])
    heuristic_scores = tc.get("scores", {})

    # Format transcript compactly
    transcript_lines = []
    for t in tc.get("transcript", []):
        role = t["role"]
        content = t["content"]
        if role == "system":
            transcript_lines.append(f"[SYSTEM] {content[:500]}{'...' if len(content) > 500 else ''}")
        elif role == "user":
            transcript_lines.append(f"[USER] {content}")
        elif role == "assistant":
            transcript_lines.append(f"[ASSISTANT] {content}")
        elif role == "tool_response":
            tool_name = t.get("tool_name", "unknown")
            transcript_lines.append(f"[TOOL_RESPONSE ({tool_name})] {content}")

    transcript_text = "\n\n".join(transcript_lines)

    return f"""You are an expert evaluator scoring an AI agent's multi-turn conversation. Be strict but fair.

## Conversation Metadata
- **Category:** {category}
- **Difficulty:** {difficulty}
- **Tools called:** {json.dumps(tools_called)}
- **Number of turns:** {heuristic_scores.get('num_turns', '?')}
- **Heuristic pass:** {heuristic_scores.get('overall_pass', '?')}

## Full Conversation Transcript
{transcript_text}

## Scoring Instructions

Evaluate the ENTIRE conversation trajectory and return ONLY a JSON object (no markdown, no explanation outside JSON):

{{"decision_quality": 1-5, "reasoning_depth": 1-5, "response_quality": 1-5, "explanation": "Brief explanation"}}

Scoring rubric:
- **decision_quality** (1-5): Did the model make the right decision at EACH turn? Correct tool selection, correct argument values, appropriate escalation/refusal when needed, correct final action. 1=wrong decisions, 3=mostly correct, 5=perfect decision chain.
- **reasoning_depth** (1-5): Quality of <think> blocks across ALL turns. 1=no reasoning or restates prompt, 2=surface level, 3=considers multiple factors, 4=strategic with edge case awareness, 5=defensive reasoning explaining what NOT to do.
- **response_quality** (1-5): Quality of responses to the user. 1=wrong/harmful, 2=partially correct, 3=correct but generic, 4=correct and helpful with context, 5=professional, synthesizes tool results clearly.

Return ONLY the JSON object."""


def extract_json_from_output(output: str) -> dict | None:
    """Extract a JSON object from LLM output."""
    # Strip markdown fences
    if "```json" in output:
        output = output.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in output:
        output = output.split("```", 1)[1].split("```", 1)[0].strip()

    lines = output.split('\n')
    json_lines = []
    in_json = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('{'):
            in_json = True
            json_lines = [stripped]
        elif in_json:
            json_lines.append(stripped)
            if stripped.endswith('}'):
                break

    if json_lines:
        json_str = ' '.join(json_lines)
        return json.loads(json_str)

    return json.loads(output)


def call_openrouter_judge(prompt: str, max_retries: int = 3) -> dict | None:
    """Call OpenRouter API with gpt-5.1-codex-mini."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="openai/gpt-5.1-codex-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            output = resp.choices[0].message.content.strip()
            if not output:
                continue
            return extract_json_from_output(output)

        except (json.JSONDecodeError,) as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"    Failed after {max_retries} attempts: {e}", file=sys.stderr)
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"    Error: {e}", file=sys.stderr)
            return None

    return None


def main():
    parser = argparse.ArgumentParser(description="LLM judge for multi-turn eval")
    parser.add_argument("--input", default="data/multiturn_results.jsonl",
                        help="Path to multi-turn results JSONL")
    parser.add_argument("--output", default="data/multiturn_scored_codex.jsonl",
                        help="Path to write scored results")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only judge first N results (0=all)")
    args = parser.parse_args()

    # Load results
    results = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    if args.limit > 0:
        results = results[:args.limit]

    # Resume support
    already_scored = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("decision_quality") is not None:
                        already_scored[entry.get("id", "")] = entry

    skipped = len(already_scored)
    print(f"Loaded {len(results)} multi-turn results to judge")
    if skipped:
        print(f"Resuming: {skipped} already scored, {len(results) - skipped} remaining")
    print(f"Using: gpt-5.1-codex-mini via OpenRouter\n")

    out_f = open(args.output, "a" if skipped else "w")

    scored_list = list(already_scored.values())
    for i, tc in enumerate(results):
        tc_id = tc.get("id", f"mt_case_{i}")

        if tc_id in already_scored:
            continue

        prompt = build_judge_prompt(tc)

        print(f"  [{i+1}/{len(results)}] {tc_id}...", end=" ", flush=True)
        scores = call_openrouter_judge(prompt)

        entry = {"id": tc_id, "category": tc.get("category", "unknown")}

        if scores:
            dq = scores.get("decision_quality", 0)
            rd = scores.get("reasoning_depth", 0)
            rq = scores.get("response_quality", 0)
            composite = round((dq + rd + rq) / 3, 2)

            entry.update({
                "heuristic_pass": tc.get("scores", {}).get("overall_pass", False),
                "decision_quality": dq,
                "reasoning_depth": rd,
                "response_quality": rq,
                "composite": composite,
                "explanation": scores.get("explanation", ""),
            })
            print(f"composite={composite:.1f}/5 (decision={dq}, reasoning={rd}, response={rq})")
        else:
            entry.update({
                "heuristic_pass": tc.get("scores", {}).get("overall_pass", False),
                "decision_quality": None,
                "reasoning_depth": None,
                "response_quality": None,
                "composite": None,
            })
            print("ERROR (no score)")

        scored_list.append(entry)
        out_f.write(json.dumps(entry) + "\n")
        out_f.flush()

        time.sleep(0.3)

    out_f.close()

    # Summary
    valid = [s for s in scored_list if s.get("composite") is not None]
    if valid:
        avg_composite = sum(s["composite"] for s in valid) / len(valid)
        avg_decision = sum(s["decision_quality"] for s in valid) / len(valid)
        avg_reasoning = sum(s["reasoning_depth"] for s in valid) / len(valid)
        avg_response = sum(s["response_quality"] for s in valid) / len(valid)

        print(f"\n{'='*60}")
        print(f"MULTI-TURN JUDGE SUMMARY ({len(valid)} scored)")
        print(f"  Avg composite:        {avg_composite:.2f}/5")
        print(f"  Avg decision quality: {avg_decision:.2f}/5")
        print(f"  Avg reasoning depth:  {avg_reasoning:.2f}/5")
        print(f"  Avg response quality: {avg_response:.2f}/5")

        # Per category
        cats = {}
        for s in valid:
            cat = s.get("category", "unknown")
            if cat not in cats:
                cats[cat] = []
            cats[cat].append(s["composite"])

        print(f"\n  By category:")
        for cat in sorted(cats):
            avg = sum(cats[cat]) / len(cats[cat])
            print(f"    {cat:30s} {avg:.2f}/5  (n={len(cats[cat])})")
        print(f"{'='*60}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
