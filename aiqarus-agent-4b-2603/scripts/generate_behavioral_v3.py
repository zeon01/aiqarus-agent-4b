#!/usr/bin/env python3
"""
V3 Behavioral Failure Training Data Generator
==============================================
Generates ~2,800 training samples (buffer for rejects → 2,500 usable)
across 5 behavioral failure categories identified from V2 evaluation.

Uses randomized tool schemas from the 799 training schema library and
Gemini 3.1 Pro Preview via CLI for generation.

Features:
- Batched generation (5 samples per CLI call)
- Shard support for parallel execution
- Resume-safe (tracks generated IDs)
- Per-category validation (think blocks, message count, tool name checks)
- 42+ enterprise contexts, rotated per batch

Usage:
  # Full run (all 5 categories)
  python scripts/generate_behavioral_v3.py

  # Single category
  python scripts/generate_behavioral_v3.py --category over_execution

  # Parallel shards
  python scripts/generate_behavioral_v3.py --shard 1/3
  python scripts/generate_behavioral_v3.py --shard 2/3
  python scripts/generate_behavioral_v3.py --shard 3/3

  # Smoke test (5 per category)
  python scripts/generate_behavioral_v3.py --limit 5
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # aiqarus-agent-4b-2603/
SCHEMA_FILE = PROJECT_ROOT / "data" / "v3" / "tool_schemas" / "schemas.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "v3" / "behavioral"

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL = "gemini-3.1-pro-preview"
BATCH_SIZE = 5
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Categories with target counts
# ---------------------------------------------------------------------------
CATEGORIES = {
    "over_execution": 800,
    "tool_loop_prevention": 500,
    "clarification_follow_through": 500,
    "handoff_routing": 500,
    "tool_chain_trajectories": 500,
}

# ---------------------------------------------------------------------------
# Enterprise contexts (42 — shared with generate_eval_v3.py)
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
def load_schemas() -> list[dict]:
    """Load training schemas from the 799-schema library."""
    with open(SCHEMA_FILE) as f:
        return json.load(f)


def sample_tools(schemas: list, n: int = 3, rng: random.Random = None) -> list:
    """Sample n tools from schema list, ensuring domain diversity."""
    if rng is None:
        rng = random.Random()

    if len(schemas) <= n:
        return schemas[:n]

    # Pick from different domains first
    by_domain: dict[str, list] = {}
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
    """Format tools as clean JSON (no domain/source metadata)."""
    clean = []
    for t in tools:
        clean.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        })
    return json.dumps(clean, indent=2)


# ---------------------------------------------------------------------------
# Prompt templates per category
# ---------------------------------------------------------------------------

def _build_over_execution_prompt(batch_size: int, start_id: int,
                                  tools_json: str, context: str) -> str:
    return f"""You are generating training data for an enterprise AI agent model. Each sample teaches the model to complete ONE task and STOP — resisting the urge to call additional tools.

## Output Format

Output EXACTLY {batch_size} JSON objects, one per line. No markdown fences, no explanations, no commentary. Each line must be a complete, valid JSON object.

## JSON Schema Per Sample

{{
  "id": "over_execution_XXXX",
  "category": "over_execution",
  "messages": [
    {{"role": "system", "content": "Enterprise system prompt (1-8 sentences). Mention the context: {context}. Include available tools and agent role."}},
    {{"role": "user", "content": "User asks for ONE specific task. Be specific and realistic."}},
    {{"role": "assistant", "content": "<think>\\nReasoning about what the user needs. Identify the specific task. Note that ONLY this one task was requested — I should not do anything beyond this.\\n</think>\\n\\n<tool_call>\\n{{\\"name\\": \\"tool_name\\", \\"arguments\\": {{...}}}}\\n</tool_call>"}},
    {{"role": "tool", "content": "Realistic tool response with useful data", "name": "tool_name"}},
    {{"role": "assistant", "content": "<think>\\nThe tool returned the result. I should present this to the user. I notice I COULD also [tempting follow-up action], but the user only asked for [original task]. I should not perform additional actions unprompted.\\n</think>\\n\\nHere is the result of your request: [natural language summary of the tool result]. Let me know if you need anything else."}}
  ],
  "tools": {tools_json},
  "action_type": "call_tool",
  "source": "v3_behavioral"
}}

## Critical Requirements

1. The user's request must be completable with 1 tool call (occasionally 2 for lookup-then-action).
2. The assistant's FINAL message must be a TEXT response presenting results — NOT another tool call.
3. The final <think> block MUST explicitly mention a tempting follow-up action the model is choosing NOT to take. Examples:
   - "I could also update the contact record, but the user only asked me to look it up."
   - "It might be helpful to also check the related invoices, but that wasn't requested."
   - "I notice the subscription is about to expire — I could flag that, but the user only asked for the current status."
4. The <think> blocks must be substantive (400+ characters each). Show genuine reasoning about scope.
5. Tool names in <tool_call> MUST match names in the tools array.
6. Vary the enterprise scenarios — different departments, roles, situations.
7. The system prompt should set up a context where follow-up actions would be natural but are NOT requested.
8. messages array must have exactly 5 entries (system, user, assistant+tool_call, tool, assistant+text).

## Available Tools

{tools_json}

## Enterprise Context

{context}

## TASK

Generate {batch_size} over_execution training samples. Start IDs at {start_id:04d} (over_execution_{start_id:04d}, over_execution_{start_id+1:04d}, ...).

Each sample should have a DIFFERENT user request and a DIFFERENT tempting follow-up the model resists."""


def _build_tool_loop_prevention_prompt(batch_size: int, start_id: int,
                                        tools_json: str, context: str) -> str:
    return f"""You are generating training data for an enterprise AI agent model. Each sample teaches the model to handle tool errors gracefully — retry once with adjusted params, then STOP and explain the failure if the retry also fails. Never retry more than 2 times total.

## Output Format

Output EXACTLY {batch_size} JSON objects, one per line. No markdown fences, no explanations, no commentary. Each line must be a complete, valid JSON object.

## JSON Schema Per Sample

{{
  "id": "tool_loop_prevention_XXXX",
  "category": "tool_loop_prevention",
  "messages": [
    {{"role": "system", "content": "Enterprise system prompt (1-8 sentences). Mention the context: {context}. Include available tools and agent role."}},
    {{"role": "user", "content": "User makes a reasonable request that requires a tool call."}},
    {{"role": "assistant", "content": "<think>\\nReasoning about the request and which tool to use. Planning the approach.\\n</think>\\n\\n<tool_call>\\n{{\\"name\\": \\"tool_name\\", \\"arguments\\": {{...}}}}\\n</tool_call>"}},
    {{"role": "tool", "content": "{{\\"error\\": \\"timeout\\", \\"message\\": \\"Service unavailable\\"}}", "name": "tool_name"}},
    {{"role": "assistant", "content": "<think>\\nThe tool returned an error: [error type]. I should try once more with adjusted parameters or a different approach. This is my first retry.\\n</think>\\n\\n<tool_call>\\n{{\\"name\\": \\"tool_name_or_alt\\", \\"arguments\\": {{adjusted params}}}}\\n</tool_call>"}},
    {{"role": "tool", "content": "{{\\"error\\": \\"...\\", \\"message\\": \\"...\\"}}", "name": "tool_name_or_alt"}},
    {{"role": "assistant", "content": "<think>\\nThe retry also failed. I have now tried 2 times and both attempts returned errors. Retrying again is unlikely to help — [specific reasoning about why]. I should explain the situation to the user and suggest alternatives they can try.\\n</think>\\n\\nI was unable to complete your request. [Explain what was tried and what failed]. Here are some alternatives:\\n- [Alternative 1]\\n- [Alternative 2]\\nWould you like me to try a different approach?"}}
  ],
  "tools": {tools_json},
  "action_type": "call_tool",
  "source": "v3_behavioral"
}}

## Critical Requirements

1. The tool must fail on BOTH the first call AND the retry (2 errors total).
2. The assistant's FINAL message must be a TEXT explanation of the failure — NOT another tool call.
3. The final <think> block MUST explicitly reason: "I've already tried X times, retrying won't help because [reason]."
4. The retry (2nd attempt) should use adjusted parameters, different arguments, or a different tool — not an identical repeat.
5. Error types should vary: timeout, 403 forbidden, 404 not found, 500 internal error, 429 rate limit, connection refused.
6. The <think> blocks must be substantive (400+ characters each). Show reasoning about error diagnosis.
7. Tool names in <tool_call> MUST match names in the tools array.
8. The final response must suggest concrete alternatives (not just "try again later").
9. messages array must have exactly 7 entries (system, user, assistant+call1, tool_error1, assistant+call2, tool_error2, assistant+explanation).

## Available Tools

{tools_json}

## Enterprise Context

{context}

## TASK

Generate {batch_size} tool_loop_prevention training samples. Start IDs at {start_id:04d}.

Vary the error types and recovery strategies across samples."""


def _build_clarification_follow_through_prompt(batch_size: int, start_id: int,
                                                tools_json: str, context: str) -> str:
    return f"""You are generating training data for an enterprise AI agent model. Each sample teaches the model to ask for clarification on ambiguous requests, then execute ONLY what was clarified — nothing more.

## Output Format

Output EXACTLY {batch_size} JSON objects, one per line. No markdown fences, no explanations, no commentary. Each line must be a complete, valid JSON object.

## JSON Schema Per Sample

{{
  "id": "clarification_follow_through_XXXX",
  "category": "clarification_follow_through",
  "messages": [
    {{"role": "system", "content": "Enterprise system prompt (1-8 sentences). Mention the context: {context}. Include available tools and agent role."}},
    {{"role": "user", "content": "User makes an ambiguous request with multiple valid interpretations. The ambiguity should be semantic (what they mean), not just a missing parameter."}},
    {{"role": "assistant", "content": "<think>\\nThe user's request is ambiguous — it could mean [interpretation A] or [interpretation B]. I should not guess; I need to ask for clarification to avoid performing the wrong action.\\n</think>\\n\\nI want to make sure I get this right. When you say [quote ambiguous part], do you mean:\\n1. [Interpretation A]\\n2. [Interpretation B]\\nPlease let me know which one, and I'll take care of it."}},
    {{"role": "user", "content": "User answers the clarification, narrowing to ONE specific interpretation."}},
    {{"role": "assistant", "content": "<think>\\nThe user clarified they want [specific interpretation]. I should now execute exactly that — [specific action]. I should NOT also do [related but unrequested action] since the user specifically chose the narrow scope. The clarification answer limits my scope to only [what was clarified].\\n</think>\\n\\n<tool_call>\\n{{\\"name\\": \\"tool_name\\", \\"arguments\\": {{...}}}}\\n</tool_call>"}},
    {{"role": "tool", "content": "Realistic tool response", "name": "tool_name"}},
    {{"role": "assistant", "content": "<think>\\nGot the result. I'll present exactly what was asked for based on the clarification. I should not expand scope beyond what the user specified.\\n</think>\\n\\nHere's what I found based on your clarification: [result summary]. Let me know if you need anything else."}}
  ],
  "tools": {tools_json},
  "action_type": "clarify",
  "source": "v3_behavioral"
}}

## Critical Requirements

1. The initial user request must be genuinely ambiguous (not just missing a parameter value).
2. The assistant's first response asks a clarification question with 2-3 specific options — NO tool call.
3. The user's follow-up NARROWS the scope to one specific interpretation.
4. After clarification, the assistant executes ONLY the clarified action — one tool call, nothing more.
5. The <think> block after clarification MUST explicitly reference the user's answer and note what it is NOT doing.
6. All <think> blocks must be substantive (400+ characters each).
7. Tool names in <tool_call> MUST match names in the tools array.
8. messages array must have exactly 7 entries (system, user, assistant+clarify, user+answer, assistant+tool_call, tool, assistant+result).
9. The ambiguity must be realistic for the enterprise context — something a real employee might ask unclearly.

## Available Tools

{tools_json}

## Enterprise Context

{context}

## TASK

Generate {batch_size} clarification_follow_through training samples. Start IDs at {start_id:04d}.

Each sample should feature a different type of ambiguity and a different narrowing clarification."""


def _build_handoff_routing_prompt(batch_size: int, start_id: int,
                                   tools_json: str, context: str) -> str:
    return f"""You are generating training data for an enterprise AI agent model. Each sample teaches the model to recognize when a request is outside its domain/authorization and route it to the right team or person via an appropriate channel — NOT just refuse.

## Output Format

Output EXACTLY {batch_size} JSON objects, one per line. No markdown fences, no explanations, no commentary. Each line must be a complete, valid JSON object.

## JSON Schema Per Sample

{{
  "id": "handoff_routing_XXXX",
  "category": "handoff_routing",
  "messages": [
    {{"role": "system", "content": "Enterprise system prompt (1-8 sentences). Mention the context: {context}. Define the agent's specific domain/scope and mention teams or departments that handle other areas."}},
    {{"role": "user", "content": "User requests an action that is outside the agent's defined scope/authorization. The request is legitimate but belongs to a different team."}},
    {{"role": "assistant", "content": "<think>\\nThe user is asking me to [action]. Let me check if this falls within my authorized scope. My system prompt defines my role as [role] with access to [tools]. This request requires [capability] which falls under [other team/department]'s responsibility. I should not attempt this myself — I'm not authorized and could cause issues. Instead, I should route this to the appropriate team via [channel: email/slack/ticket/team mention].\\n</think>\\n\\nThis request falls outside my current authorization. The [team/department name] team handles [type of request]. Let me route this to them for you.\\n\\n<tool_call>\\n{{\\"name\\": \\"communication_tool\\", \\"arguments\\": {{\\"to\\": \\"team/person\\", \\"message\\": \\"routing message\\"}}}}\\n</tool_call>"}},
    {{"role": "tool", "content": "Confirmation that the message/ticket/email was sent", "name": "communication_tool"}},
    {{"role": "assistant", "content": "<think>\\nThe handoff has been routed successfully. I should confirm to the user what happened and set expectations for next steps.\\n</think>\\n\\nI've routed your request to [team/person] via [channel]. They typically respond within [timeframe]. Is there anything else within my scope that I can help with?"}}
  ],
  "tools": {tools_json},
  "action_type": "handoff",
  "source": "v3_behavioral"
}}

## Critical Requirements

1. The user's request must be legitimate but outside the agent's defined scope.
2. The system prompt MUST define the agent's scope clearly so the boundary is obvious.
3. The assistant must NOT attempt the action itself — it routes via a communication tool.
4. The <think> block must explicitly reason: "I'm not authorized for X, but [team] handles this via [channel]."
5. The routing action should use one of the available tools (email, slack, ticket, etc.). If no communication tool is available in the tools list, the assistant should include a hypothetical one AND explain the routing in text.
6. All <think> blocks must be substantive (400+ characters each).
7. Tool names in <tool_call> MUST match names in the tools array.
8. messages array must have exactly 5 entries (system, user, assistant+handoff_call, tool, assistant+confirmation).
9. Include at least one communication-capable tool in the tools array. You may add a generic one like {{"name": "send_internal_message", "description": "Send a message to an internal team or individual via the company messaging system", "parameters": {{"type": "object", "properties": {{"to": {{"type": "string", "description": "Recipient team or individual"}}, "subject": {{"type": "string", "description": "Message subject"}}, "message": {{"type": "string", "description": "Message body"}}, "priority": {{"type": "string", "enum": ["low", "normal", "high", "urgent"], "description": "Message priority"}}}}, "required": ["to", "message"]}}}} alongside the sampled tools.
10. Different handoff scenarios: wrong department, needs manager approval, needs legal review, needs compliance check, needs finance authorization, needs IT admin access.

## Available Tools (use these PLUS add a communication tool if none is communication-capable)

{tools_json}

## Enterprise Context

{context}

## TASK

Generate {batch_size} handoff_routing training samples. Start IDs at {start_id:04d}.

Vary the handoff reasons and routing channels across samples."""


def _build_tool_chain_trajectories_prompt(batch_size: int, start_id: int,
                                           tools_json: str, context: str) -> str:
    return f"""You are generating training data for an enterprise AI agent model. Each sample teaches the model to execute 3-4 sequential dependent tool calls, where each call uses data from the previous result. At least 20% of samples should include a mid-chain failure where the model must adapt.

## Output Format

Output EXACTLY {batch_size} JSON objects, one per line. No markdown fences, no explanations, no commentary. Each line must be a complete, valid JSON object.

## JSON Schema Per Sample — SUCCESS PATH (80%)

{{
  "id": "tool_chain_trajectories_XXXX",
  "category": "tool_chain_trajectories",
  "messages": [
    {{"role": "system", "content": "Enterprise system prompt (1-8 sentences). Mention the context: {context}. Include available tools and agent role."}},
    {{"role": "user", "content": "User requests something that requires 3-4 sequential tool calls with data dependencies."}},
    {{"role": "assistant", "content": "<think>\\nLet me plan the full chain before starting. To fulfill this request, I need to:\\n1. First, call [tool A] to get [data X]\\n2. Use [data X] to call [tool B] for [data Y]\\n3. Use [data Y] to call [tool C] for [final result]\\nLet me start with step 1.\\n</think>\\n\\n<tool_call>\\n{{\\"name\\": \\"tool_a\\", \\"arguments\\": {{...}}}}\\n</tool_call>"}},
    {{"role": "tool", "content": "Result with data X", "name": "tool_a"}},
    {{"role": "assistant", "content": "<think>\\nGot [data X] from step 1. Now I can proceed to step 2 using this data. Specifically, I need [field] from the result to pass to [tool B].\\n</think>\\n\\n<tool_call>\\n{{\\"name\\": \\"tool_b\\", \\"arguments\\": {{\\"param_from_step1\\": \\"value from data X\\"}}}}\\n</tool_call>"}},
    {{"role": "tool", "content": "Result with data Y", "name": "tool_b"}},
    {{"role": "assistant", "content": "<think>\\nStep 2 complete. Got [data Y]. Now for the final step, I'll use [specific field from data Y] to call [tool C].\\n</think>\\n\\n<tool_call>\\n{{\\"name\\": \\"tool_c\\", \\"arguments\\": {{\\"param_from_step2\\": \\"value from data Y\\"}}}}\\n</tool_call>"}},
    {{"role": "tool", "content": "Final result", "name": "tool_c"}},
    {{"role": "assistant", "content": "<think>\\nAll 3 steps of the chain are complete. Let me synthesize the results into a clear response for the user.\\n</think>\\n\\nHere is the complete result: [synthesis of all chain results]. The process involved [brief summary of the chain]."}}
  ],
  "tools": {tools_json},
  "action_type": "call_tool",
  "source": "v3_behavioral"
}}

## JSON Schema Per Sample — FAILURE PATH (20%)

For failure-path samples, the 2nd or 3rd tool call returns an error. The model must adapt: explain partial results, try alternative, or report what succeeded and what failed.

The messages array will include the error and the assistant's adaptation response. The final assistant message must present whatever partial results were obtained plus explain the failure.

The <think> block after the error must reason: "Step N failed with [error]. I have partial results from steps 1 through N-1. I should [report partial results / try alternative / explain to user]."

## Critical Requirements

1. Each sample must have 3-4 sequential tool calls with genuine data dependencies (step N uses output from step N-1).
2. The FIRST <think> block must plan the full chain before any tool call.
3. Subsequent <think> blocks must reference specific data from previous results.
4. For failure-path samples (at least 1 in every batch of 5): the error tool response must be realistic, and the model must adapt gracefully.
5. All <think> blocks must be substantive (400+ characters each).
6. Tool names in <tool_call> MUST match names in the tools array.
7. Success-path messages: 9-11 entries. Failure-path messages: 7-11 entries.
8. The chain must be logically coherent — each step should genuinely need the previous step's output.
9. Do NOT create artificial chains where steps are independent; each step must truly depend on prior data.

## Available Tools

{tools_json}

## Enterprise Context

{context}

## TASK

Generate {batch_size} tool_chain_trajectories training samples. Start IDs at {start_id:04d}.

Make at least 1 of the {batch_size} samples a FAILURE PATH (mid-chain error with adaptation).
Vary the chain patterns and tool combinations across samples."""


PROMPT_BUILDERS = {
    "over_execution": _build_over_execution_prompt,
    "tool_loop_prevention": _build_tool_loop_prevention_prompt,
    "clarification_follow_through": _build_clarification_follow_through_prompt,
    "handoff_routing": _build_handoff_routing_prompt,
    "tool_chain_trajectories": _build_tool_chain_trajectories_prompt,
}


# ---------------------------------------------------------------------------
# Gemini CLI invocation (same pattern as generate_eval_v3.py)
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
            buffer = ""
            brace_depth = 0
            for line in output.split("\n"):
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

            # Save raw output for debugging
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
    """Load IDs of already-generated samples for resume."""
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
        return list(CATEGORIES.keys())

    parts = shard_str.split("/")
    shard_num = int(parts[0])
    total_shards = int(parts[1])

    all_cats = list(CATEGORIES.keys())
    return [cat for i, cat in enumerate(all_cats) if (i % total_shards) + 1 == shard_num]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _extract_tool_call_names(text: str) -> list[str]:
    """Extract tool names from <tool_call> blocks in assistant messages."""
    names = []
    # Match <tool_call> blocks and parse the JSON inside
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            call = json.loads(match.group(1))
            if "name" in call:
                names.append(call["name"])
        except json.JSONDecodeError:
            continue
    return names


def validate_sample(sample: dict, category: str) -> tuple[bool, str]:
    """Validate a generated training sample."""
    if "_parse_error" in sample:
        return False, "parse_error"

    # Required fields
    for field in ["id", "category", "messages", "tools", "action_type", "source"]:
        if field not in sample:
            return False, f"missing_{field}"

    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return False, "messages_not_list"

    # Minimum message count (varies by category)
    min_messages = {
        "over_execution": 5,
        "tool_loop_prevention": 7,
        "clarification_follow_through": 7,
        "handoff_routing": 5,
        "tool_chain_trajectories": 7,
    }
    if len(messages) < min_messages.get(category, 5):
        return False, f"too_few_messages_{len(messages)}"

    # Check for <think> blocks in assistant messages (at least one must be 400+ chars)
    has_substantive_think = False
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if think_match:
                think_text = think_match.group(1).strip()
                if len(think_text) >= 400:
                    has_substantive_think = True
                    break

    if not has_substantive_think:
        return False, "no_substantive_think_block"

    # Check tool names match tools array
    tool_names_in_schema = set()
    tools = sample.get("tools", [])
    if isinstance(tools, list):
        for t in tools:
            if isinstance(t, dict) and "name" in t:
                tool_names_in_schema.add(t["name"])

    # For handoff_routing, allow extra communication tools added by the model
    if category == "handoff_routing":
        tool_names_in_schema.add("send_internal_message")
        tool_names_in_schema.add("send_email")
        tool_names_in_schema.add("create_ticket")
        tool_names_in_schema.add("send_slack_message")
        tool_names_in_schema.add("route_to_team")

    # Extract tool call names from assistant messages
    for msg in messages:
        if msg.get("role") == "assistant":
            called_names = _extract_tool_call_names(msg.get("content", ""))
            for name in called_names:
                if name not in tool_names_in_schema:
                    return False, f"tool_name_mismatch_{name}"

    # Check that the last message is from assistant (not tool)
    if messages and messages[-1].get("role") != "assistant":
        return False, "last_message_not_assistant"

    # Category-specific checks
    if category == "over_execution":
        # Last assistant message should NOT contain <tool_call>
        last_assistant = messages[-1].get("content", "")
        if "<tool_call>" in last_assistant:
            return False, "over_execution_last_msg_has_tool_call"

    if category == "tool_loop_prevention":
        # Last assistant message should NOT contain <tool_call>
        last_assistant = messages[-1].get("content", "")
        if "<tool_call>" in last_assistant:
            return False, "loop_prevention_last_msg_has_tool_call"

    if category == "clarification_follow_through":
        # First assistant message should NOT contain <tool_call> (it's a clarification)
        for msg in messages:
            if msg.get("role") == "assistant":
                first_assistant = msg.get("content", "")
                if "<tool_call>" in first_assistant:
                    return False, "clarification_first_response_has_tool_call"
                break

    return True, "ok"


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate_category(category: str, schemas: list, limit: int = None):
    """Generate all training samples for one category."""
    target = CATEGORIES[category]
    if limit is not None:
        target = min(target, limit)

    output_file = OUTPUT_DIR / f"{category}.jsonl"
    malformed_file = OUTPUT_DIR / f"{category}.malformed.jsonl"

    existing_ids = load_existing_ids(output_file)
    existing_count = len(existing_ids)

    if existing_count >= target:
        print(f"  [{category}] Already have {existing_count}/{target}. Skipping.")
        return existing_count

    remaining = target - existing_count
    print(f"  [{category}] Have {existing_count}/{target}. Generating {remaining} more...")

    # Seeded RNG for reproducibility per category
    rng = random.Random(hash(category) + 2026)
    contexts = list(ENTERPRISE_CONTEXTS)
    rng.shuffle(contexts)
    context_idx = 0

    generated = existing_count
    batch_num = 0
    prompt_builder = PROMPT_BUILDERS[category]

    while generated < target:
        batch_size = min(BATCH_SIZE, target - generated)
        start_id = generated + 1

        # Sample tools for this batch (2-4 per batch)
        n_tools = rng.randint(2, 4)
        tools = sample_tools(schemas, n=n_tools, rng=rng)
        tools_json = format_tools_for_prompt(tools)

        # Rotate enterprise context
        context = contexts[context_idx % len(contexts)]
        context_idx += 1

        # Build prompt
        prompt = prompt_builder(batch_size, start_id, tools_json, context)

        print(f"    Batch {batch_num+1}: {category} #{start_id:04d}-{start_id+batch_size-1:04d} "
              f"({n_tools} tools, {context[:40]}...)", flush=True)

        results = call_gemini(prompt, batch_size)

        if not results:
            print(f"    WARNING: Empty results for batch {batch_num+1}", flush=True)
            batch_num += 1
            time.sleep(2)
            continue

        valid_count = 0
        for case in results:
            is_valid, reason = validate_sample(case, category)

            if is_valid:
                case_id = case.get("id", f"{category}_{start_id + valid_count:04d}")
                if case_id in existing_ids:
                    case_id = f"{category}_{generated + valid_count + 1:04d}"
                    case["id"] = case_id

                # Ensure metadata
                case["category"] = category
                case["source"] = "v3_behavioral"
                if "tools" not in case or not case["tools"]:
                    case["tools"] = [{"name": t["name"], "description": t["description"],
                                      "parameters": t["parameters"]} for t in tools]

                with open(output_file, "a") as f:
                    f.write(json.dumps(case, ensure_ascii=False) + "\n")

                existing_ids.add(case_id)
                valid_count += 1
            else:
                case["_rejection_reason"] = reason
                case["_category"] = category
                with open(malformed_file, "a") as f:
                    f.write(json.dumps(case, ensure_ascii=False) + "\n")

        generated += valid_count
        batch_num += 1

        if valid_count < batch_size:
            print(f"    Got {valid_count}/{batch_size} valid cases", flush=True)

        # Brief pause between batches
        time.sleep(1)

    print(f"  [{category}] Done: {generated} samples generated.")
    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Generate V3 behavioral failure training data (~2,800 samples)"
    )
    parser.add_argument("--shard", type=str, default=None,
                        help="Shard spec: '1/3', '2/3', '3/3'")
    parser.add_argument("--category", type=str, default=None,
                        help="Generate for single category")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples per category (for testing)")
    args = parser.parse_args()

    # Load schemas
    print("Loading schema library...", flush=True)
    schemas = load_schemas()
    print(f"  {len(schemas)} training schemas loaded")

    # Determine categories to generate
    if args.category:
        if args.category not in CATEGORIES:
            print(f"ERROR: Unknown category '{args.category}'. "
                  f"Valid: {list(CATEGORIES.keys())}")
            sys.exit(1)
        categories = [args.category]
    else:
        categories = get_shard_categories(args.shard)

    print(f"\nCategories to generate: {categories}")
    if args.shard:
        print(f"Shard: {args.shard}")
    if args.limit:
        print(f"Limit: {args.limit} per category")

    # Print targets
    total_target = 0
    for cat in categories:
        t = min(CATEGORIES[cat], args.limit) if args.limit else CATEGORIES[cat]
        total_target += t
        print(f"  {cat}: {t}")
    print(f"  TOTAL TARGET: {total_target}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate
    total = 0
    for category in categories:
        print(f"\n{'='*60}")
        print(f"Category: {category} (target: {CATEGORIES[category]})")
        print(f"{'='*60}")
        count = generate_category(category, schemas, args.limit)
        total += count

    # Summary
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Categories: {len(categories)}")
    print(f"Output: {OUTPUT_DIR}/")

    # Per-category breakdown
    print(f"\nPer-category counts:")
    for cat in categories:
        cat_file = OUTPUT_DIR / f"{cat}.jsonl"
        if cat_file.exists():
            count = sum(1 for _ in open(cat_file))
            malformed = OUTPUT_DIR / f"{cat}.malformed.jsonl"
            mal_count = sum(1 for _ in open(malformed)) if malformed.exists() else 0
            print(f"  {cat}: {count} valid, {mal_count} malformed")
        else:
            print(f"  {cat}: 0 valid (file not created)")


if __name__ == "__main__":
    main()
