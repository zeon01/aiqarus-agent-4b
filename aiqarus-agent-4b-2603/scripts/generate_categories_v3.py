#!/usr/bin/env python3
"""
V3 New Category Training Data Generator
========================================
Generates ~1,200 training samples (1,400 with reject buffer) across 5 NEW
categories being added in V3. Uses randomized tool schemas from the 799
training schema library.

Categories (1,400 total including reject buffer):
  - clarification_seeking:   400 samples
  - pii_data_sensitivity:    200 samples (+50 buffer = 250)
  - permission_verification: 200 samples (+50 buffer = 250)
  - correction_handling:     200 samples (+50 buffer = 250)
  - multi_turn_context:      200 samples (+50 buffer = 250)

Features:
  - Randomized tool schemas from 799 training library (2-4 per batch)
  - 42 enterprise contexts, rotated per batch
  - Batched generation (5 samples per Gemini CLI call)
  - Shard support: --shard 1/2, --shard 2/2
  - Resume-safe (tracks generated IDs)
  - Validation: <think> block (400+ chars), 5+ messages (10+ for multi_turn)
  - Malformed output saved separately

Usage:
  # Full run
  python scripts/generate_categories_v3.py

  # Single category
  python scripts/generate_categories_v3.py --category clarification_seeking

  # Parallel shards
  python scripts/generate_categories_v3.py --shard 1/2
  python scripts/generate_categories_v3.py --shard 2/2

  # Smoke test
  python scripts/generate_categories_v3.py --limit 5
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
SCHEMA_FILE = PROJECT_ROOT / "data" / "v3" / "tool_schemas" / "schemas.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "v3" / "categories"

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL = "gemini-3.1-pro-preview"
BATCH_SIZE = 5
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Categories: target counts (includes buffer for rejects)
# ---------------------------------------------------------------------------
CATEGORY_TARGETS = {
    "clarification_seeking": 400,
    "pii_data_sensitivity": 250,
    "permission_verification": 250,
    "correction_handling": 250,
    "multi_turn_context": 250,
}

CATEGORIES = list(CATEGORY_TARGETS.keys())

# ---------------------------------------------------------------------------
# Category definitions (used in prompt construction)
# ---------------------------------------------------------------------------
CATEGORY_DEFINITIONS = {
    "clarification_seeking": {
        "description": (
            "The user's request is genuinely ambiguous with multiple valid "
            "interpretations. The model MUST ask a clarifying question before "
            "taking any action — it must NOT guess and call a tool. The "
            "ambiguity should be semantic (not just a missing required "
            "parameter). After the user clarifies, the model executes the "
            "clarified intent."
        ),
        "action_type": "clarify",
        "behavioral_pattern": (
            "1. User sends an ambiguous request\n"
            "2. Assistant <think> block identifies the ambiguity, lists 2-3 "
            "possible interpretations, and explains why guessing would be risky\n"
            "3. Assistant asks a targeted clarifying question (NOT a tool call)\n"
            "4. User clarifies their intent\n"
            "5. Assistant <think> block confirms the interpretation, then "
            "executes the correct tool call"
        ),
        "constraints": (
            "- The ambiguity must be REAL — not just a missing parameter\n"
            "- The assistant's first response MUST be a clarifying question, "
            "NOT a tool call\n"
            "- <think> block must explicitly list the competing interpretations\n"
            "- After clarification, the assistant must execute ONLY what was "
            "clarified\n"
            "- Mix of ambiguity types: scope (40%), entity (30%), action (30%)"
        ),
        "examples": (
            '- "Update the account" — which account? Update what field?\n'
            '- "Send the report to the team" — which report? Which team?\n'
            '- "Check the status" — status of what? The order? The ticket? '
            "The deployment?\n"
            '- "Cancel it" — cancel the subscription? The meeting? The order?'
        ),
    },
    "pii_data_sensitivity": {
        "description": (
            "The scenario involves Personally Identifiable Information (PII) "
            "such as account numbers, SSNs, salary data, medical records, or "
            "credit card numbers. The model must handle PII responsibly — "
            "masking, redacting, refusing to share across unauthorized "
            "contexts, or escalating to compliance."
        ),
        "action_type": "varies (call_tool with masking, refuse, escalate)",
        "behavioral_pattern": (
            "Mix of three patterns:\n"
            "PATTERN A — Mask and proceed (50%):\n"
            "  1. User requests data containing PII\n"
            "  2. Assistant <think> identifies PII fields, decides to mask\n"
            "  3. Assistant calls the tool\n"
            "  4. Tool returns data with PII\n"
            "  5. Assistant presents results with PII masked (***-**-1234)\n\n"
            "PATTERN B — Refuse and explain (30%):\n"
            "  1. User requests PII across unauthorized boundary\n"
            "  2. Assistant <think> identifies the policy violation\n"
            "  3. Assistant refuses and explains WHY (cites policy/regulation)\n\n"
            "PATTERN C — Escalate to compliance (20%):\n"
            "  1. User requests bulk PII export or cross-department sharing\n"
            "  2. Assistant <think> identifies this needs compliance review\n"
            "  3. Assistant escalates to compliance/legal team"
        ),
        "constraints": (
            "- PII types: SSN, credit cards, medical records, salary data, "
            "bank accounts, dates of birth, home addresses\n"
            "- NEVER output raw PII in the assistant response\n"
            "- Masking format: last 4 digits visible (***-**-1234)\n"
            "- Cite specific regulation or policy when refusing (HIPAA, GDPR, "
            "SOC 2, company policy)\n"
            "- action_type: 'call_tool' for mask-and-proceed, 'refuse' for "
            "refuse-and-explain, 'escalate' for escalate-to-compliance"
        ),
        "examples": (
            '- "Pull the full employee records including SSNs for the '
            'marketing team" — refuse or escalate\n'
            '- "Look up customer John Smith\'s account details" — call tool, '
            "mask PII in response\n"
            '- "Export all patient records for the Q1 audit" — escalate to '
            "compliance\n"
            '- "What\'s Sarah\'s salary?" — refuse (unauthorized cross-'
            "department query)"
        ),
    },
    "permission_verification": {
        "description": (
            "The user requests an action that may require elevated permissions "
            "or authorization. The model must verify whether it (or the user) "
            "has the necessary access before proceeding. This is different "
            "from escalation (which is 'this is too risky') — this is 'do I "
            "have access to do this?'"
        ),
        "action_type": "clarify or call_tool (checking permission first)",
        "behavioral_pattern": (
            "Mix of two patterns:\n"
            "PATTERN A — Ask about permissions (50%):\n"
            "  1. User requests a privileged action\n"
            "  2. Assistant <think> identifies the permission concern\n"
            "  3. Assistant asks user to confirm authorization level\n"
            "  4. User confirms/denies\n"
            "  5. Assistant proceeds or explains why it cannot\n\n"
            "PATTERN B — Check permission tool first (50%):\n"
            "  1. User requests a privileged action\n"
            "  2. Assistant <think> identifies the permission concern\n"
            "  3. Assistant calls a permission/access-check tool first\n"
            "  4. Tool returns access level\n"
            "  5. Assistant proceeds if authorized, explains limitation if not"
        ),
        "constraints": (
            "- The permission concern must be REALISTIC — admin actions, "
            "write access, delete operations, cross-tenant queries\n"
            "- Assistant must NOT blindly execute the action first\n"
            "- <think> block must explicitly reason about WHY permissions are "
            "a concern\n"
            "- Include the available tools in the schema — one should be a "
            "permission-check or role-verification tool when using Pattern B\n"
            "- action_type: 'clarify' for Pattern A, 'call_tool' for Pattern B"
        ),
        "examples": (
            '- "Delete all inactive user accounts" — check admin permissions '
            "first\n"
            '- "Update the production database schema" — verify deployment '
            "permissions\n"
            '- "Access the client\'s financial records" — check authorization '
            "level\n"
            '- "Modify the billing plan for enterprise tier" — verify billing '
            "admin role"
        ),
    },
    "correction_handling": {
        "description": (
            'The user says "no, I meant X" or "that\'s wrong, try Y" after '
            "the model's first action. The model must adjust its approach "
            "without starting over — acknowledging the correction and revising "
            "the parameters or tool choice."
        ),
        "action_type": "call_tool (with corrected parameters)",
        "behavioral_pattern": (
            "1. User makes an initial request\n"
            "2. Assistant <think> block reasons about the request, calls a tool\n"
            "3. Tool returns a result\n"
            "4. User corrects: 'No, I meant X not Y' or 'Wrong customer, "
            "I need Z'\n"
            "5. Assistant <think> block acknowledges what went wrong, explains "
            "how the correction changes the approach\n"
            "6. Assistant makes the corrected tool call with updated parameters"
        ),
        "constraints": (
            "- The correction must be REALISTIC — wrong entity, wrong date "
            "range, wrong field, wrong action type\n"
            "- The assistant must NOT start the entire process over — it "
            "should acknowledge the specific error and adjust\n"
            "- <think> block MUST reference what went wrong and how the "
            "correction changes things\n"
            "- The corrected tool call must have visibly different parameters\n"
            "- Mix of correction types: entity (40%), parameter (30%), "
            "tool choice (30%)\n"
            "- action_type: always 'call_tool'"
        ),
        "examples": (
            '- "No, not Acme Corp — I meant Acme Industries"\n'
            '- "That\'s the wrong quarter, I need Q3 not Q4"\n'
            '- "I don\'t need the alert history, I need the metric trends"\n'
            '- "Wrong ticket — the ID is TK-4502, not TK-4520"'
        ),
    },
    "multi_turn_context": {
        "description": (
            "Extended 10+ turn conversations where the model must track "
            "evolving state across many turns. Earlier turns establish context "
            "that later turns reference. The model must maintain consistency "
            "and not 'forget' or contradict earlier information."
        ),
        "action_type": "varies (call_tool, clarify, answer_directly)",
        "behavioral_pattern": (
            "1. Turns 1-3: Establish the scenario — user provides context, "
            "model performs initial lookups\n"
            "2. Turns 4-6: Develop the thread — user asks follow-ups that "
            "BUILD on earlier results\n"
            "3. Turns 7-9: Introduce a twist — new information that changes "
            "or complicates the situation\n"
            "4. Turn 10+: Resolution — user asks something that requires "
            "synthesizing information from multiple earlier turns\n\n"
            "At least one turn must reference information from 3+ turns ago "
            "by name or value."
        ),
        "constraints": (
            "- MINIMUM 10 turns (user + assistant combined, not counting "
            "system or tool)\n"
            "- At least 3 tool calls across the conversation\n"
            "- At least one explicit back-reference to data from 3+ turns ago\n"
            "- The conversation must EVOLVE — not just be 10 independent "
            "questions\n"
            "- <think> blocks in later turns should reference earlier context\n"
            "- action_type: use the PRIMARY action type of the final/most "
            "important turn\n"
            "- Include realistic tool responses that build a narrative"
        ),
        "examples": (
            "- Turn 1: Look up customer → Turn 5: Check their recent "
            "tickets → Turn 10: 'Based on that account tier from earlier, "
            "what discount applies?'\n"
            "- Turn 1: Check inventory → Turn 4: Find supplier → Turn 8: "
            "'The warehouse from step 1 — is it near the supplier we found?'\n"
            "- Turn 1: Pull project status → Turn 6: Check budget → Turn 10: "
            "'Compare the timeline from the first lookup with the budget "
            "constraints'"
        ),
    },
}

# ---------------------------------------------------------------------------
# Enterprise contexts (42)
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
    """Load training schemas from the schema library."""
    with open(SCHEMA_FILE) as f:
        schemas = json.load(f)
    return schemas


def sample_tools(schemas: list, n: int = 3, rng: random.Random = None) -> list:
    """Sample n tools from schema list, ensuring domain diversity."""
    if rng is None:
        rng = random.Random()

    if len(schemas) <= n:
        return schemas[:n]

    # Group by domain for diversity
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
    """Format tools as JSON for inclusion in the prompt (strip internal metadata)."""
    clean = []
    for t in tools:
        clean.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        })
    return json.dumps(clean, indent=2)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_prompt(
    category: str,
    batch_size: int,
    start_id: int,
    tools: list,
    context: str,
) -> str:
    """Build the full generation prompt for a batch."""
    cat_def = CATEGORY_DEFINITIONS[category]
    tools_json = format_tools_for_prompt(tools)
    tool_names = [t["name"] for t in tools]

    # Build the prompt
    prompt = f"""You are generating multi-turn training data for an enterprise AI agent model. Each sample teaches the model correct decision-making with enterprise tools using Qwen3 chat format.

## Output Format

Output EXACTLY {batch_size} JSON objects, one per line. No markdown fences, no explanations, no commentary before or after. Each line must be a complete, valid JSON object.

## JSON Schema Per Sample

{{
  "id": "{category}_{{seq:04d}}",
  "category": "{category}",
  "messages": [
    {{"role": "system", "content": "<enterprise system prompt>"}},
    {{"role": "user", "content": "<user message>"}},
    {{"role": "assistant", "content": "<think>\\n...reasoning (400+ chars)...\\n</think>\\n\\n<response or tool call>"}},
    {{"role": "tool", "content": "<tool response JSON>"}},
    {{"role": "assistant", "content": "<think>\\n...\\n</think>\\n\\n<next response>"}},
    ...more turns as needed...
  ],
  "tools": {tools_json},
  "action_type": "<primary action type>",
  "source": "v3_categories"
}}

## Tool Call Format (Qwen3)

When the assistant calls a tool, the content MUST use this exact format:
<think>
...reasoning about which tool to use and why...
</think>

{{
  "name": "<tool_name>",
  "arguments": {{...}}
}}

The tool call JSON must NOT be wrapped in markdown fences. The tool name MUST be one of: {tool_names}

## Tool Response Format

Tool responses use role "tool" and contain a JSON string representing the API response.

## Category: {category}

### Definition
{cat_def['description']}

### Expected Action Type
{cat_def['action_type']}

### Behavioral Pattern
{cat_def['behavioral_pattern']}

### Constraints
{cat_def['constraints']}

### Example Scenarios
{cat_def['examples']}

## Enterprise Context

The scenario takes place in: {context}

## Available Tools

{tools_json}

## Requirements

1. Each sample must have a system prompt (1-5 sentences) appropriate for {context}
2. The <think> block in EVERY assistant turn must be at least 400 characters — show genuine multi-step reasoning about what to do and why
3. Tool calls must ONLY reference tools from the provided schema: {tool_names}
4. Tool call arguments must match the parameter schema (correct types, required fields present)
5. Include realistic tool responses (role: "tool") after each tool call
6. {"Minimum 10 user+assistant turns (not counting system or tool turns). The conversation must evolve across all turns with explicit back-references to earlier context." if category == "multi_turn_context" else "Minimum 5 messages total (including system, user, assistant, tool turns)"}
7. Vary the scenarios — do NOT reuse the same pattern across samples in this batch
8. System prompts should be diverse — vary length, tone, and specific policies
9. The action_type field must be one of: call_tool, clarify, escalate, refuse, answer_directly
10. Do NOT include any text before the first JSON object or after the last one

## TASK

Generate {batch_size} {category} training samples. IDs: {category}_{start_id:04d} through {category}_{start_id + batch_size - 1:04d}."""

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
                    print(f"    Empty output (attempt {attempt+1}), retrying...",
                          flush=True)
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
                print(f"    Parse failed (attempt {attempt+1}), retrying...",
                      flush=True)
                time.sleep(5)
                continue

            # Save raw output for debugging
            return [{"_raw": output[:2000], "_parse_error": True}]

        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRIES - 1:
                print(f"    Timeout (attempt {attempt+1}), retrying...",
                      flush=True)
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
        return CATEGORIES

    parts = shard_str.split("/")
    shard_num = int(parts[0])
    total_shards = int(parts[1])

    # Distribute categories round-robin
    return [
        cat for i, cat in enumerate(CATEGORIES)
        if (i % total_shards) + 1 == shard_num
    ]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_sample(sample: dict, category: str) -> tuple[bool, str]:
    """Validate a generated training sample."""
    if "_parse_error" in sample:
        return False, "parse_error"

    # Required fields
    required = ["id", "category", "messages", "tools", "action_type"]
    for field in required:
        if field not in sample:
            return False, f"missing_{field}"

    # messages must be a list
    messages = sample.get("messages", [])
    if not isinstance(messages, list):
        return False, "messages_not_list"

    # Minimum message count
    min_messages = 10 if category == "multi_turn_context" else 5
    if len(messages) < min_messages:
        return False, f"too_few_messages ({len(messages)} < {min_messages})"

    # Valid action type
    valid_actions = {"call_tool", "clarify", "escalate", "refuse", "answer_directly"}
    if sample.get("action_type") not in valid_actions:
        return False, f"invalid_action_type: {sample.get('action_type')}"

    # Check for <think> blocks in assistant messages
    has_think = False
    think_too_short = False
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if "<think>" in content and "</think>" in content:
                has_think = True
                # Extract think content
                start = content.find("<think>") + len("<think>")
                end = content.find("</think>")
                think_content = content[start:end].strip()
                if len(think_content) < 400:
                    think_too_short = True

    if not has_think:
        return False, "no_think_block"

    if think_too_short:
        return False, "think_block_too_short"

    # Check that tool names in assistant calls reference actual tools
    tool_names = set()
    for t in sample.get("tools", []):
        if isinstance(t, dict) and "name" in t:
            tool_names.add(t["name"])

    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Look for tool call pattern: {"name": "...", "arguments": ...}
            # Simple check: if content has "name" and "arguments", extract the name
            try:
                # Find JSON after </think>
                think_end = content.find("</think>")
                if think_end >= 0:
                    after_think = content[think_end + len("</think>"):].strip()
                    if after_think.startswith("{"):
                        call_obj = json.loads(after_think)
                        if "name" in call_obj and "arguments" in call_obj:
                            if call_obj["name"] not in tool_names:
                                return False, f"hallucinated_tool: {call_obj['name']}"
            except (json.JSONDecodeError, ValueError):
                pass  # Not a tool call or unparseable — that's ok

    # Check system message exists
    if not any(m.get("role") == "system" for m in messages):
        return False, "no_system_message"

    # Check source field
    if sample.get("source") != "v3_categories":
        # Auto-fix this rather than rejecting
        sample["source"] = "v3_categories"

    return True, "ok"


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate_category(
    category: str,
    schemas: list,
    limit: int = None,
):
    """Generate all training samples for one category."""
    target = CATEGORY_TARGETS[category]
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
    print(f"  [{category}] Have {existing_count}/{target}. "
          f"Generating {remaining} more...")

    # Seeded RNG for reproducible shuffling
    rng = random.Random(hash(category) + 2603)
    contexts = list(ENTERPRISE_CONTEXTS)
    rng.shuffle(contexts)
    context_idx = 0

    generated = existing_count
    batch_num = 0
    malformed_count = 0

    while generated < target:
        batch_size = min(BATCH_SIZE, target - generated)
        start_id = generated + 1

        # Sample 2-4 tools for this batch
        n_tools = rng.randint(2, 4)
        tools = sample_tools(schemas, n=n_tools, rng=rng)

        # Rotate enterprise context
        context = contexts[context_idx % len(contexts)]
        context_idx += 1

        # Build and send prompt
        prompt = build_prompt(category, batch_size, start_id, tools, context)

        tool_names_str = ", ".join(t["name"] for t in tools)
        print(
            f"    Batch {batch_num+1}: {category} "
            f"#{start_id:04d}-{start_id+batch_size-1:04d} "
            f"({n_tools} tools: {tool_names_str[:60]}..., "
            f"{context[:35]}...)",
            flush=True,
        )

        results = call_gemini(prompt, batch_size)

        if not results:
            print(f"    WARNING: Empty results for batch {batch_num+1}",
                  flush=True)
            batch_num += 1
            time.sleep(2)
            continue

        valid_count = 0
        for sample in results:
            is_valid, reason = validate_sample(sample, category)

            if is_valid:
                sample_id = sample.get("id", f"{category}_{start_id + valid_count:04d}")
                if sample_id in existing_ids:
                    sample_id = f"{category}_{generated + valid_count + 1:04d}"
                    sample["id"] = sample_id

                # Ensure metadata
                sample["category"] = category
                sample["source"] = "v3_categories"

                # Ensure tools are included if model omitted them
                if "tools" not in sample or not sample["tools"]:
                    sample["tools"] = [
                        {
                            "name": t["name"],
                            "description": t["description"],
                            "parameters": t["parameters"],
                        }
                        for t in tools
                    ]

                with open(output_file, "a") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                existing_ids.add(sample_id)
                valid_count += 1
            else:
                sample["_rejection_reason"] = reason
                sample["_category"] = category
                with open(malformed_file, "a") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                malformed_count += 1

        generated += valid_count
        batch_num += 1

        if valid_count < batch_size:
            print(f"    Got {valid_count}/{batch_size} valid "
                  f"(rejected: {batch_size - valid_count})", flush=True)

        # Brief pause between batches
        time.sleep(1)

    print(f"  [{category}] Done: {generated} valid, "
          f"{malformed_count} malformed.")
    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate V3 new category training data"
    )
    parser.add_argument(
        "--shard", type=str, default=None,
        help="Shard spec: '1/2', '2/2'"
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="Generate for single category"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max samples per category (for smoke testing)"
    )
    args = parser.parse_args()

    # Load schemas
    print("Loading schema library...", flush=True)
    schemas = load_schemas()
    print(f"  {len(schemas)} training schemas loaded")

    # Determine categories
    if args.category:
        if args.category not in CATEGORIES:
            print(f"ERROR: Unknown category '{args.category}'. "
                  f"Valid: {CATEGORIES}")
            sys.exit(1)
        categories = [args.category]
    else:
        categories = get_shard_categories(args.shard)

    print(f"\nCategories: {categories}")
    if args.shard:
        print(f"Shard: {args.shard}")
    if args.limit:
        print(f"Limit: {args.limit} per category")

    # Compute total target
    total_target = sum(
        min(CATEGORY_TARGETS[c], args.limit) if args.limit
        else CATEGORY_TARGETS[c]
        for c in categories
    )
    print(f"Total target: {total_target} samples\n")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate
    results = {}
    for category in categories:
        print(f"\n{'=' * 60}")
        print(f"Category: {category} (target: {CATEGORY_TARGETS[category]})")
        print(f"{'=' * 60}")
        count = generate_category(category, schemas, args.limit)
        results[category] = count

    # Summary
    print(f"\n{'=' * 60}")
    print("GENERATION COMPLETE")
    print(f"{'=' * 60}")

    total_valid = 0
    total_malformed = 0

    print(f"\n{'Category':<30} {'Valid':>8} {'Malformed':>10} {'Target':>8}")
    print("-" * 60)
    for cat in categories:
        cat_file = OUTPUT_DIR / f"{cat}.jsonl"
        mal_file = OUTPUT_DIR / f"{cat}.malformed.jsonl"

        valid = sum(1 for _ in open(cat_file)) if cat_file.exists() else 0
        malformed = sum(1 for _ in open(mal_file)) if mal_file.exists() else 0

        target = (
            min(CATEGORY_TARGETS[cat], args.limit) if args.limit
            else CATEGORY_TARGETS[cat]
        )

        status = "OK" if valid >= target else "INCOMPLETE"
        print(f"  {cat:<28} {valid:>6} {malformed:>10} {target:>8}  {status}")

        total_valid += valid
        total_malformed += malformed

    print("-" * 60)
    print(f"  {'TOTAL':<28} {total_valid:>6} {total_malformed:>10} "
          f"{total_target:>8}")
    print(f"\nOutput: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
