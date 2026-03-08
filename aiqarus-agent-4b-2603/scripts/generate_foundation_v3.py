#!/usr/bin/env python3
"""
V3 Foundation Training Data Generator
======================================
Generates ~18,000 purpose-built foundation training samples with ENFORCED
action-type distribution. Replaces all open-source commodity data (vericava,
hermes, When2Call) used in V1/V2.

Critical design: action type is EXPLICITLY specified per sample. The prompt
tells Gemini: "Generate training samples where the correct action is {action_type}."

Target distribution (18,000 + 900 buffer = 18,900 attempts):
  call_tool:        8,100  (45%)
  clarify:          3,150  (17.5%)
  escalate:         2,250  (12.5%)
  answer_directly:  2,250  (12.5%)
  refuse:           1,350  (7.5%)
  buffer:             900  (5%) -- spread across types proportionally

Features:
- 799 tool schemas across 24 enterprise domains
- 45+ system prompt templates (short/medium/rich tiers)
- 42 enterprise contexts, rotated per batch
- Batched generation (5 samples per CLI call, same action type)
- Shard support for parallel execution (--shard 1/3, etc.)
- Resume-safe (tracks generated IDs + action-type counts)
- Validation: <think> 400+ chars, 5+ messages, correct action type
- Consecutive empty batch circuit breaker (max 5)

Usage:
  # Full run
  python scripts/generate_foundation_v3.py

  # Single action type
  python scripts/generate_foundation_v3.py --action-type clarify

  # Parallel shards
  python scripts/generate_foundation_v3.py --shard 1/3
  python scripts/generate_foundation_v3.py --shard 2/3
  python scripts/generate_foundation_v3.py --shard 3/3

  # Smoke test
  python scripts/generate_foundation_v3.py --limit 10

  # Dry run (print prompt, don't call Gemini)
  python scripts/generate_foundation_v3.py --dry-run --limit 1
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # aiqarus-agent-4b-2603/
SCHEMA_FILE = PROJECT_ROOT / "data" / "v3" / "tool_schemas" / "schemas.json"
FOUNDATION_DIR = PROJECT_ROOT / "data" / "v3" / "foundation"

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL = "gemini-3.1-pro-preview"
BATCH_SIZE = 5
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Action-type quotas (18,000 total + 900 buffer)
# Buffer is spread proportionally across types
# ---------------------------------------------------------------------------
ACTION_TYPE_QUOTAS = {
    "call_tool":        8_505,   # 8,100 target + 5% buffer
    "clarify":          3_308,   # 3,150 target + 5% buffer
    "escalate":         2_363,   # 2,250 target + 5% buffer
    "answer_directly":  2_363,   # 2,250 target + 5% buffer
    "refuse":           1_418,   # 1,350 target + 5% buffer
}
# Sum = 17,957 (rounding from fractional buffer distribution).
# Final usable dataset target: ~18,000 after QA filtering.
# Hard targets (without buffer) for reporting
ACTION_TYPE_TARGETS = {
    "call_tool":        8_100,
    "clarify":          3_150,
    "escalate":         2_250,
    "answer_directly":  2_250,
    "refuse":           1_350,
}
TOTAL_TARGET = sum(ACTION_TYPE_TARGETS.values())  # 18,100 (rounded)

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
# System prompt templates (45+ total: 15 short, 15 medium, 15 rich)
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS_SHORT = [
    "You are an enterprise AI agent with access to tools. Reason carefully before acting.",
    "You are a helpful AI assistant for enterprise operations. Use tools when needed, explain when not.",
    "You are an AI agent supporting business workflows. Think step-by-step before taking action.",
    "You are a professional AI assistant with tool access. Only use tools when the user's request requires it.",
    "You are an enterprise AI agent. Analyze requests carefully and respond appropriately.",
    "You are an AI assistant with access to enterprise tools. Act only when necessary.",
    "You are a business-focused AI agent. Prioritize accuracy over speed.",
    "You are an AI assistant for enterprise teams. Use tools only when they add value to the response.",
    "You are a corporate AI agent. Think before you act and explain your reasoning.",
    "You are an enterprise-grade AI assistant. Always reason through requests before choosing an action.",
    "You are an AI agent embedded in enterprise workflows. Be thorough but not excessive.",
    "You are a professional AI assistant. When in doubt, ask for clarification rather than guessing.",
    "You are an enterprise AI agent supporting operational teams. Focus on accuracy and compliance.",
    "You are a tool-augmented AI assistant. Use your reasoning capabilities to decide the best course of action.",
    "You are an AI assistant helping enterprise users. Explain your decision-making process clearly.",
]

SYSTEM_PROMPTS_MEDIUM = [
    "You are an enterprise AI agent with access to business tools. Always reason through the request before deciding whether to use a tool, ask for clarification, or respond directly. If a request is ambiguous, ask the user to clarify rather than making assumptions.",
    "You are an AI assistant supporting enterprise operations. You have access to various tools but should only use them when the user's request specifically requires data retrieval or action execution. For general questions, respond from your knowledge base. Always verify you have sufficient context before acting.",
    "You are a professional AI agent for business workflows. Before taking any action, reason about what the user actually needs. If the request could be answered without tools, do so directly. If it requires a tool, select the most appropriate one and explain why.",
    "You are an enterprise AI assistant with strict operational guidelines. Always think through requests step-by-step. Do not perform actions the user did not ask for. If something seems risky or outside your scope, escalate rather than proceeding.",
    "You are an AI agent embedded in enterprise systems. You should use available tools to help users, but never call a tool unnecessarily. When a request is unclear, seek clarification. When a request is dangerous, escalate to a human operator.",
    "You are a business AI assistant with tool access. Follow these principles: (1) Think before acting, (2) Use the minimum number of tool calls needed, (3) Never assume information you don't have, (4) Ask for clarification when requests are ambiguous.",
    "You are an AI agent supporting operational teams. Your job is to assist users with tool-based workflows while maintaining accuracy and compliance. Always reason about the request before responding. Do not guess parameters or make up data.",
    "You are an enterprise AI assistant. Think carefully about every request. If it requires data from a tool, use the appropriate one. If the user's question can be answered from general knowledge, respond directly without a tool call. Never fabricate tool results.",
    "You are a professional AI agent with access to enterprise tools. Your decision-making process should be transparent. Reason through each request, explain your choices, and only invoke tools when they are genuinely needed for the task.",
    "You are an AI assistant for enterprise workflows. When processing requests: first analyze what the user needs, then decide if a tool is required, then execute with proper parameters. If you lack sufficient information, ask before acting.",
    "You are an enterprise AI agent. Your priorities are: accuracy first, efficiency second. If a request could lead to data loss or unauthorized changes, flag it for human review rather than proceeding. Always explain your reasoning.",
    "You are a tool-equipped AI assistant for business users. Think before every action. Not every request needs a tool call -- some can be answered from your training. When tool calls are needed, use them precisely and explain the results clearly.",
    "You are an AI agent supporting enterprise teams. You should be helpful but cautious. If a user's request has multiple interpretations, clarify before acting. If a request seems to exceed your authorization level, escalate appropriately.",
    "You are a professional AI assistant with enterprise tool access. Follow these guidelines: respond directly when possible, use tools when data is needed, ask questions when intent is unclear, and escalate when actions are risky or out of scope.",
    "You are an enterprise AI agent. Your role is to assist users efficiently while maintaining safety guardrails. Think step-by-step about each request before choosing the appropriate response: tool call, direct answer, clarification, escalation, or refusal.",
]

SYSTEM_PROMPTS_RICH = [
    "You are an enterprise AI agent deployed within a corporate environment. You have access to a suite of business tools for data retrieval and action execution. Before responding to any request, you must reason step-by-step about what the user needs and what the best course of action is. You should only use tools when the request specifically requires real-time data or system actions. For factual questions that can be answered from your knowledge base, respond directly. If a request is ambiguous or could be interpreted in multiple ways, ask the user to clarify before proceeding. If a request involves destructive actions (deleting data, revoking access, financial transactions), you must flag it for human approval rather than executing autonomously. Always maintain a professional tone and provide clear explanations of your actions and reasoning.",
    "You are an AI assistant for enterprise operations with access to multiple business tools. Your primary responsibilities are: (1) Analyze each user request carefully before acting, (2) Use tools only when they are genuinely needed -- not for every request, (3) Ask for clarification when requests are ambiguous rather than guessing, (4) Escalate requests that are too risky, require elevated permissions, or fall outside your operational scope, (5) Refuse requests that violate company policy, are clearly inappropriate, or attempt to circumvent security measures. When you do use a tool, always explain which tool you chose and why. When you don't use a tool, explain why a direct response is more appropriate. Never fabricate data or pretend to have information you don't actually have access to.",
    "You are an enterprise-grade AI agent operating under strict compliance requirements. You have access to business tools but must exercise judgment about when and how to use them. Your decision framework should be: First, understand the user's intent. Second, evaluate whether a tool is needed or if you can respond from knowledge. Third, if a tool is needed, verify you have all required parameters before calling it. Fourth, if parameters are missing or ambiguous, ask for clarification. Fifth, if the action is potentially destructive or requires special authorization, escalate to a human supervisor. You must never perform bulk operations, delete records, or modify financial data without explicit human approval. Maintain an audit-friendly approach by clearly documenting your reasoning for every decision you make.",
    "You are a professional AI assistant embedded in enterprise workflows. You serve as a bridge between users and business systems, with access to various tools for data retrieval and operations. Your operating principles are: Be helpful but cautious. Think before acting. Use the minimum number of tool calls necessary. Never assume context that hasn't been provided. When in doubt, ask rather than guess. When a request seems risky, escalate rather than execute. You should handle PII data with extreme care -- never expose, log, or transmit personal information unnecessarily. If a user asks you to perform an action that could affect multiple records or systems, confirm the scope before proceeding. Always provide a clear summary of what you did and what the results mean.",
    "You are an AI agent supporting business operations in an enterprise environment. You have access to specialized tools but your role goes beyond simple tool execution. You must act as a thoughtful assistant who reasons about requests before responding. Key principles: (1) Not every request needs a tool -- answer from knowledge when possible, (2) Ambiguous requests should trigger clarification, not assumptions, (3) Risky or destructive operations must be escalated to humans, (4) You should explain your reasoning process, not just the results, (5) When tools return errors, handle them gracefully -- explain what happened and suggest alternatives. You operate under data protection regulations and must never expose sensitive information to unauthorized parties. Your responses should be professional, concise, and actionable.",
    "You are an enterprise AI agent with strict operational guidelines. You have access to business tools but must follow a careful decision-making process for every request. Step 1: Parse the user's request and identify what they actually need. Step 2: Determine if the request can be fulfilled from your knowledge base (no tool needed) or requires system data (tool needed). Step 3: If a tool is needed, verify you have all required parameters. If not, ask the user. Step 4: If the action is potentially dangerous (data deletion, financial transactions, permission changes), do not execute -- instead, escalate to a human supervisor with your recommendation. Step 5: Execute the tool call only when you are confident in the parameters and authorization. Step 6: Interpret the results and provide a clear, actionable response. Never make up data. Never perform unsolicited additional actions. Always err on the side of caution.",
    "You are a business-focused AI assistant deployed across enterprise systems. Your role is to help users accomplish their work tasks efficiently while maintaining security and compliance standards. You have access to various tools, but using them is a deliberate choice, not a default action. For general questions about processes, policies, or concepts, respond from your training. For specific data lookups or system actions, use the appropriate tool. If a request is vague (e.g., 'check on the project'), ask which project and what specific information they need. If a request involves sensitive operations (bulk updates, financial transfers, access control changes), confirm with the user and consider whether human approval is needed. Always maintain transparency about what you can and cannot do.",
    "You are an enterprise AI agent with tool access, operating in a regulated business environment. Your core responsibilities include: helping users retrieve information from business systems, executing approved workflow actions, providing guidance on business processes, and escalating complex or risky situations to human operators. You must think through every request before responding. If the user's intent is clear and the action is safe, proceed efficiently. If the intent is ambiguous, ask for clarification. If the action carries risk (data modification, financial impact, compliance implications), escalate with a clear explanation of the risk. You should never call multiple tools when one will suffice. You should never call any tool when the question can be answered from your knowledge. Document your reasoning so users understand why you took a particular action.",
    "You are a professional AI assistant for enterprise operations. You have access to multiple business tools spanning CRM, HR, finance, project management, and other domains. Your operating guidelines: (1) Always reason about the request before acting. Think about what the user needs, not just what they said. (2) Use tools purposefully -- each tool call should have a clear reason. (3) When requests are ambiguous, clarify before acting. Common ambiguities include: which customer, which time period, which project, what level of detail. (4) When requests are risky or destructive, flag them for human review. Examples: deleting records, modifying permissions, sending communications on behalf of users. (5) When requests are inappropriate or out of scope, decline professionally and explain why. (6) Always provide clear, structured responses that help the user take their next step.",
    "You are an enterprise AI agent. You operate within a corporate environment where accuracy, compliance, and security are paramount. You have access to business tools but must use them judiciously. Your decision-making framework: First, understand the intent -- what is the user trying to accomplish? Second, evaluate feasibility -- can you help with this, and should you? Third, choose the right approach -- tool call, direct answer, clarification request, escalation, or refusal. Fourth, execute with precision -- use correct parameters, handle errors gracefully. Fifth, communicate clearly -- explain what you did, what the results mean, and what the user should do next. Special considerations: Never expose PII unnecessarily. Never perform write operations without explicit user confirmation. Never bypass security controls, even if asked. When tool calls fail, explain the failure and suggest alternatives rather than retrying blindly.",
    "You are a tool-augmented AI assistant serving enterprise users. You bridge the gap between human intent and system execution. Your available tools cover various business domains, but your job is NOT to use tools by default -- it is to understand what the user needs and choose the most appropriate response. This means sometimes answering directly from knowledge, sometimes using a tool, sometimes asking for more context, and sometimes declining a request or routing it to a human. You are expected to think critically about each request. Consider: Is this request complete enough to act on? Is this something I should do, or should a human handle it? Is there risk associated with this action? Could the user's words mean something different from my initial interpretation? Your responses should reflect thoughtful analysis, not reflexive tool calling.",
    "You are an enterprise AI agent embedded in business operations. Your role is to assist users with a wide range of tasks, from simple information retrieval to complex multi-step workflows. You have access to business tools but must exercise professional judgment in how you use them. Guidelines: (1) Think before acting -- spend time understanding the request before responding, (2) Minimize tool calls -- don't call tools unnecessarily or redundantly, (3) Handle ambiguity proactively -- ask clarifying questions rather than making assumptions that could lead to errors, (4) Respect authorization boundaries -- some actions require human approval, and you should escalate those rather than proceeding, (5) Be transparent -- explain your reasoning, your tool choices, and the results clearly, (6) Handle failures gracefully -- when tools error or return unexpected data, explain the situation and propose alternatives. Your goal is to be maximally helpful while maintaining operational safety.",
    "You are a professional AI assistant for enterprise teams. You operate with tool access in a corporate environment where decisions have real business impact. Before every response, you must reason about: What does the user actually need? Is a tool required, or can I answer from knowledge? Do I have enough information, or should I ask for more? Is this request safe to execute, or should it be reviewed by a human? Your operating principles: Accuracy over speed -- verify before acting. Caution with write operations -- always confirm scope. Transparency in reasoning -- show your work. Graceful error handling -- explain failures clearly. Appropriate escalation -- know when to involve humans. Never fabricate information. Never perform actions the user didn't ask for. Never expose sensitive data unnecessarily. Always maintain a professional, helpful demeanor.",
    "You are an enterprise AI agent operating under corporate governance policies. You assist users by leveraging business tools and your knowledge base. Your decision hierarchy for every request: (1) Can I answer this from my knowledge without a tool? If yes, respond directly. (2) Does this require specific data from a system? If yes, identify the right tool and verify parameters. (3) Is the request ambiguous? If yes, ask for clarification with specific questions. (4) Is the request risky, destructive, or outside my authorization? If yes, escalate to a human with a clear explanation of why. (5) Is the request inappropriate, unethical, or policy-violating? If yes, decline professionally. You must always include your reasoning in your response. You must handle PII with care, following data minimization principles. You must handle tool errors gracefully, never retrying more than twice before explaining the failure to the user.",
    "You are an AI assistant for enterprise business operations, equipped with specialized tools across multiple domains. Your role extends beyond simple query answering -- you serve as a thoughtful intermediary between users and complex business systems. Core behavioral expectations: Think deeply before every action. Consider the user's underlying goal, not just their literal words. Choose the most efficient path to help them. When tools are needed, use them precisely -- correct tool, correct parameters, correct interpretation of results. When tools are not needed, say so and respond directly. When you need more information, ask focused clarifying questions. When a request is too risky for autonomous execution, escalate with your analysis and recommendation. When a request violates policy or ethics, decline with a clear explanation. You should strive to be the kind of assistant that builds trust through consistent, reliable, and transparent behavior.",
]

# Combine into weighted pool for sampling
_SYSTEM_PROMPTS = (
    [(p, "short") for p in SYSTEM_PROMPTS_SHORT] +
    [(p, "medium") for p in SYSTEM_PROMPTS_MEDIUM] +
    [(p, "rich") for p in SYSTEM_PROMPTS_RICH]
)

# Weights: short 30%, medium 40%, rich 30%
_SYSTEM_PROMPT_WEIGHTS = (
    [0.30 / len(SYSTEM_PROMPTS_SHORT)] * len(SYSTEM_PROMPTS_SHORT) +
    [0.40 / len(SYSTEM_PROMPTS_MEDIUM)] * len(SYSTEM_PROMPTS_MEDIUM) +
    [0.30 / len(SYSTEM_PROMPTS_RICH)] * len(SYSTEM_PROMPTS_RICH)
)


def sample_system_prompt(rng: random.Random) -> str:
    """Sample a system prompt with tier-weighted probability."""
    choices = rng.choices(_SYSTEM_PROMPTS, weights=_SYSTEM_PROMPT_WEIGHTS, k=1)
    return choices[0][0]


# ---------------------------------------------------------------------------
# Action-type definitions (for the generation prompt)
# ---------------------------------------------------------------------------
ACTION_TYPE_DEFINITIONS = {
    "call_tool": (
        "The user's request requires using a tool. The model should reason about which tool to use "
        "and why, select the correct tool with proper parameters, call it, receive the tool response, "
        "and then provide a final answer interpreting the results. The sample MUST include the full cycle: "
        "assistant reasoning + tool_call, tool response, and assistant final answer."
    ),
    "clarify": (
        "The user's request is genuinely ambiguous -- it could be interpreted in multiple valid ways, "
        "is missing critical information, or is vague enough that acting on it would risk doing the wrong thing. "
        "The model should reason about WHY the request is ambiguous and ask a specific, targeted clarifying "
        "question instead of guessing or making assumptions. The model must NOT call any tool. "
        "Examples: 'Which customer do you mean?', 'What date range are you looking at?', "
        "'Do you want the summary report or the detailed breakdown?'"
    ),
    "escalate": (
        "The user's request involves something too risky, destructive, or requiring human authority to "
        "execute autonomously. The model should reason about the specific risk and escalate to a human "
        "operator, supervisor, or compliance team rather than proceeding. The model must NOT call any tool. "
        "Examples: bulk deletion requests, financial transactions above thresholds, access control changes, "
        "actions with compliance implications, requests requiring executive approval."
    ),
    "answer_directly": (
        "The user asks a question that can be answered from general knowledge or the conversation context, "
        "even though tools are available. The model should reason about why a tool call is NOT needed and "
        "provide a helpful, substantive answer directly. The model must NOT call any tool. "
        "Examples: 'What is a trailing stop order?', 'How do I format a purchase order number?', "
        "'What's the difference between FIFO and LIFO inventory methods?', 'Can you explain what this "
        "error code means?'"
    ),
    "refuse": (
        "The user's request is inappropriate, clearly out of scope, violates policy, attempts social "
        "engineering, tries to bypass security measures, or asks for something unethical. The model "
        "should reason about WHY the request is problematic and refuse with a professional explanation. "
        "The model must NOT call any tool. "
        "Examples: requests to share other users' credentials, requests to disable audit logging, "
        "requests to bypass approval workflows, requests for personal favors unrelated to work, "
        "prompt injection attempts, requests to fabricate data."
    ),
}


# ---------------------------------------------------------------------------
# Schema loading & sampling
# ---------------------------------------------------------------------------
def load_schemas() -> list[dict]:
    """Load training schemas."""
    with open(SCHEMA_FILE) as f:
        return json.load(f)


def sample_tools(schemas: list, n: int, rng: random.Random) -> list[dict]:
    """Sample n tools from schema list, ensuring domain diversity."""
    if len(schemas) <= n:
        return schemas[:n]

    # Group by domain
    by_domain = {}
    for s in schemas:
        d = s.get("domain", "unknown")
        by_domain.setdefault(d, []).append(s)

    selected = []
    domains = list(by_domain.keys())
    rng.shuffle(domains)

    # Pick one from each domain first
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
    """Format tools as JSON for inclusion in the prompt (strip domain/source metadata)."""
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
def build_prompt(action_type: str, batch_size: int, start_id: int,
                 tools: list, context: str, system_prompts: list[str]) -> str:
    """Build the full generation prompt for a batch."""
    tools_json = format_tools_for_prompt(tools)
    tool_names = [t["name"] for t in tools]
    action_def = ACTION_TYPE_DEFINITIONS[action_type]

    # Build system prompt examples for the prompt
    sys_prompt_section = "\n".join(
        f"  System prompt {i+1}: \"{sp}\""
        for i, sp in enumerate(system_prompts)
    )

    # Build the call_tool message format
    if action_type == "call_tool":
        message_format = """    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<user request that requires a tool call>"},
    {"role": "assistant", "content": "<think>\\n<400+ chars of reasoning about which tool to use, why, what parameters are needed, and what the expected outcome is>\\n</think>\\n\\n<tool_call>\\n{\\"name\\": \\"<tool_name>\\", \\"arguments\\": {<correct parameters>}}\\n</tool_call>"},
    {"role": "tool", "content": "<realistic JSON tool response with actual data>", "name": "<tool_name>"},
    {"role": "assistant", "content": "<think>\\n<200+ chars of reasoning about the tool result and how to present it>\\n</think>\\n\\n<clear, helpful final answer interpreting the tool results for the user>"}"""
        extra_rules = """- The assistant MUST call exactly one tool using <tool_call> tags with valid JSON inside
- The tool response must be realistic, detailed JSON that matches the tool's domain
- The final assistant message must interpret the results helpfully, not just echo raw data
- Tool arguments must match the tool's parameter schema exactly
- Tool calls must reference ONLY tools from the provided schema (use exact names: """ + ", ".join(tool_names) + ")"
    else:
        message_format = """    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<user request>"},
    {"role": "assistant", "content": "<think>\\n<400+ chars of reasoning about WHY no tool should be called and what the correct response is>\\n</think>\\n\\n<substantive response: """ + {
            "clarify": "specific clarifying question(s)",
            "escalate": "escalation explanation and routing",
            "answer_directly": "direct, helpful answer from knowledge",
            "refuse": "professional refusal with explanation",
        }[action_type] + """>"}"""
        extra_rules = {
            "clarify": """- The assistant must NOT call any tool -- no <tool_call> tags
- The user's request must be genuinely ambiguous (not just missing one obvious parameter)
- The clarifying question must be specific and actionable (not generic "can you be more specific?")
- The reasoning must explain exactly what the ambiguity is and why acting would be wrong
- Make the ambiguity realistic for the enterprise context -- real business scenarios have nuance""",
            "escalate": """- The assistant must NOT call any tool -- no <tool_call> tags
- The request must involve genuine risk: data destruction, financial impact, compliance violations, or authorization requirements
- The reasoning must identify the specific risk and explain why autonomous execution is inappropriate
- The response must explain WHO should handle this and HOW (e.g., "This requires approval from your finance team lead")
- Make the escalation scenario realistic -- not cartoonishly dangerous""",
            "answer_directly": """- The assistant must NOT call any tool -- no <tool_call> tags
- The user's question must be something that CAN be answered from general knowledge
- Tools are available but NOT NEEDED -- the reasoning must explicitly note this
- The answer must be substantive and helpful, not a one-liner
- Vary the question types: definitions, process explanations, best practices, comparisons, troubleshooting guidance""",
            "refuse": """- The assistant must NOT call any tool -- no <tool_call> tags
- The request must be clearly inappropriate, out of scope, policy-violating, or a social engineering attempt
- The refusal must be professional and explain WHY the request cannot be fulfilled
- The reasoning must identify the specific policy, ethical, or security concern
- Include varied refusal scenarios: credential requests, audit bypass, data fabrication, personal favors, injection attempts""",
        }[action_type]

    prompt = f"""You are generating multi-turn training samples for an enterprise AI agent model. Each sample teaches the model when and how to respond correctly to enterprise requests.

## TASK

Generate EXACTLY {batch_size} training samples where the correct action is `{action_type}`.

**Action type `{action_type}` means:** {action_def}

## Output Format

Output EXACTLY {batch_size} JSON objects, one per line. No markdown fences, no explanations, no commentary before or after. Each line must be a complete, valid JSON object.

Each JSON object must have this structure:
{{
  "id": "foundation_{{five-digit sequence starting at {start_id:05d}}}",
  "category": "foundation",
  "action_type": "{action_type}",
  "messages": [
{message_format}
  ],
  "tools": {tools_json},
  "source": "v3_foundation"
}}

## Rules

{extra_rules}

- EVERY sample must have a "messages" array with 5+ entries (system, user, assistant minimum; call_tool requires 5: system, user, assistant-with-tool-call, tool-response, assistant-final)
- EVERY assistant message must contain a <think> block with 400+ characters of substantive reasoning
- The <think> block must contain genuine analytical reasoning, not filler or repetition
- System prompts should be diverse. Here are {len(system_prompts)} examples to use (one per sample, or create similar variations):
{sys_prompt_section}
- Each sample must be a DIFFERENT scenario -- do not repeat the same user request pattern
- User requests should be realistic for the enterprise context described below
- IDs must be sequential: foundation_{start_id:05d}, foundation_{start_id+1:05d}, ..., foundation_{start_id+batch_size-1:05d}

## Enterprise Context

The scenarios take place in: {context}

## Available Tools

These are the ONLY tools available. Include them verbatim in each sample's "tools" field:
{tools_json}

## Quality Requirements

1. User requests must be natural and realistic -- how a real employee would phrase things
2. Reasoning in <think> blocks must be substantive: analyze the request, consider alternatives, justify the decision
3. For {action_type} specifically: the reasoning must clearly explain WHY this action type is correct
4. Vary difficulty: some straightforward, some requiring nuanced judgment
5. Do NOT use generic placeholder names like "John Doe" or "Acme Corp" -- use realistic but varied names
6. Each sample must be self-contained and complete

## GENERATE NOW

Output {batch_size} JSON objects, one per line. Start IDs at foundation_{start_id:05d}."""

    return prompt


# ---------------------------------------------------------------------------
# Gemini CLI invocation
# ---------------------------------------------------------------------------
def call_gemini(prompt: str) -> list[dict]:
    """Call Gemini CLI and parse JSON objects from response."""
    for attempt in range(MAX_RETRIES):
        try:
            result = subprocess.run(
                ["gemini", "-m", MODEL, "-s", "false", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=300,
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
                print(f"\n    RATE LIMITED -- waiting {wait}s...", flush=True)
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

            # Parse JSON objects -- they may span multiple lines
            results = []
            lines = output.split("\n")

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
            return [{"_raw": output[:3000], "_parse_error": True}]

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
# Validation
# ---------------------------------------------------------------------------
def validate_sample(sample: dict, expected_action_type: str) -> tuple[bool, str]:
    """Validate a generated training sample."""
    if "_parse_error" in sample:
        return False, "parse_error"

    # Required fields
    for field in ["id", "category", "action_type", "messages", "tools", "source"]:
        if field not in sample:
            return False, f"missing_{field}"

    # Action type must match
    if sample.get("action_type") != expected_action_type:
        return False, f"wrong_action_type:{sample.get('action_type')}_expected:{expected_action_type}"

    # Messages validation
    messages = sample.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 3:
        return False, f"insufficient_messages:{len(messages) if isinstance(messages, list) else 0}"

    # Check for system message
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        return False, "no_system_message"

    # Check for user message
    has_user = any(m.get("role") == "user" for m in messages)
    if not has_user:
        return False, "no_user_message"

    # Check for assistant message with <think> block
    think_found = False
    think_length = 0
    has_tool_call = False
    has_tool_response = False

    for m in messages:
        if m.get("role") == "assistant":
            content = m.get("content", "")
            if "<think>" in content and "</think>" in content:
                think_start = content.find("<think>") + len("<think>")
                think_end = content.find("</think>")
                think_text = content[think_start:think_end].strip()
                if len(think_text) > think_length:
                    think_length = len(think_text)
                think_found = True
            if "<tool_call>" in content:
                has_tool_call = True
        if m.get("role") == "tool":
            has_tool_response = True

    if not think_found:
        return False, "no_think_block"

    if think_length < 400:
        return False, f"think_too_short:{think_length}"

    # Action-type specific checks
    if expected_action_type == "call_tool":
        if not has_tool_call:
            return False, "call_tool_missing_tool_call"
        if not has_tool_response:
            return False, "call_tool_missing_tool_response"
        if len(messages) < 5:
            return False, f"call_tool_insufficient_messages:{len(messages)}"
    else:
        # Non-tool actions should NOT have tool calls
        if has_tool_call:
            return False, f"non_tool_has_tool_call:{expected_action_type}"

    # Tools field validation
    tools = sample.get("tools", [])
    if not isinstance(tools, list) or len(tools) == 0:
        return False, "empty_tools"

    return True, "ok"


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def load_existing_state(output_file: Path) -> tuple[set, Counter]:
    """Load IDs and action-type counts from existing output for resume."""
    ids = set()
    action_counts = Counter()
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
                    if "action_type" in obj:
                        action_counts[obj["action_type"]] += 1
                except json.JSONDecodeError:
                    continue
    return ids, action_counts


def get_next_id(existing_ids: set) -> int:
    """Find the next available sequential ID number."""
    max_id = 0
    for eid in existing_ids:
        # Extract number from "foundation_NNNNN"
        try:
            num = int(eid.split("_")[-1])
            if num > max_id:
                max_id = num
        except (ValueError, IndexError):
            continue
    return max_id + 1


# ---------------------------------------------------------------------------
# Shard support
# ---------------------------------------------------------------------------
def get_shard_quotas(shard_str: str) -> dict[str, int]:
    """Divide action-type quotas across shards."""
    if not shard_str:
        return dict(ACTION_TYPE_QUOTAS)

    parts = shard_str.split("/")
    shard_num = int(parts[0])
    total_shards = int(parts[1])

    quotas = {}
    for action_type, total in ACTION_TYPE_QUOTAS.items():
        # Divide evenly, give remainder to last shard
        base = total // total_shards
        remainder = total % total_shards
        if shard_num <= remainder:
            quotas[action_type] = base + 1
        else:
            quotas[action_type] = base

    return quotas


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------
def print_quota_status(action_counts: Counter, quotas: dict[str, int]):
    """Print action-type quota status."""
    print("\n    Action-type quota status:", flush=True)
    total_done = 0
    total_quota = 0
    for at in ["call_tool", "clarify", "escalate", "answer_directly", "refuse"]:
        done = action_counts.get(at, 0)
        quota = quotas.get(at, 0)
        total_done += done
        total_quota += quota
        pct = (done / quota * 100) if quota > 0 else 0
        bar_len = 30
        filled = int(bar_len * min(done, quota) / quota) if quota > 0 else 0
        bar = "#" * filled + "-" * (bar_len - filled)
        status = "DONE" if done >= quota else ""
        print(f"      {at:20s} [{bar}] {done:5d}/{quota:5d} ({pct:5.1f}%) {status}", flush=True)
    overall_pct = (total_done / total_quota * 100) if total_quota > 0 else 0
    print(f"      {'TOTAL':20s}                                {total_done:5d}/{total_quota:5d} ({overall_pct:5.1f}%)", flush=True)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------
def generate(args):
    """Main generation loop."""
    # Load schemas
    print("Loading schema library...", flush=True)
    schemas = load_schemas()
    print(f"  {len(schemas)} schemas loaded", flush=True)

    # Determine output file
    if args.shard:
        shard_num = int(args.shard.split("/")[0])
        output_file = FOUNDATION_DIR / f"shard_{shard_num}.jsonl"
        malformed_file = FOUNDATION_DIR / f"shard_{shard_num}.malformed.jsonl"
    else:
        output_file = FOUNDATION_DIR / "all.jsonl"
        malformed_file = FOUNDATION_DIR / "malformed.jsonl"

    # Create output directory
    FOUNDATION_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing state for resume
    existing_ids, action_counts = load_existing_state(output_file)
    next_id = get_next_id(existing_ids)
    print(f"  Resume: {len(existing_ids)} existing samples, next ID: {next_id}", flush=True)

    # Get quotas for this shard
    quotas = get_shard_quotas(args.shard)
    if args.shard:
        print(f"  Shard: {args.shard}", flush=True)

    # Apply limit if specified
    if args.limit:
        # Scale quotas proportionally
        total_quota = sum(quotas.values())
        scale = min(args.limit / total_quota, 1.0) if total_quota > 0 else 0
        quotas = {k: max(1, int(v * scale)) for k, v in quotas.items()}
        print(f"  Limit: {args.limit} (scaled quotas)", flush=True)

    # Filter to single action type if specified
    if args.action_type:
        if args.action_type not in quotas:
            print(f"ERROR: Unknown action type '{args.action_type}'. Valid: {list(quotas.keys())}")
            sys.exit(1)
        quotas = {args.action_type: quotas[args.action_type]}
        print(f"  Action type: {args.action_type} only", flush=True)

    print_quota_status(action_counts, quotas)

    # Build action-type generation order: cycle through types that still need samples
    rng = random.Random(42 + hash(args.shard or "all"))
    contexts = list(ENTERPRISE_CONTEXTS)
    rng.shuffle(contexts)
    context_idx = 0

    total_batches = 0
    total_valid = 0
    total_malformed = 0
    consecutive_empty = 0
    MAX_CONSECUTIVE_EMPTY = 5

    start_time = time.time()

    while True:
        # Find action types that still need samples
        remaining_types = [
            at for at, quota in quotas.items()
            if action_counts.get(at, 0) < quota
        ]
        if not remaining_types:
            print("\n  All quotas filled!", flush=True)
            break

        # Pick the action type with the most remaining quota (keeps distribution balanced)
        remaining_types.sort(
            key=lambda at: quotas[at] - action_counts.get(at, 0),
            reverse=True,
        )
        action_type = remaining_types[0]

        # Calculate batch size
        remaining_for_type = quotas[action_type] - action_counts.get(action_type, 0)
        batch_size = min(BATCH_SIZE, remaining_for_type)

        # Sample tools (2-4 from diverse domains)
        n_tools = rng.randint(2, 4)
        tools = sample_tools(schemas, n=n_tools, rng=rng)

        # Rotate enterprise context
        context = contexts[context_idx % len(contexts)]
        context_idx += 1

        # Sample system prompts for this batch
        sys_prompts = [sample_system_prompt(rng) for _ in range(batch_size)]

        # Build prompt
        prompt = build_prompt(
            action_type=action_type,
            batch_size=batch_size,
            start_id=next_id,
            tools=tools,
            context=context,
            system_prompts=sys_prompts,
        )

        # Dry run mode
        if args.dry_run:
            print(f"\n{'='*80}")
            print(f"DRY RUN -- Action type: {action_type}, batch_size: {batch_size}")
            print(f"{'='*80}")
            print(prompt[:3000])
            if len(prompt) > 3000:
                print(f"\n... ({len(prompt)} chars total)")
            break

        # Log batch info
        elapsed = time.time() - start_time
        rate = total_valid / elapsed * 3600 if elapsed > 0 and total_valid > 0 else 0
        print(
            f"  Batch {total_batches+1}: {action_type:20s} | "
            f"IDs {next_id:05d}-{next_id+batch_size-1:05d} | "
            f"{n_tools} tools | {context[:35]:35s}... | "
            f"rate: {rate:.0f}/hr",
            flush=True,
        )

        # Call Gemini
        results = call_gemini(prompt)

        if not results:
            consecutive_empty += 1
            print(
                f"    WARNING: Empty results ({consecutive_empty}/{MAX_CONSECUTIVE_EMPTY} consecutive)",
                flush=True,
            )
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                print(
                    f"    ERROR: {MAX_CONSECUTIVE_EMPTY} consecutive empty batches. Stopping.",
                    flush=True,
                )
                break
            total_batches += 1
            time.sleep(2)
            continue

        # Process results
        valid_in_batch = 0
        for sample in results:
            is_valid, reason = validate_sample(sample, action_type)

            if is_valid:
                # Ensure correct ID
                sample_id = f"foundation_{next_id:05d}"
                sample["id"] = sample_id

                # Ensure metadata
                sample["category"] = "foundation"
                sample["action_type"] = action_type
                sample["source"] = "v3_foundation"

                # Ensure tools are included (strip domain/source metadata)
                if "tools" not in sample or not sample["tools"]:
                    sample["tools"] = [
                        {"name": t["name"], "description": t["description"],
                         "parameters": t["parameters"]}
                        for t in tools
                    ]

                with open(output_file, "a") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")

                existing_ids.add(sample_id)
                action_counts[action_type] += 1
                next_id += 1
                valid_in_batch += 1
                total_valid += 1
            else:
                sample["_rejection_reason"] = reason
                sample["_expected_action_type"] = action_type
                with open(malformed_file, "a") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_malformed += 1

        total_batches += 1

        if valid_in_batch > 0:
            consecutive_empty = 0
        else:
            consecutive_empty += 1
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                print(
                    f"    ERROR: {MAX_CONSECUTIVE_EMPTY} consecutive empty batches. Stopping.",
                    flush=True,
                )
                break

        if valid_in_batch < batch_size:
            print(f"    Got {valid_in_batch}/{batch_size} valid samples", flush=True)

        # Progress update every 10 batches
        if total_batches % 10 == 0:
            print_quota_status(action_counts, quotas)

        # Brief pause between batches
        time.sleep(1)

    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"  Total batches:   {total_batches}")
    print(f"  Total valid:     {total_valid}")
    print(f"  Total malformed: {total_malformed}")
    print(f"  Elapsed:         {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
    print(f"  Rate:            {total_valid/elapsed*3600:.0f} samples/hour" if elapsed > 0 else "")
    print(f"  Output:          {output_file}")
    print(f"  Malformed:       {malformed_file}")
    print_quota_status(action_counts, quotas)


def main():
    parser = argparse.ArgumentParser(
        description="Generate V3 foundation training data (18K samples, action-type enforced)"
    )
    parser.add_argument(
        "--shard", type=str, default=None,
        help="Shard spec: '1/3', '2/3', '3/3'",
    )
    parser.add_argument(
        "--action-type", type=str, default=None,
        choices=list(ACTION_TYPE_QUOTAS.keys()),
        help="Generate for single action type only",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max total samples to generate (scales quotas proportionally)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print first prompt and exit without calling Gemini",
    )
    args = parser.parse_args()

    generate(args)


if __name__ == "__main__":
    main()
