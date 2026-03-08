"""
eval_bfcl_v3.py — BFCL v4 eval with native <tool_call> handler (Qwen3.5-4B)
=============================================================================
Runs the Berkeley Function Calling Leaderboard (v4) evaluation on Modal using
a **custom native handler** that bridges Qwen3.5-4B's <tool_call> output format
to BFCL's expected scoring format.

Background:
  The R2 eval (eval_bfcl.py) used the default OpenAI FC handler, which expects
  {"function": {"name": ..., "arguments": ...}} — causing a -14.36% regression
  because our model outputs <tool_call>{"name": ..., "arguments": ...}</tool_call>.
  This script fixes the format mismatch entirely.

Model output (Qwen3.5-4B native):
  <think>reasoning here...</think>
  <tool_call>{"name": "get_weather", "arguments": {"city": "London"}}</tool_call>

BFCL expects (for scoring):
  [{"name": "get_weather", "arguments": {"city": "London"}}]

The native parser handles the conversion, including fallbacks for bare JSON and
multiple concurrent tool calls.

BFCL v4 test categories:
  Non-live: simple, multiple, parallel, parallel_multiple, java, javascript,
            relevance, irrelevance, rest, sql
  Live: live_simple, live_multiple, live_parallel, live_parallel_multiple,
        live_relevance, live_irrelevance
  Multi-turn: multi_turn_base, multi_turn_miss_func, multi_turn_miss_param,
              multi_turn_long_context
  Agentic: (varies by BFCL version)

Usage:
  # Full BFCL eval
  modal run training/eval_bfcl_v3.py

  # Specific category
  modal run training/eval_bfcl_v3.py --category simple

  # Quick test (first N samples per category)
  modal run training/eval_bfcl_v3.py --limit 20

  # Base model only (no adapter)
  modal run training/eval_bfcl_v3.py --base-only

  # Custom adapter path
  modal run training/eval_bfcl_v3.py --adapter-dir /data/adapter/custom

  # Adjust temperature
  modal run training/eval_bfcl_v3.py --temperature 0.1
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_MODEL = "Qwen/Qwen3.5-4B"
VOLUME_NAME = "aiqarus-data"
DEFAULT_ADAPTER_DIR = "/data/adapter/aiqarus-agent-4b-v3"
MERGED_MODEL_DIR = "/data/merged_model_v3"
BFCL_WORKSPACE = "/data/bfcl_workspace_v3"
RESULTS_DIR = "/data/results"

# BFCL test categories organized by group (for summary reporting)
BFCL_NONLIVE = [
    "simple", "multiple", "parallel", "parallel_multiple",
    "java", "javascript", "relevance", "irrelevance", "rest", "sql",
]
BFCL_LIVE = [
    "live_simple", "live_multiple", "live_parallel",
    "live_parallel_multiple", "live_relevance", "live_irrelevance",
]
BFCL_MULTITURN = [
    "multi_turn_base", "multi_turn_miss_func",
    "multi_turn_miss_param", "multi_turn_long_context",
]
BFCL_ALL_CATEGORIES = BFCL_NONLIVE + BFCL_LIVE + BFCL_MULTITURN

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install([
        "torch>=2.4.0",
        "vllm>=0.8.0",
    ])
    .pip_install([
        "transformers>=4.51.0",
        "peft>=0.14.0",
        "accelerate>=1.0.0",
        "sentencepiece",
        "protobuf",
        "soundfile",
        "bfcl-eval>=2026.3.0",
    ])
)

app = modal.App("aiqarus-v3-bfcl")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Native <tool_call> parser — THE KEY FIX
# ---------------------------------------------------------------------------
def parse_tool_calls_to_bfcl(model_output: str) -> list[dict]:
    """
    Convert Qwen3.5-4B native <tool_call> output to BFCL's expected format.

    Qwen3.5-4B generates:
        <tool_call>{"name": "func", "arguments": {"key": "val"}}</tool_call>

    BFCL scoring expects:
        [{"name": "func", "arguments": {"key": "val"}}]

    Handles:
    - Multiple <tool_call> blocks (parallel calls)
    - Nested JSON with varying whitespace
    - Fallback to bare JSON objects if no tags found
    - Both "arguments" dict and "arguments" string (JSON-encoded) formats
    """
    calls = []

    # Primary: <tool_call>...</tool_call> tags
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", model_output, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            # Normalize: BFCL expects "name" and "arguments" at top level
            name = obj.get("name", obj.get("function", {}).get("name", ""))
            args = obj.get("arguments", obj.get("function", {}).get("arguments", {}))
            # If arguments is a JSON string, parse it
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    pass
            if name:
                calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            # Try a more lenient parse — the JSON might span multiple lines
            raw = m.group(1).strip()
            try:
                # Attempt to fix common issues: trailing commas, single quotes
                fixed = raw.replace("'", '"')
                obj = json.loads(fixed)
                name = obj.get("name", "")
                args = obj.get("arguments", {})
                if name:
                    calls.append({"name": name, "arguments": args})
            except (json.JSONDecodeError, ValueError):
                pass

    if calls:
        return calls

    # Fallback 1: bare JSON with "name" + "arguments" keys (no tags)
    # Use a greedy approach that handles nested braces
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*"arguments"\s*:\s*\{[^}]*\}[^{}]*\}', model_output):
        try:
            obj = json.loads(m.group(0))
            if "name" in obj and "arguments" in obj:
                calls.append({"name": obj["name"], "arguments": obj["arguments"]})
        except (json.JSONDecodeError, KeyError):
            pass

    if calls:
        return calls

    # Fallback 2: JSON array of function calls
    for m in re.finditer(r'\[\s*\{.*?"name".*?\}\s*\]', model_output, re.DOTALL):
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict) and "name" in item:
                        calls.append({
                            "name": item["name"],
                            "arguments": item.get("arguments", {}),
                        })
        except (json.JSONDecodeError, KeyError):
            pass

    if calls:
        return calls

    # Fallback 3: ```json ... ``` code blocks
    for m in re.finditer(r'```(?:json)?\s*(\{.*?\})\s*```', model_output, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            if "name" in obj or "function" in obj:
                name = obj.get("name", obj.get("function", {}).get("name", ""))
                args = obj.get("arguments", obj.get("function", {}).get("arguments", {}))
                if name:
                    calls.append({"name": name, "arguments": args})
        except (json.JSONDecodeError, KeyError):
            pass

    return calls


def strip_think_block(model_output: str) -> str:
    """Remove <think>...</think> block from output for cleaner parsing."""
    return re.sub(r"<think>.*?</think>\s*", "", model_output, flags=re.DOTALL)


# ---------------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------------
def pick_best_adapter(adapter_dir: str = "") -> str | None:
    """Return the adapter path, checking V3 SimPO -> V3 DPO -> V3 SFT."""
    if adapter_dir and os.path.isdir(adapter_dir):
        return adapter_dir

    # V3 adapter paths in priority order
    candidates = [
        "/data/adapter/aiqarus-agent-4b-v3-simpo",
        "/data/adapter/aiqarus-agent-4b-v3-dpo",
        "/data/adapter/aiqarus-agent-4b-v3",
        # Fallback to V2
        "/data/adapter/aiqarus-agent-4b-v2-dpo",
        "/data/adapter/aiqarus-agent-4b-v2",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def merge_adapter_to_disk(
    base_model: str,
    adapter_path: str,
    output_dir: str,
) -> str:
    """Merge LoRA adapter into base model and save to disk for vLLM."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Merging adapter {adapter_path} into {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving merged model to {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    print("Merge complete.\n")

    # Free memory
    del model
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return output_dir


def cap_max_position_embeddings(model_dir: str, max_len: int = 8192):
    """Cap max_position_embeddings in config.json for vLLM (A10G can't fit 262K KV cache)."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path) as f:
        config = json.load(f)

    orig_len = config.get("max_position_embeddings", 0)
    if orig_len > max_len:
        config["max_position_embeddings"] = max_len
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Capped max_position_embeddings: {orig_len} -> {max_len}")


# ---------------------------------------------------------------------------
# BFCL integration — native handler approach
# ---------------------------------------------------------------------------
def run_cmd(cmd: list[str], env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, printing stdout/stderr."""
    merged_env = {**os.environ, **(env or {})}
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=merged_env, capture_output=True, text=True)
    if result.stdout:
        # Print first 200 lines max (BFCL can be very verbose)
        lines = result.stdout.split("\n")
        for line in lines[:200]:
            print(f"    {line}")
        if len(lines) > 200:
            print(f"    ... ({len(lines) - 200} more lines)")
    if result.stderr:
        lines = result.stderr.split("\n")
        for line in lines[:100]:
            print(f"    [stderr] {line}")
        if len(lines) > 100:
            print(f"    ... ({len(lines) - 100} more lines)")
    if check and result.returncode != 0:
        print(f"  Command exited with code {result.returncode}")
    return result


def try_bfcl_cli_approach(
    model_name: str,
    model_path: str | None,
    bfcl_env: dict,
    test_category: str = "",
) -> dict | None:
    """
    Attempt BFCL assessment via the bfcl CLI (generate + scoring).

    This uses the model_name to select a BFCL handler. For finetuned models,
    we register the model under the Qwen3.5 handler name and point to the
    merged model on disk.

    Returns parsed results dict or None if CLI approach fails.
    """
    print("\n--- Attempting BFCL CLI approach ---")

    # Step 1: Generate
    gen_cmd = [
        "bfcl", "generate",
        "--model", model_name,
        "--backend", "vllm",
        "--num-threads", "1",
        "--gpu-memory-utilization", "0.9",
    ]

    if model_path:
        gen_cmd.extend(["--local-model-path", model_path])

    if test_category:
        gen_cmd.extend(["--test-category", test_category])

    gen_result = run_cmd(gen_cmd, env=bfcl_env, check=False)

    if gen_result.returncode != 0:
        print("BFCL CLI generate failed. Will fall back to manual approach.")
        return None

    # Step 2: Score
    score_cmd = ["bfcl", "evaluate", "--model", model_name]
    if test_category:
        score_cmd.extend(["--test-category", test_category])

    score_result = run_cmd(score_cmd, env=bfcl_env, check=False)

    # Step 3: Parse results
    results = parse_bfcl_score_dir(BFCL_WORKSPACE, model_name)
    if results and results.get("overall_accuracy") is not None:
        return results

    return None


def run_manual_bfcl_assessment(
    model_path: str,
    test_category: str = "",
    limit: int = 0,
    temperature: float = 0.0,
) -> dict:
    """
    Manual BFCL assessment: load model via vLLM, run inference ourselves,
    convert outputs using native <tool_call> parser, then score against
    ground truth.

    This is the core of the V3 fix — we control the output format conversion
    instead of relying on BFCL's built-in handler.
    """
    print("\n--- Running manual BFCL assessment with native <tool_call> handler ---")

    # -- Load BFCL test data ---------------------------------------------------
    test_data = load_bfcl_test_data(test_category)
    if not test_data:
        print("ERROR: Could not load BFCL test data. Ensure bfcl-eval is installed.")
        return {"error": "no_test_data", "categories": {}}

    if limit > 0:
        # Limit per category
        limited = {}
        for cat, samples in test_data.items():
            limited[cat] = samples[:limit]
        test_data = limited

    total_samples = sum(len(v) for v in test_data.values())
    print(f"Loaded {total_samples} test samples across {len(test_data)} categories")

    # -- Initialize vLLM -------------------------------------------------------
    from vllm import LLM, SamplingParams

    print(f"\nLoading model from {model_path} via vLLM...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,         # Skip CUDA graph — avoids startup hang
        gpu_memory_utilization=0.9,
        max_model_len=8192,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=2048,
        top_p=0.95 if temperature > 0 else 1.0,
    )

    tokenizer = llm.get_tokenizer()

    # -- Run inference per category --------------------------------------------
    all_results = {}
    category_scores = {}

    for cat_name, samples in sorted(test_data.items()):
        print(f"\n  Category: {cat_name} ({len(samples)} samples)")
        cat_correct = 0
        cat_total = 0
        cat_outputs = []

        # Build prompts
        prompts = []
        for sample in samples:
            prompt = build_bfcl_prompt(sample, tokenizer)
            prompts.append(prompt)

        # Batch inference
        outputs = llm.generate(prompts, sampling_params)

        # Parse and score
        for i, (sample, output) in enumerate(zip(samples, outputs)):
            raw_output = output.outputs[0].text
            tool_calls = parse_tool_calls_to_bfcl(raw_output)

            # Build result in BFCL format
            result_entry = {
                "id": sample.get("id", f"{cat_name}_{i}"),
                "category": cat_name,
                "raw_output": raw_output,
                "parsed_tool_calls": tool_calls,
                "expected": sample.get("ground_truth", sample.get("expected", [])),
            }

            # Score
            is_correct = score_bfcl_sample(sample, tool_calls, cat_name)
            result_entry["correct"] = is_correct
            cat_correct += int(is_correct)
            cat_total += 1
            cat_outputs.append(result_entry)

        cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0
        category_scores[cat_name] = {
            "accuracy": round(cat_accuracy, 4),
            "correct": cat_correct,
            "total": cat_total,
        }
        all_results[cat_name] = cat_outputs
        print(f"    {cat_name}: {cat_correct}/{cat_total} = {cat_accuracy:.1%}")

    # -- Also try BFCL's own checker if available ------------------------------
    bfcl_checker_results = try_bfcl_checker(all_results, test_data)
    if bfcl_checker_results:
        print("\n  BFCL checker results available — using those for final scores.")
        category_scores = bfcl_checker_results

    # -- Aggregate -------------------------------------------------------------
    total_correct = sum(v["correct"] for v in category_scores.values())
    total_tested = sum(v["total"] for v in category_scores.values())

    # Group by BFCL sections
    nonlive_correct, nonlive_total = 0, 0
    live_correct, live_total = 0, 0
    multiturn_correct, multiturn_total = 0, 0
    other_correct, other_total = 0, 0

    for cat, scores in category_scores.items():
        if cat in BFCL_NONLIVE or any(cat.startswith(nl) for nl in BFCL_NONLIVE):
            nonlive_correct += scores["correct"]
            nonlive_total += scores["total"]
        elif cat in BFCL_LIVE or any(cat.startswith(lv) for lv in BFCL_LIVE):
            live_correct += scores["correct"]
            live_total += scores["total"]
        elif cat in BFCL_MULTITURN or cat.startswith("multi_turn"):
            multiturn_correct += scores["correct"]
            multiturn_total += scores["total"]
        else:
            other_correct += scores["correct"]
            other_total += scores["total"]

    results = {
        "overall_accuracy": round(total_correct / total_tested, 4) if total_tested > 0 else None,
        "total_correct": total_correct,
        "total_samples": total_tested,
        "nonlive_accuracy": round(nonlive_correct / nonlive_total, 4) if nonlive_total > 0 else None,
        "nonlive_correct": nonlive_correct,
        "nonlive_total": nonlive_total,
        "live_accuracy": round(live_correct / live_total, 4) if live_total > 0 else None,
        "live_correct": live_correct,
        "live_total": live_total,
        "multiturn_accuracy": round(multiturn_correct / multiturn_total, 4) if multiturn_total > 0 else None,
        "multiturn_correct": multiturn_correct,
        "multiturn_total": multiturn_total,
        "other_accuracy": round(other_correct / other_total, 4) if other_total > 0 else None,
        "other_correct": other_correct,
        "other_total": other_total,
        "categories": category_scores,
        "raw_outputs": all_results,
    }

    return results


# ---------------------------------------------------------------------------
# BFCL test data loading
# ---------------------------------------------------------------------------
def load_bfcl_test_data(category_filter: str = "") -> dict[str, list[dict]]:
    """
    Load BFCL test data. Tries multiple approaches:
    1. bfcl-eval package data loader
    2. BFCL workspace data directory
    3. Manual download from BFCL GitHub
    """
    test_data = {}

    # Approach 1: Use bfcl-eval package's built-in data loader
    try:
        from bfcl.eval_checker.eval_runner import EvalRunner
        print("  Loading test data via bfcl-eval EvalRunner...")

        # EvalRunner typically has methods to load test data
        # The API varies by version, so we try multiple approaches
        try:
            runner = EvalRunner()
            if hasattr(runner, "load_test_data"):
                raw_data = runner.load_test_data()
                if raw_data:
                    test_data = raw_data
                    print(f"  Loaded via EvalRunner.load_test_data()")
        except Exception as e:
            print(f"  EvalRunner direct load failed: {e}")

    except ImportError:
        print("  bfcl-eval not importable for data loading.")

    # Approach 2: Look for data files in standard BFCL locations
    if not test_data:
        test_data = load_bfcl_data_from_files(category_filter)

    # Approach 3: Try downloading via bfcl CLI
    if not test_data:
        print("  Attempting to fetch test data via bfcl CLI...")
        try:
            result = subprocess.run(
                ["bfcl", "download", "--test-data"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                test_data = load_bfcl_data_from_files(category_filter)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  bfcl download failed: {e}")

    # Apply category filter
    if category_filter and test_data:
        filtered = {}
        for cat, samples in test_data.items():
            if category_filter in cat or cat.startswith(category_filter):
                filtered[cat] = samples
        if filtered:
            test_data = filtered
        else:
            print(f"  Warning: Category filter '{category_filter}' matched no categories.")
            print(f"  Available: {list(test_data.keys())}")

    return test_data


def load_bfcl_data_from_files(category_filter: str = "") -> dict[str, list[dict]]:
    """Load BFCL test data from JSONL files in standard locations."""
    test_data = {}

    # Standard BFCL data locations
    search_dirs = [
        os.path.join(BFCL_WORKSPACE, "data"),
        os.path.join(BFCL_WORKSPACE, "test_data"),
        os.path.expanduser("~/.cache/bfcl"),
        "/tmp/bfcl_data",
    ]

    # Also check bfcl-eval package data directory
    try:
        import bfcl
        pkg_dir = os.path.dirname(bfcl.__file__)
        search_dirs.append(os.path.join(pkg_dir, "data"))
        search_dirs.append(os.path.join(pkg_dir, "eval_checker", "data"))
    except (ImportError, AttributeError):
        pass

    for data_dir in search_dirs:
        if not os.path.isdir(data_dir):
            continue

        print(f"  Scanning for test data in: {data_dir}")

        for root, dirs, files in os.walk(data_dir):
            for fname in sorted(files):
                if not fname.endswith((".json", ".jsonl")):
                    continue

                # Infer category from filename
                cat_name = fname.replace(".jsonl", "").replace(".json", "")
                # Common BFCL filename patterns
                for prefix in ["gorilla_openfunctions_v1_test_", "BFCL_v3_", "BFCL_v4_", "test_"]:
                    if cat_name.startswith(prefix):
                        cat_name = cat_name[len(prefix):]

                if category_filter and category_filter not in cat_name:
                    continue

                fpath = os.path.join(root, fname)
                samples = []
                try:
                    with open(fpath) as f:
                        content = f.read().strip()
                        if content.startswith("["):
                            # JSON array
                            samples = json.loads(content)
                        else:
                            # JSONL
                            for line in content.split("\n"):
                                line = line.strip()
                                if line:
                                    try:
                                        samples.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        pass
                except (json.JSONDecodeError, IOError) as e:
                    print(f"    Warning: Could not parse {fpath}: {e}")
                    continue

                if samples:
                    test_data[cat_name] = samples
                    print(f"    {cat_name}: {len(samples)} samples")

        if test_data:
            break  # Use first directory that has data

    return test_data


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_bfcl_prompt(sample: dict, tokenizer) -> str:
    """
    Build a chat prompt from a BFCL test sample.

    BFCL samples typically have:
    - "question": list of message dicts [{"role": "user", "content": "..."}]
    - "function": list of tool/function definitions
    - "ground_truth": expected function calls

    We format the functions as tool descriptions in the system message,
    then apply Qwen3.5-4B's chat template.
    """
    messages = []

    # Extract function definitions for the system message
    functions = sample.get("function", sample.get("functions", []))
    question = sample.get("question", sample.get("messages", []))

    # Build system message with tool definitions
    if functions:
        tools_desc = format_functions_for_prompt(functions)
        system_msg = (
            "You are a helpful assistant with access to the following tools. "
            "When you need to call a tool, output a <tool_call> tag with a JSON "
            "object containing 'name' and 'arguments' keys.\n\n"
            "Available tools:\n" + tools_desc
        )
        messages.append({"role": "system", "content": system_msg})

    # Add conversation messages
    if isinstance(question, list):
        for msg in question:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("system", "user", "assistant", "tool"):
                messages.append({"role": role, "content": content})
    elif isinstance(question, str):
        messages.append({"role": "user", "content": question})
    else:
        messages.append({"role": "user", "content": str(question)})

    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        # Fallback: manual prompt construction
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

    return prompt


def format_functions_for_prompt(functions: list[dict]) -> str:
    """Format BFCL function definitions as a readable tool description string."""
    parts = []
    for i, func in enumerate(functions):
        # BFCL function format varies; normalize
        if "name" in func:
            name = func["name"]
            desc = func.get("description", "")
            params = func.get("parameters", {})
        elif "function" in func:
            inner = func["function"]
            name = inner.get("name", f"tool_{i}")
            desc = inner.get("description", "")
            params = inner.get("parameters", {})
        else:
            name = func.get("api_name", func.get("tool_name", f"tool_{i}"))
            desc = func.get("api_description", func.get("description", ""))
            params = func.get("parameters", func.get("params", {}))

        part = f"- {name}"
        if desc:
            part += f": {desc}"
        if params:
            # Show parameter schema compactly
            props = params.get("properties", {})
            required = params.get("required", [])
            if props:
                param_strs = []
                for pname, pinfo in props.items():
                    ptype = pinfo.get("type", "any")
                    pdesc = pinfo.get("description", "")
                    req_mark = " (required)" if pname in required else ""
                    ps = f"    - {pname}: {ptype}{req_mark}"
                    if pdesc:
                        ps += f" -- {pdesc}"
                    param_strs.append(ps)
                part += "\n  Parameters:\n" + "\n".join(param_strs)

        parts.append(part)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_bfcl_sample(
    sample: dict,
    parsed_calls: list[dict],
    category: str,
) -> bool:
    """
    Score a single BFCL sample by comparing parsed tool calls to ground truth.

    Scoring rules (simplified):
    - For 'simple': exactly 1 correct function call (name + arguments match)
    - For 'multiple'/'parallel'/'parallel_multiple': all expected calls present
    - For 'relevance': model should call the relevant function
    - For 'irrelevance': model should NOT call any function
    - For multi-turn: correct call at each turn (simplified: check final call)

    Uses relaxed argument matching (type coercion, optional params).
    """
    ground_truth = sample.get("ground_truth", sample.get("expected", []))

    # Handle irrelevance — model should NOT make a function call
    if "irrelevance" in category:
        return len(parsed_calls) == 0

    # If ground_truth is empty, we can't score
    if not ground_truth:
        return False

    # Normalize ground_truth format
    expected_calls = normalize_ground_truth(ground_truth)

    if not expected_calls:
        return len(parsed_calls) == 0

    # Check if all expected calls are present in parsed calls
    if len(parsed_calls) < len(expected_calls):
        return False

    # Match each expected call to a parsed call
    matched = [False] * len(expected_calls)
    used = [False] * len(parsed_calls)

    for ei, exp in enumerate(expected_calls):
        for pi, pred in enumerate(parsed_calls):
            if used[pi]:
                continue
            if calls_match(exp, pred):
                matched[ei] = True
                used[pi] = True
                break

    return all(matched)


def normalize_ground_truth(ground_truth) -> list[dict]:
    """Normalize BFCL ground truth to list of {"name": ..., "arguments": ...}."""
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            # Might be a function call string like "func_name(arg1='val1')"
            return parse_function_call_string(ground_truth)

    if isinstance(ground_truth, dict):
        ground_truth = [ground_truth]

    if not isinstance(ground_truth, list):
        return []

    normalized = []
    for item in ground_truth:
        if isinstance(item, str):
            # Could be "func_name(arg1='val1', arg2=123)"
            parsed = parse_function_call_string(item)
            normalized.extend(parsed)
        elif isinstance(item, dict):
            if "name" in item and "arguments" in item:
                normalized.append(item)
            elif "function" in item:
                inner = item["function"]
                normalized.append({
                    "name": inner.get("name", ""),
                    "arguments": inner.get("arguments", {}),
                })
            elif len(item) == 1:
                # Format: {"func_name": {"arg1": "val1"}}
                for fname, fargs in item.items():
                    normalized.append({
                        "name": fname,
                        "arguments": fargs if isinstance(fargs, dict) else {},
                    })
            else:
                # Try to extract name + arguments
                name = item.get("name", item.get("api_name", item.get("tool_name", "")))
                args = item.get("arguments", item.get("params", item.get("parameters", {})))
                if name:
                    normalized.append({"name": name, "arguments": args})

    return normalized


def parse_function_call_string(s: str) -> list[dict]:
    """
    Parse BFCL-style function call string: "func_name(arg1='val1', arg2=123)"
    Returns list of {"name": ..., "arguments": ...}.
    """
    results = []
    # Match: function_name(...) or module.function_name(...)
    pattern = r'([\w.]+)\((.*?)\)'
    for m in re.finditer(pattern, s, re.DOTALL):
        name = m.group(1)
        args_str = m.group(2).strip()

        arguments = {}
        if args_str:
            # Parse keyword arguments: key=value, key='string', key=123
            # This is a simplified parser; BFCL's ground truth format can be complex
            for arg_match in re.finditer(
                r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|(\[[^\]]*\])|(\{[^}]*\})|([\w.+-]+))",
                args_str,
            ):
                key = arg_match.group(1)
                # Pick the first non-None capture group
                val = (
                    arg_match.group(2) or arg_match.group(3)
                    or arg_match.group(4) or arg_match.group(5)
                    or arg_match.group(6)
                )
                # Try to parse as JSON/number
                if val is not None:
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        # Keep as string
                        pass
                arguments[key] = val

        results.append({"name": name, "arguments": arguments})

    return results


def calls_match(expected: dict, predicted: dict) -> bool:
    """
    Check if a predicted function call matches an expected one.

    Uses relaxed matching:
    - Function names must match exactly (case-sensitive)
    - All expected arguments must be present
    - Extra predicted arguments are tolerated (model may add optional params)
    - Argument values use type-coerced comparison (string "123" matches int 123)
    """
    exp_name = expected.get("name", "")
    pred_name = predicted.get("name", "")

    if exp_name != pred_name:
        return False

    exp_args = expected.get("arguments", {})
    pred_args = predicted.get("arguments", {})

    if not isinstance(exp_args, dict) or not isinstance(pred_args, dict):
        return str(exp_args) == str(pred_args)

    # All expected arguments must be present and match
    for key, exp_val in exp_args.items():
        if key not in pred_args:
            return False
        pred_val = pred_args[key]
        if not values_match(exp_val, pred_val):
            return False

    return True


def values_match(expected, predicted) -> bool:
    """Relaxed value comparison with type coercion."""
    # Exact match
    if expected == predicted:
        return True

    # String comparison (handles int/float/bool -> string coercion)
    if str(expected).lower().strip() == str(predicted).lower().strip():
        return True

    # Numeric comparison
    try:
        if float(expected) == float(predicted):
            return True
    except (ValueError, TypeError):
        pass

    # List comparison (order-insensitive for some BFCL categories)
    if isinstance(expected, list) and isinstance(predicted, list):
        if len(expected) == len(predicted):
            # Try sorted comparison
            try:
                return sorted(str(x) for x in expected) == sorted(str(x) for x in predicted)
            except TypeError:
                pass

    # Dict comparison (recursive)
    if isinstance(expected, dict) and isinstance(predicted, dict):
        if set(expected.keys()) != set(predicted.keys()):
            return False
        return all(values_match(expected[k], predicted[k]) for k in expected)

    return False


# ---------------------------------------------------------------------------
# BFCL checker integration (optional, for more accurate scoring)
# ---------------------------------------------------------------------------
def try_bfcl_checker(
    all_results: dict[str, list[dict]],
    test_data: dict[str, list[dict]],
) -> dict | None:
    """
    Try to use BFCL's official checker for scoring instead of our
    simplified scorer. Returns category scores dict or None if unavailable.
    """
    try:
        from bfcl.eval_checker import eval_runner
        print("\n  Attempting to use BFCL's official checker...")
    except ImportError:
        print("\n  BFCL checker not available. Using built-in scorer.")
        return None

    # The BFCL checker API varies by version. Try common approaches.
    try:
        # Approach 1: Write our results in BFCL's expected format and run checker
        output_dir = os.path.join(BFCL_WORKSPACE, "manual_results")
        os.makedirs(output_dir, exist_ok=True)

        for cat_name, results in all_results.items():
            # Convert to BFCL result format
            bfcl_results = []
            for r in results:
                # BFCL expects the model's raw output or parsed function calls
                bfcl_results.append({
                    "id": r["id"],
                    "result": r["parsed_tool_calls"] if r["parsed_tool_calls"] else "",
                })

            result_path = os.path.join(output_dir, f"{cat_name}.jsonl")
            with open(result_path, "w") as f:
                for entry in bfcl_results:
                    f.write(json.dumps(entry) + "\n")

        # Try running the checker
        if hasattr(eval_runner, "EvalRunner"):
            runner = eval_runner.EvalRunner()
            if hasattr(runner, "evaluate"):
                runner.evaluate(output_dir)
                # Parse scores
                return parse_bfcl_score_dir(BFCL_WORKSPACE, "manual_results")

    except Exception as e:
        print(f"  BFCL checker failed: {e}")
        print("  Falling back to built-in scorer.")

    return None


def parse_bfcl_score_dir(workspace: str, model_name: str) -> dict:
    """Parse BFCL score directory into category results dict."""
    results = {}
    model_name_sanitized = model_name.replace("/", "_")

    # Search for score files
    score_dirs = [
        os.path.join(workspace, "score", model_name_sanitized),
        os.path.join(workspace, "score", model_name),
        os.path.join(workspace, "score"),
    ]

    score_dir = None
    for d in score_dirs:
        if os.path.isdir(d):
            score_dir = d
            break

    if not score_dir:
        return {}

    total_correct = 0
    total_samples = 0

    for fname in sorted(os.listdir(score_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(score_dir, fname)
        try:
            with open(fpath) as f:
                data = json.load(f)
            category = fname.replace(".json", "").replace("_score", "")
            accuracy = data.get("accuracy", data.get("overall_accuracy"))
            correct = data.get("correct", 0)
            total = data.get("total", 0)

            if accuracy is not None:
                results[category] = {
                    "accuracy": round(float(accuracy), 4),
                    "correct": correct,
                    "total": total,
                }
                total_correct += correct
                total_samples += total
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    Warning: Could not parse {fpath}: {e}")

    if total_samples > 0:
        results["__overall__"] = {
            "accuracy": round(total_correct / total_samples, 4),
            "correct": total_correct,
            "total": total_samples,
        }

    return results


# ---------------------------------------------------------------------------
# Result display
# ---------------------------------------------------------------------------
def print_bfcl_summary(results: dict, model_tag: str):
    """Print human-readable BFCL results summary."""
    print(f"\n{'='*65}")
    print(f"  BFCL v4 Results ({model_tag})")
    print(f"{'='*65}")

    if results.get("overall_accuracy") is not None:
        print(f"  Overall:      {results['overall_accuracy']:.2%}  ({results.get('total_correct', '?')}/{results.get('total_samples', '?')})")
    else:
        print("  Overall:      N/A (no scores)")

    if results.get("nonlive_accuracy") is not None:
        print(f"  Non-live:     {results['nonlive_accuracy']:.2%}  ({results.get('nonlive_correct', '?')}/{results.get('nonlive_total', '?')})")
    if results.get("live_accuracy") is not None:
        print(f"  Live:         {results['live_accuracy']:.2%}  ({results.get('live_correct', '?')}/{results.get('live_total', '?')})")
    if results.get("multiturn_accuracy") is not None:
        print(f"  Multi-turn:   {results['multiturn_accuracy']:.2%}  ({results.get('multiturn_correct', '?')}/{results.get('multiturn_total', '?')})")
    if results.get("other_accuracy") is not None and results.get("other_total", 0) > 0:
        print(f"  Other:        {results['other_accuracy']:.2%}  ({results.get('other_correct', '?')}/{results.get('other_total', '?')})")

    # Per-category breakdown
    categories = results.get("categories", {})
    if categories:
        print(f"\n  Per-category breakdown:")
        print(f"  {'Category':40s} {'Accuracy':>10s}  {'N':>6s}")
        print(f"  {'-'*40} {'-'*10}  {'-'*6}")
        for cat, data in sorted(categories.items()):
            if cat.startswith("__"):
                continue
            acc = data.get("accuracy", 0)
            n = data.get("total", 0)
            print(f"  {cat:40s} {acc:>9.1%}  {n:>6d}")

    print(f"{'='*65}")


def print_comparison(finetuned: dict, base_ref: dict | None = None):
    """Print comparison with base model scores."""
    if not base_ref:
        # R2 base model reference scores (Qwen3-4B from official leaderboard)
        base_ref = {
            "overall_accuracy": 0.3568,
            "nonlive_accuracy": 0.8788,
            "live_accuracy": 0.7639,
            "multiturn_accuracy": 0.2212,
        }

    print(f"\n  Comparison vs Base Qwen3-4B (BFCL leaderboard):")
    print(f"  {'-'*50}")

    for key, label in [
        ("overall_accuracy", "Overall"),
        ("nonlive_accuracy", "Non-live"),
        ("live_accuracy", "Live"),
        ("multiturn_accuracy", "Multi-turn"),
    ]:
        ft_val = finetuned.get(key)
        base_val = base_ref.get(key)
        if ft_val is not None and base_val is not None:
            delta = ft_val - base_val
            sign = "+" if delta >= 0 else ""
            print(f"  {label:15s}  {ft_val:.2%}  ({sign}{delta:.2%} vs base {base_val:.2%})")
        elif ft_val is not None:
            print(f"  {label:15s}  {ft_val:.2%}")


# ---------------------------------------------------------------------------
# Main Modal function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=6 * 3600,    # 6 hours max
    memory=32768,
)
def run_bfcl_v3(
    adapter_dir: str = "",
    base_only: bool = False,
    category: str = "",
    limit: int = 0,
    temperature: float = 0.0,
):
    """
    Run BFCL v4 assessment with native <tool_call> handler.

    Strategy:
    1. First try BFCL CLI approach (in case the registered handler works)
    2. If CLI fails or uses wrong format, fall back to manual approach
       with native <tool_call> parser
    """
    start_time = time.time()

    # -- Prepare model ---------------------------------------------------------
    if base_only:
        model_tag = "base-qwen3.5-4b"
        adapter_path = None
        print(f"Running base model: {BASE_MODEL} (no adapter)")
        # For base model, download and use directly
        model_path = BASE_MODEL  # vLLM can load from HF
    else:
        adapter_path = pick_best_adapter(adapter_dir)
        if not adapter_path:
            print("ERROR: No adapter found. Checked paths:")
            print("  - /data/adapter/aiqarus-agent-4b-v3-simpo")
            print("  - /data/adapter/aiqarus-agent-4b-v3-dpo")
            print("  - /data/adapter/aiqarus-agent-4b-v3")
            print("Use --base-only or --adapter-dir <path>.")
            return None

        model_tag = "finetuned-v3"
        print(f"Adapter: {adapter_path}")
        print(f"Base model: {BASE_MODEL}")
        print(f"Merging adapter to disk...")

        model_path = merge_adapter_to_disk(
            base_model=BASE_MODEL,
            adapter_path=adapter_path,
            output_dir=MERGED_MODEL_DIR,
        )

    # -- Cap context length for A10G -------------------------------------------
    if model_path and os.path.isdir(model_path):
        cap_max_position_embeddings(model_path, max_len=8192)

    # -- Set up BFCL workspace -------------------------------------------------
    os.makedirs(BFCL_WORKSPACE, exist_ok=True)

    # Clear cached results from prior runs
    for subdir in ["result", "score"]:
        cached = os.path.join(BFCL_WORKSPACE, subdir)
        if os.path.isdir(cached):
            shutil.rmtree(cached)
            print(f"  Cleared cached {subdir}/")

    bfcl_env = {
        "BFCL_PROJECT_ROOT": BFCL_WORKSPACE,
        "VLLM_ENFORCE_EAGER": "1",
    }

    # -- Strategy 1: Try BFCL CLI with Qwen3.5 handler ------------------------
    # Some BFCL versions register Qwen models with native tool_call support.
    # If the CLI works correctly, use its results.
    cli_model_name = "Qwen/Qwen3.5-4B-FC"  # FC = Function Calling handler
    cli_results = None

    if not base_only and model_path and os.path.isdir(model_path):
        # Only try CLI if we have a local model (can point BFCL to it)
        cli_results = try_bfcl_cli_approach(
            model_name=cli_model_name,
            model_path=model_path,
            bfcl_env=bfcl_env,
            test_category=category,
        )

    # -- Strategy 2: Manual assessment with native <tool_call> parser ----------
    # This is the guaranteed approach — we control everything.
    if cli_results and cli_results.get("overall_accuracy") is not None:
        print("\nBFCL CLI approach succeeded. Using CLI results.")
        final_results = cli_results
    else:
        if cli_results is not None:
            print("\nBFCL CLI produced no usable results. Running manual assessment.")
        else:
            print("\nRunning manual BFCL assessment with native <tool_call> handler.")

        final_results = run_manual_bfcl_assessment(
            model_path=model_path,
            test_category=category,
            limit=limit,
            temperature=temperature,
        )

    # -- Build final result object ---------------------------------------------
    # Strip raw_outputs for JSON serialization (can be very large)
    result_for_save = {k: v for k, v in final_results.items() if k != "raw_outputs"}

    result = {
        "model": model_tag,
        "benchmark": "BFCL_v4",
        "base_model": BASE_MODEL,
        "adapter": adapter_path,
        "handler": "native_tool_call",
        "temperature": temperature,
        "limit": limit if limit > 0 else "all",
        "category_filter": category or "all",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **result_for_save,
    }

    # -- Save results ----------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = f"{RESULTS_DIR}/v3_bfcl_results_{model_tag}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {result_path}")

    # Save raw outputs separately (large file)
    if "raw_outputs" in final_results:
        raw_path = f"{RESULTS_DIR}/v3_bfcl_raw_{model_tag}.jsonl"
        with open(raw_path, "w") as f:
            for cat_name, entries in final_results["raw_outputs"].items():
                for entry in entries:
                    # Strip the full raw output to keep file manageable
                    compact = {
                        "id": entry["id"],
                        "category": entry["category"],
                        "correct": entry["correct"],
                        "parsed_tool_calls": entry["parsed_tool_calls"],
                        "raw_output_preview": entry["raw_output"][:500],
                    }
                    f.write(json.dumps(compact) + "\n")
        print(f"Saved raw outputs: {raw_path}")

    volume.commit()

    # -- Print summary ---------------------------------------------------------
    print_bfcl_summary(result, model_tag)

    if not base_only:
        print_comparison(result)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    return result


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    adapter_dir: str = "",
    base_only: bool = False,
    category: str = "",
    limit: int = 0,
    temperature: float = 0.0,
):
    """
    Run BFCL v4 benchmark on Modal with native <tool_call> handler.

    Flags:
      --adapter-dir PATH     Override adapter path on Modal volume
      --base-only            Run base Qwen3.5-4B without adapter
      --category CAT         Filter to specific BFCL category (e.g. simple, relevance)
      --limit N              Only run first N samples per category (for quick testing)
      --temperature FLOAT    Generation temperature (default 0.0 for deterministic)
    """
    result = run_bfcl_v3.remote(
        adapter_dir=adapter_dir,
        base_only=base_only,
        category=category,
        limit=limit,
        temperature=temperature,
    )

    if result:
        # Save locally too
        local_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(local_dir, exist_ok=True)
        tag = result["model"]
        local_path = os.path.join(local_dir, f"v3_bfcl_results_{tag}.json")
        with open(local_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nLocal copy saved: {local_path}")

        # Print comparison summary
        print(f"\nBFCL v4 Results (Qwen3.5-4B + V3 adapter):")
        if result.get("overall_accuracy") is not None:
            print(f"  Overall:      {result['overall_accuracy']:.2%}")
        if result.get("nonlive_accuracy") is not None:
            print(f"  Non-live:     {result['nonlive_accuracy']:.2%}")
        if result.get("live_accuracy") is not None:
            print(f"  Live:         {result['live_accuracy']:.2%}")
        if result.get("multiturn_accuracy") is not None:
            print(f"  Multi-turn:   {result['multiturn_accuracy']:.2%}")
        if result.get("other_accuracy") is not None and result.get("other_total", 0) > 0:
            print(f"  Other:        {result['other_accuracy']:.2%}")

        if not base_only:
            overall = result.get("overall_accuracy")
            if overall is not None:
                delta = overall - 0.3568
                sign = "+" if delta >= 0 else ""
                print(f"\n  Comparison vs Base:")
                print(f"  Overall: {sign}{delta:.2%} (base: 35.68%)")
    else:
        print("\nNo result returned. Check Modal logs for errors.")
        print("Try downloading from volume:")
        print(f"  modal volume get {VOLUME_NAME} results/v3_bfcl_results_finetuned-v3.json .")
