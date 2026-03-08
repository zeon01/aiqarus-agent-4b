"""
eval_comparative.py — Base vs Fine-Tuned Comparison on Modal
=============================================================
Orchestrator that runs multiple benchmarks (custom harness, When2Call, BFCL)
across model variants (base, SFT, DPO) and produces side-by-side comparison
tables for the model card.

Usage:
  modal run --detach training/eval_comparative.py                        # full: all × base,dpo
  modal run training/eval_comparative.py --benchmark custom --limit 10   # smoke test
  modal run training/eval_comparative.py --models base,sft,dpo           # 3-way
  modal run training/eval_comparative.py --report-only                   # aggregate existing
"""

import gc
import json
import modal
import os
import sys
import time

# Add training/ to path for imports
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import (
    BASE_MODEL, VOLUME_NAME, RESULTS_DIR,
    ADAPTER_V2_SFT, ADAPTER_V2_DPO,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.4.0",
        "transformers>=5.0.0",
        "peft>=0.14.0",
        "accelerate>=1.0.0",
        "sentencepiece",
        "protobuf",
        "datasets>=3.0.0",
        "scikit-learn>=1.5.0",
        "vllm>=0.7.0",
        "bfcl-eval",
    ])
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "eval_utils.py"),
        remote_path="/root/eval_utils.py",
    )
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "eval_when2call.py"),
        remote_path="/root/eval_when2call.py",
    )
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "eval_bfcl.py"),
        remote_path="/root/eval_bfcl.py",
    )
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), "..", "data"),
        remote_path="/test_data",
    )
)

app = modal.App("aiqarus-eval-comparative")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

MODEL_VARIANTS = {
    "base": {"adapter": None, "label": "Qwen3-4B-Instruct (base)"},
    "sft": {"adapter": ADAPTER_V2_SFT, "label": "aiqarus-agent-4b-v2 (SFT)"},
    "dpo": {"adapter": ADAPTER_V2_DPO, "label": "aiqarus-agent-4b-v2 (SFT+DPO)"},
}


def run_custom_harness(model, tokenizer, limit: int = 0) -> dict | None:
    """Run the custom 230-case harness using transformers inference."""
    sys.path.insert(0, "/root")
    from eval_utils import (
        generate_response, extract_tool_calls, extract_think_block,
        classify_action, score_functional, score_adversarial,
        compute_summary, save_json, save_jsonl,
    )

    # Load test cases
    test_file = "/test_data/test_cases.jsonl"
    if not os.path.exists(test_file):
        # Try alternative path
        test_file = "/test_data/eval_test_cases.jsonl"
    if not os.path.exists(test_file):
        print("  Warning: No test cases found for custom harness.")
        print(f"  Looked in: /test_data/test_cases.jsonl, /test_data/eval_test_cases.jsonl")
        # List what's available
        if os.path.isdir("/test_data"):
            files = os.listdir("/test_data")
            jsonl_files = [f for f in files if f.endswith(".jsonl")]
            print(f"  Available .jsonl files: {jsonl_files[:10]}")
        return None

    test_cases = []
    with open(test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                test_cases.append(json.loads(line))

    if limit > 0:
        test_cases = test_cases[:limit]

    print(f"  Running custom harness on {len(test_cases)} test cases...")

    results = []
    for i, tc in enumerate(test_cases):
        tools_text = json.dumps(tc.get("tools", []), indent=2)
        system_msg = (
            "You are an AI assistant with access to the following tools:\n\n"
            f"{tools_text}\n\n"
            "Respond to the user's request appropriately."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": tc["input"]},
        ]

        output = generate_response(model, tokenizer, messages)
        tool_calls = extract_tool_calls(output)
        expected = tc.get("expected", {})

        if tc.get("type") == "adversarial":
            score = score_adversarial(expected, output, tool_calls)
        else:
            score = score_functional(expected, output, tool_calls)

        score["category"] = tc.get("category", "unknown")
        score["difficulty"] = tc.get("difficulty", "unknown")
        score["type"] = tc.get("type", "functional")
        results.append(score)

        if (i + 1) % 50 == 0:
            running_acc = sum(r["action_correct"] for r in results) / len(results)
            print(f"    [{i+1}/{len(test_cases)}] acc={running_acc:.1%}")

    summary = compute_summary(results)
    return summary


def run_when2call_benchmark(model, tokenizer, limit: int = 0) -> dict | None:
    """Run When2Call MCQ benchmark using the loaded model."""
    sys.path.insert(0, "/root")
    from eval_when2call import score_mcq_sample, compute_when2call_metrics

    from datasets import load_dataset

    print("  Loading When2Call MCQ test split...")
    ds = load_dataset("nvidia/When2Call", split="test")

    mcq_samples = []
    for sample in ds:
        if sample.get("answers") and sample.get("correct_answer"):
            mcq_samples.append(sample)

    print(f"  Loaded {len(mcq_samples)} MCQ samples.")

    if limit > 0:
        mcq_samples = mcq_samples[:limit]
        print(f"    (limited to {limit})")

    predictions = []
    ground_truths = []

    for i, sample in enumerate(mcq_samples):
        predicted = score_mcq_sample(model, tokenizer, sample)
        ground_truth = sample["correct_answer"]
        predictions.append(predicted)
        ground_truths.append(ground_truth)

        if (i + 1) % 100 == 0:
            running_acc = sum(p == g for p, g in zip(predictions, ground_truths)) / len(predictions)
            print(f"    [{i+1}/{len(mcq_samples)}] acc={running_acc:.1%}")

    metrics = compute_when2call_metrics(predictions, ground_truths)
    return metrics


def run_bfcl_benchmark(model_path: str, model_name: str) -> dict | None:
    """Run BFCL benchmark via subprocess (needs merged model on disk)."""
    sys.path.insert(0, "/root")
    from eval_bfcl import run_cmd, parse_bfcl_results, BFCL_WORKSPACE

    os.makedirs(BFCL_WORKSPACE, exist_ok=True)
    bfcl_env = {"BFCL_PROJECT_ROOT": BFCL_WORKSPACE}

    # Generate
    gen_cmd = [
        "bfcl", "generate",
        "--model", model_name,
        "--backend", "vllm",
        "--local-model-path", model_path,
        "--num-threads", "1",
    ]
    gen_result = run_cmd(gen_cmd, env=bfcl_env, check=False)

    if gen_result.returncode != 0:
        print("  BFCL generate failed with vLLM, trying huggingface backend...")
        gen_cmd_hf = [
            "bfcl", "generate",
            "--model", model_name,
            "--backend", "huggingface",
            "--local-model-path", model_path,
            "--num-threads", "1",
        ]
        gen_result = run_cmd(gen_cmd_hf, env=bfcl_env, check=False)
        if gen_result.returncode != 0:
            print("  BFCL generate failed with both backends.")
            return None

    # Assess
    assess_cmd = ["bfcl", "evaluate", "--model", model_name]
    run_cmd(assess_cmd, env=bfcl_env, check=False)

    return parse_bfcl_results(BFCL_WORKSPACE, model_name)


def free_gpu_memory():
    """Release GPU memory between model variants."""
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  GPU memory freed.")


def generate_markdown_report(all_results: dict) -> str:
    """Generate a Markdown comparison table from all results."""
    lines = [
        "# Comparative Assessment Report",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Base model:** {BASE_MODEL}",
        "",
    ]

    model_names = list(all_results.keys())

    # ── Custom Harness ────────────────────────────────────────────
    lines.append("## Custom Harness (230 test cases)")
    lines.append("")
    header = "| Metric | " + " | ".join(model_names) + " |"
    sep = "|---|" + "|".join(["---"] * len(model_names)) + "|"
    lines.append(header)
    lines.append(sep)

    for metric in ["action_accuracy", "must_not_call_compliance", "tool_name_accuracy"]:
        row = f"| {metric} |"
        for m in model_names:
            custom = all_results[m].get("custom")
            if custom and metric in custom:
                val = custom[metric]
                row += f" {val}% |" if val is not None else " N/A |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")

    # ── When2Call ─────────────────────────────────────────────────
    lines.append("## When2Call (MCQ Benchmark)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for metric in ["accuracy", "macro_f1", "tool_hallucination_rate"]:
        row = f"| {metric} |"
        for m in model_names:
            w2c = all_results[m].get("when2call")
            if w2c and metric in w2c:
                val = w2c[metric]
                if metric in ("accuracy", "tool_hallucination_rate"):
                    row += f" {val:.1%} |"
                else:
                    row += f" {val:.4f} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")

    # ── BFCL ─────────────────────────────────────────────────────
    lines.append("## BFCL (Berkeley Function Calling Leaderboard)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    row = "| overall_accuracy |"
    for m in model_names:
        bfcl = all_results[m].get("bfcl")
        if bfcl and bfcl.get("overall_accuracy") is not None:
            row += f" {bfcl['overall_accuracy']:.1%} |"
        else:
            row += " — |"
    lines.append(row)

    # Per-category BFCL
    all_cats = set()
    for m in model_names:
        bfcl = all_results[m].get("bfcl", {})
        all_cats.update(bfcl.get("categories", {}).keys())

    for cat in sorted(all_cats):
        row = f"| {cat} |"
        for m in model_names:
            bfcl = all_results[m].get("bfcl", {})
            cat_data = bfcl.get("categories", {}).get(cat)
            if cat_data:
                row += f" {cat_data['accuracy']:.1%} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("*Generated by eval_comparative.py*")

    return "\n".join(lines)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=18 * 3600,
    memory=32768,
)
def run_comparative(
    benchmark: str = "all",
    models: str = "base,dpo",
    limit: int = 0,
    report_only: bool = False,
):
    """
    Run comparative assessment across model variants and benchmarks.

    Args:
        benchmark: Which benchmark(s) to run: "custom", "when2call", "bfcl", "all"
        models: Comma-separated model variants: "base", "sft", "dpo"
        limit: Limit samples per benchmark (0 = no limit)
        report_only: Only aggregate existing results, don't run benchmarks
    """
    sys.path.insert(0, "/root")
    from eval_utils import (
        load_model, merge_adapter_to_disk, pick_best_adapter,
        save_json,
    )

    start_time = time.time()
    model_list = [m.strip() for m in models.split(",")]
    do_custom = benchmark in ("all", "custom")
    do_when2call = benchmark in ("all", "when2call")
    do_bfcl = benchmark in ("all", "bfcl")

    print(f"Comparative: models={model_list}, benchmark={benchmark}, limit={limit}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    if report_only:
        print("\n--- Report-only mode: loading existing results ---")
        for variant in model_list:
            all_results[variant] = {}

            # Load custom
            custom_path = f"{RESULTS_DIR}/custom_results_{variant}.json"
            if os.path.exists(custom_path):
                with open(custom_path) as f:
                    all_results[variant]["custom"] = json.load(f)
                print(f"  Loaded custom results for {variant}")

            # Load when2call
            w2c_path = f"{RESULTS_DIR}/when2call_results_{variant}.json"
            if os.path.exists(w2c_path):
                with open(w2c_path) as f:
                    all_results[variant]["when2call"] = json.load(f)
                print(f"  Loaded when2call results for {variant}")

            # Load BFCL
            bfcl_path = f"{RESULTS_DIR}/bfcl_results_{variant}.json"
            if os.path.exists(bfcl_path):
                with open(bfcl_path) as f:
                    all_results[variant]["bfcl"] = json.load(f)
                print(f"  Loaded BFCL results for {variant}")

    else:
        for variant in model_list:
            print(f"\n{'='*60}")
            print(f"MODEL VARIANT: {variant}")
            print(f"{'='*60}")

            variant_info = MODEL_VARIANTS.get(variant)
            if not variant_info:
                print(f"  Unknown variant: {variant}. Skipping.")
                continue

            adapter_path = variant_info["adapter"]
            all_results[variant] = {}
            is_base = (adapter_path is None)

            # Check if adapter exists (for sft/dpo)
            if adapter_path and not os.path.isdir(adapter_path):
                print(f"  Warning: Adapter not found at {adapter_path}")
                if variant == "dpo":
                    if os.path.isdir(ADAPTER_V2_DPO):
                        adapter_path = ADAPTER_V2_DPO
                    elif os.path.isdir(ADAPTER_V2_SFT):
                        print(f"  Falling back to SFT adapter for '{variant}'")
                        adapter_path = ADAPTER_V2_SFT
                    else:
                        print(f"  No adapter available. Skipping {variant}.")
                        continue
                elif variant == "sft":
                    if os.path.isdir(ADAPTER_V2_SFT):
                        adapter_path = ADAPTER_V2_SFT
                    else:
                        print(f"  No adapter available. Skipping {variant}.")
                        continue

            # ── Run custom + when2call (transformers, same loaded model) ──
            if do_custom or do_when2call:
                print(f"\n  Loading model for transformers-based benchmarks...")
                model, tokenizer = load_model(
                    BASE_MODEL,
                    adapter_path=None if is_base else adapter_path,
                )

                if do_custom:
                    print(f"\n  --- Custom Harness ({variant}) ---")
                    custom_result = run_custom_harness(model, tokenizer, limit=limit)
                    if custom_result:
                        all_results[variant]["custom"] = custom_result
                        save_json(custom_result, f"{RESULTS_DIR}/custom_results_{variant}.json")

                if do_when2call:
                    print(f"\n  --- When2Call ({variant}) ---")
                    w2c_result = run_when2call_benchmark(model, tokenizer, limit=limit)
                    if w2c_result:
                        all_results[variant]["when2call"] = w2c_result
                        save_json(w2c_result, f"{RESULTS_DIR}/when2call_results_{variant}.json")

                # Free model from GPU
                del model, tokenizer
                free_gpu_memory()

            # ── Run BFCL (needs merged model on disk) ─────────────────
            if do_bfcl:
                print(f"\n  --- BFCL ({variant}) ---")
                if is_base:
                    bfcl_model_path = BASE_MODEL
                    bfcl_model_name = "Qwen3-4B-Instruct-base"
                else:
                    merged_dir = f"/data/merged_model_{variant}"
                    print(f"  Merging adapter to {merged_dir}...")
                    bfcl_model_path = merge_adapter_to_disk(
                        base_model=BASE_MODEL,
                        adapter_path=adapter_path,
                        output_dir=merged_dir,
                    )
                    bfcl_model_name = f"aiqarus-agent-4b-v2-{variant}"

                bfcl_result = run_bfcl_benchmark(bfcl_model_path, bfcl_model_name)
                if bfcl_result:
                    all_results[variant]["bfcl"] = bfcl_result
                    save_json(bfcl_result, f"{RESULTS_DIR}/bfcl_results_{variant}.json")

                free_gpu_memory()

            volume.commit()

    # ── Generate comparison report ────────────────────────────────
    print(f"\n{'='*60}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*60}")

    report_json = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_model": BASE_MODEL,
        "models_assessed": model_list,
        "benchmarks_run": benchmark,
        "results": all_results,
    }

    save_json(report_json, f"{RESULTS_DIR}/comparative_report.json")

    # Generate Markdown
    md_report = generate_markdown_report(all_results)
    md_path = f"{RESULTS_DIR}/comparative_report.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"Saved: {md_path}")

    # Print summary to console
    print(f"\n{md_report}")

    volume.commit()

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")

    return report_json


@app.local_entrypoint()
def main(
    benchmark: str = "all",
    models: str = "base,dpo",
    limit: int = 0,
    report_only: bool = False,
):
    """
    Run comparative assessment on Modal.

    Flags:
      --benchmark BENCH   Which benchmark: custom, when2call, bfcl, all (default: all)
      --models MODELS     Comma-separated variants: base, sft, dpo (default: base,dpo)
      --limit N           Limit samples per benchmark (smoke test)
      --report-only       Only aggregate existing results into report
    """
    result = run_comparative.remote(
        benchmark=benchmark,
        models=models,
        limit=limit,
        report_only=report_only,
    )

    if result:
        local_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(local_dir, exist_ok=True)

        # Save JSON report
        local_json = os.path.join(local_dir, "comparative_report.json")
        with open(local_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nLocal JSON saved: {local_json}")

        # Save Markdown report
        md_report = generate_markdown_report(result.get("results", {}))
        local_md = os.path.join(local_dir, "comparative_report.md")
        with open(local_md, "w") as f:
            f.write(md_report)
        print(f"Local Markdown saved: {local_md}")
