"""
eval_bfcl.py — Berkeley Function Calling Leaderboard on Modal
==============================================================
Runs the official BFCL evaluation suite against our model using vLLM backend.
Merges LoRA adapter to disk first, then calls `bfcl generate` + `bfcl evaluate`.

Metrics: Overall accuracy, per-category accuracy (simple, multiple, parallel,
         parallel_multiple, java, javascript, relevance, etc.)

Usage:
  modal run training/eval_bfcl.py                        # full, best adapter
  modal run training/eval_bfcl.py --base-only             # base Qwen3-4B
  modal run training/eval_bfcl.py --test-category simple  # single category
  modal run training/eval_bfcl.py --adapter /data/adapter/aiqarus-agent-4b-v2
"""

import json
import modal
import os
import subprocess
import sys
import time

# Add training/ to path for eval_utils import
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import (
    BASE_MODEL, VOLUME_NAME, RESULTS_DIR,
    merge_adapter_to_disk, pick_best_adapter, save_json,
)

MERGED_MODEL_DIR = "/data/merged_model"
BFCL_WORKSPACE = "/data/bfcl_workspace"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install([
        "torch>=2.4.0",
        "vllm>=0.7.0",
    ])
    .pip_install([
        "peft>=0.14.0",
        "accelerate>=1.0.0",
        "sentencepiece",
        "protobuf",
        "soundfile",
        "bfcl-eval",
    ])
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "eval_utils.py"),
        remote_path="/root/eval_utils.py",
    )
)

app = modal.App("aiqarus-eval-bfcl")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# BFCL test categories
BFCL_CATEGORIES = [
    "simple", "multiple", "parallel", "parallel_multiple",
    "java", "javascript",
    "relevance", "irrelevance",
    "rest", "sql",
    "live_simple", "live_multiple", "live_parallel",
    "live_parallel_multiple", "live_relevance", "live_irrelevance",
]


def run_cmd(cmd: list[str], env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, printing output in real time."""
    merged_env = {**os.environ, **(env or {})}
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(
        cmd, env=merged_env, capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        print(f"  Command failed with return code {result.returncode}")
    return result


def parse_bfcl_results(workspace: str, model_name: str) -> dict:
    """Parse BFCL evaluation output files into a summary dict."""
    results = {"categories": {}, "overall_accuracy": None}

    # BFCL stores results in score/ directory
    score_dir = os.path.join(workspace, "score", model_name)
    if not os.path.isdir(score_dir):
        # Try alternative paths
        for candidate in [
            os.path.join(workspace, "score"),
            os.path.join(workspace, "result"),
        ]:
            if os.path.isdir(candidate):
                score_dir = candidate
                break

    if not os.path.isdir(score_dir):
        print(f"  Warning: No score directory found at {score_dir}")
        # Try to find any JSON results
        for root, dirs, files in os.walk(workspace):
            for f in files:
                if f.endswith(".json") and "score" in root:
                    print(f"  Found: {os.path.join(root, f)}")
        return results

    # Parse score files
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
                results["categories"][category] = {
                    "accuracy": round(float(accuracy), 4),
                    "correct": correct,
                    "total": total,
                }
                total_correct += correct
                total_samples += total
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Warning: Could not parse {fpath}: {e}")

    if total_samples > 0:
        results["overall_accuracy"] = round(total_correct / total_samples, 4)
        results["total_correct"] = total_correct
        results["total_samples"] = total_samples

    return results


def print_bfcl_summary(results: dict, model_tag: str):
    """Print human-readable BFCL results."""
    print(f"\n{'='*60}")
    print(f"BFCL RESULTS — {model_tag}")
    print(f"{'='*60}")

    if results.get("overall_accuracy") is not None:
        print(f"  Overall accuracy:  {results['overall_accuracy']:.1%}")
        print(f"  Total correct:     {results.get('total_correct', 'N/A')}")
        print(f"  Total samples:     {results.get('total_samples', 'N/A')}")
    else:
        print("  Overall accuracy:  N/A (no scores parsed)")

    if results.get("categories"):
        print(f"\n  Per-category:")
        for cat, data in sorted(results["categories"].items()):
            acc = data.get("accuracy", 0)
            n = data.get("total", 0)
            print(f"    {cat:30s}  {acc:.1%}  (n={n})")

    print(f"{'='*60}")


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": volume},
    timeout=6 * 3600,
    memory=32768,
)
def run_bfcl(
    adapter: str = "",
    base_only: bool = False,
    test_category: str = "",
    limit: int = 0,
):
    """Run BFCL benchmark."""
    # Add eval_utils to path inside Modal
    sys.path.insert(0, "/root")
    from eval_utils import merge_adapter_to_disk, pick_best_adapter, save_json

    start_time = time.time()

    # ── Prepare model ──────────────────────────────────────────────
    # BFCL requires a registered model name for handler/template selection.
    # Use Qwen3-4B-Instruct-2507-FC (native function calling handler).
    # NOTE: BFCL caches results by model name. We clear the cache before each
    # run to ensure fresh generation (base vs finetuned use same registered name).
    model_name = "Qwen/Qwen3-4B-Instruct-2507-FC"

    if base_only:
        model_tag = "base"
        model_path = None  # BFCL downloads from HF
        print(f"Using base model: {BASE_MODEL}")
    else:
        adapter_path = adapter or pick_best_adapter()
        if not adapter_path:
            print("ERROR: No adapter found. Use --base-only or --adapter.")
            return None
        model_tag = "finetuned"
        print(f"Merging adapter {adapter_path} into base model...")
        model_path = merge_adapter_to_disk(
            base_model=BASE_MODEL,
            adapter_path=adapter_path,
            output_dir=MERGED_MODEL_DIR,
        )
        print(f"Merged model saved to {model_path}")

    # ── Cap max_model_len for vLLM (A10G can't fit 262K context KV cache) ──
    config_path = os.path.join(MERGED_MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            model_config = json.load(f)
        orig_len = model_config.get("max_position_embeddings", 0)
        if orig_len > 8192:
            model_config["max_position_embeddings"] = 8192
            with open(config_path, "w") as f:
                json.dump(model_config, f, indent=2)
            print(f"Capped max_position_embeddings to 8192 for vLLM (was {orig_len})")

    # ── Set up BFCL workspace ──────────────────────────────────────
    os.makedirs(BFCL_WORKSPACE, exist_ok=True)

    # Clear any cached results from prior runs so BFCL generates fresh
    import shutil
    model_name_sanitized = model_name.replace("/", "_")
    for subdir in ["result", "score"]:
        cached = os.path.join(BFCL_WORKSPACE, subdir, model_name_sanitized)
        if os.path.isdir(cached):
            shutil.rmtree(cached)
            print(f"  Cleared cached {subdir} for {model_name_sanitized}")
        # Also clear top-level result subdirs that contain model results
        for cat_dir in ["non_live", "live", "multi_turn", "agentic"]:
            cached_cat = os.path.join(BFCL_WORKSPACE, subdir, model_name_sanitized, cat_dir)
            if os.path.isdir(cached_cat):
                shutil.rmtree(cached_cat)
    # Clear CSV score files too (they aggregate across models)
    for csv_file in ["data_overall.csv", "data_non_live.csv", "data_live.csv",
                     "data_multi_turn.csv", "data_agentic.csv", "data_format_sensitivity.csv"]:
        csv_path = os.path.join(BFCL_WORKSPACE, "score", csv_file)
        if os.path.exists(csv_path):
            os.remove(csv_path)

    bfcl_env = {
        "BFCL_PROJECT_ROOT": BFCL_WORKSPACE,
        "VLLM_ENFORCE_EAGER": "1",  # Skip CUDA graph capture — avoids 10-30 min startup hang
    }

    # ── Run BFCL generate ──────────────────────────────────────────
    print("\n--- BFCL Generate ---")
    gen_cmd = [
        "bfcl", "generate",
        "--model", model_name,
        "--backend", "vllm",
        "--num-threads", "1",
        "--gpu-memory-utilization", "0.9",
    ]

    # Point to merged model on disk (finetuned) or let BFCL download (base)
    if model_path:
        gen_cmd.extend(["--local-model-path", model_path])

    if test_category:
        gen_cmd.extend(["--test-category", test_category])

    # Note: BFCL CLI does not support --num-sample. Use --test-category to limit scope.

    gen_result = run_cmd(gen_cmd, env=bfcl_env, check=False)

    if gen_result.returncode != 0:
        print("\nBFCL generate failed.")
        return None

    # ── Run BFCL evaluate ──────────────────────────────────────────
    print("\n--- BFCL Evaluate ---")
    eval_cmd = [
        "bfcl", "evaluate",
        "--model", model_name,
    ]

    if test_category:
        eval_cmd.extend(["--test-category", test_category])

    eval_result = run_cmd(eval_cmd, env=bfcl_env, check=False)

    if eval_result.returncode != 0:
        print("\nBFCL evaluate returned non-zero. Checking for partial results...")

    # ── Parse results ──────────────────────────────────────────────
    bfcl_results = parse_bfcl_results(BFCL_WORKSPACE, model_name)

    result = {
        "model": model_tag,
        "benchmark": "BFCL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "adapter": adapter or (pick_best_adapter() if not base_only else None),
        "model_name": model_name,
        "model_path": model_path,
        **bfcl_results,
    }

    # ── Save results ──────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_json(result, f"{RESULTS_DIR}/bfcl_results_{model_tag}.json")
    volume.commit()

    print_bfcl_summary(bfcl_results, model_tag)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    return result


@app.local_entrypoint()
def main(
    adapter: str = "",
    base_only: bool = False,
    test_category: str = "",
    limit: int = 0,
):
    """
    Run BFCL benchmark on Modal.

    Flags:
      --adapter PATH         Override adapter path
      --base-only            Run on base Qwen3-4B-Instruct (no adapter)
      --test-category CAT    Run single category (e.g. simple, relevance)
      --limit N              Only run first N samples per category
    """
    result = run_bfcl.remote(
        adapter=adapter,
        base_only=base_only,
        test_category=test_category,
        limit=limit,
    )

    if result:
        # Save locally too
        local_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(local_dir, exist_ok=True)
        tag = result["model"]
        local_path = os.path.join(local_dir, f"bfcl_results_{tag}.json")
        with open(local_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nLocal copy saved: {local_path}")
    else:
        print("\nNo result returned. Downloading from Modal volume...")
        import subprocess
        for tag in ["finetuned", "base"]:
            remote = f"results/bfcl_results_{tag}.json"
            local = os.path.join(os.path.dirname(__file__), "..", "data", f"bfcl_results_{tag}.json")
            subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote, local], capture_output=True)
            if os.path.exists(local):
                print(f"  Downloaded: {local}")
