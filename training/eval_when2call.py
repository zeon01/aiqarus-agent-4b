"""
eval_when2call.py — NVIDIA When2Call MCQ benchmark on Modal
============================================================
Tests "when NOT to call tools" — the core weakness from Round 1.
Runs MCQ assessment via log-probability scoring on the When2Call test split.

Metrics: accuracy, length-normed accuracy, Macro F1, per-category F1,
         tool hallucination rate, confusion matrix.

Usage:
  modal run training/eval_when2call.py                   # full, best adapter
  modal run training/eval_when2call.py --base-only       # base Qwen3-4B
  modal run training/eval_when2call.py --limit 50        # smoke test
  modal run training/eval_when2call.py --adapter /data/adapter/aiqarus-agent-4b-v2
"""

import json
import modal
import os
import sys
import time

# Add training/ to path for eval_utils import
sys.path.insert(0, os.path.dirname(__file__))
from eval_utils import (
    BASE_MODEL, VOLUME_NAME, RESULTS_DIR,
    load_model, pick_best_adapter, save_json, save_jsonl,
)

ANSWER_CATEGORIES = ["direct", "tool_call", "request_for_info", "cannot_answer"]

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
    ])
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "eval_utils.py"),
        remote_path="/root/eval_utils.py",
    )
)

app = modal.App("aiqarus-when2call")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def score_mcq_sample(model, tokenizer, sample: dict) -> str:
    """Compute log-prob for each answer option, return predicted category."""
    import torch

    tools_text = json.dumps(sample["tools"], indent=2) if sample.get("tools") else "No tools available."
    system_msg = (
        "You are an AI assistant with access to the following tools:\n\n"
        f"{tools_text}\n\n"
        "Respond to the user's request appropriately. "
        "Call a tool if needed, ask for clarification if the request is unclear, "
        "answer directly if no tool is required, or say you cannot answer if the "
        "question is outside your capabilities."
    )

    messages_prefix = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": sample["question"]},
    ]

    prefix_text = tokenizer.apply_chat_template(
        messages_prefix, tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(model.device)
    prefix_len = prefix_ids.shape[1]

    best_score = float("-inf")
    best_category = None

    for category in ANSWER_CATEGORIES:
        answer_text = sample["answers"].get(category, "")
        if not answer_text:
            continue

        full_text = prefix_text + answer_text
        full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=8192).input_ids.to(model.device)

        # Skip if answer adds no tokens
        if full_ids.shape[1] <= prefix_len:
            continue

        with torch.no_grad():
            outputs = model(full_ids)
            logits = outputs.logits

        # Log-prob of answer tokens only (shifted by 1 for next-token prediction)
        answer_logits = logits[0, prefix_len - 1:-1, :]
        answer_targets = full_ids[0, prefix_len:]
        log_probs = torch.nn.functional.log_softmax(answer_logits, dim=-1)
        token_log_probs = log_probs.gather(1, answer_targets.unsqueeze(1)).squeeze(1)

        # Length-normalized score (average log-prob per token)
        avg_log_prob = token_log_probs.mean().item()

        if avg_log_prob > best_score:
            best_score = avg_log_prob
            best_category = category

    return best_category or "cannot_answer"


def compute_when2call_metrics(predictions: list[str], ground_truths: list[str]) -> dict:
    """Compute When2Call metrics: accuracy, Macro F1, tool hallucination rate."""
    from sklearn.metrics import f1_score, confusion_matrix, classification_report

    labels = ANSWER_CATEGORIES

    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    accuracy = correct / len(predictions) if predictions else 0

    macro_f1 = f1_score(ground_truths, predictions, labels=labels, average="macro", zero_division=0)

    per_category = {}
    report = classification_report(
        ground_truths, predictions, labels=labels, output_dict=True, zero_division=0,
    )
    for cat in labels:
        if cat in report:
            per_category[cat] = {
                "f1": round(report[cat]["f1-score"], 4),
                "precision": round(report[cat]["precision"], 4),
                "recall": round(report[cat]["recall"], 4),
                "support": int(report[cat]["support"]),
            }

    cm = confusion_matrix(ground_truths, predictions, labels=labels)

    # Tool hallucination rate: predicted tool_call when ground truth != tool_call
    non_tool_cases = [(p, g) for p, g in zip(predictions, ground_truths) if g != "tool_call"]
    hallucinations = sum(1 for p, _ in non_tool_cases if p == "tool_call")
    hallucination_rate = hallucinations / len(non_tool_cases) if non_tool_cases else 0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "tool_hallucination_rate": round(hallucination_rate, 4),
        "per_category": per_category,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(predictions),
        "total_correct": correct,
    }


def print_when2call_summary(metrics: dict, model_tag: str):
    """Print human-readable When2Call results."""
    print(f"\n{'='*60}")
    print(f"WHEN2CALL RESULTS — {model_tag}")
    print(f"{'='*60}")
    print(f"  Accuracy:              {metrics['accuracy']:.1%}")
    print(f"  Macro F1:              {metrics['macro_f1']:.4f}")
    print(f"  Tool hallucination:    {metrics['tool_hallucination_rate']:.1%}")
    print(f"  Total samples:         {metrics['total_samples']}")
    print(f"  Total correct:         {metrics['total_correct']}")

    print(f"\n  Per-category F1:")
    for cat, data in metrics.get("per_category", {}).items():
        print(f"    {cat:25s}  F1={data['f1']:.4f}  P={data['precision']:.4f}  R={data['recall']:.4f}  n={data['support']}")

    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    print(f"    {'':25s} {'direct':>10s} {'tool_call':>10s} {'req_info':>10s} {'cannot':>10s}")
    labels = ANSWER_CATEGORIES
    for i, row in enumerate(metrics.get("confusion_matrix", [])):
        print(f"    {labels[i]:25s} {row[0]:>10d} {row[1]:>10d} {row[2]:>10d} {row[3]:>10d}")
    print(f"{'='*60}")


@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": volume},
    timeout=4 * 3600,
    memory=32768,
)
def run_when2call(
    adapter: str = "",
    base_only: bool = False,
    limit: int = 0,
):
    """Run When2Call MCQ benchmark."""
    # Add eval_utils to path inside Modal
    sys.path.insert(0, "/root")
    from eval_utils import load_model, pick_best_adapter, save_json, save_jsonl

    from datasets import load_dataset

    # ── Load model ────────────────────────────────────────────────────
    if base_only:
        model_tag = "base"
        model, tokenizer = load_model(BASE_MODEL, adapter_path=None)
    else:
        adapter_path = adapter or pick_best_adapter()
        if not adapter_path:
            print("ERROR: No adapter found. Use --base-only or --adapter.")
            return None
        model_tag = "finetuned"
        model, tokenizer = load_model(BASE_MODEL, adapter_path=adapter_path)

    # ── Load When2Call test data ──────────────────────────────────────
    print("Loading When2Call MCQ test split from HuggingFace...")
    ds = load_dataset("nvidia/When2Call", split="mcq")

    # Filter for MCQ format samples (have 'answers' dict)
    mcq_samples = []
    for sample in ds:
        if sample.get("answers") and sample.get("correct_answer"):
            mcq_samples.append(sample)

    print(f"Loaded {len(mcq_samples)} MCQ samples.")

    if limit > 0:
        mcq_samples = mcq_samples[:limit]
        print(f"  (limited to {limit})")

    # ── Run MCQ scoring ───────────────────────────────────────────────
    predictions = []
    ground_truths = []
    detailed = []
    start_time = time.time()

    for i, sample in enumerate(mcq_samples):
        predicted = score_mcq_sample(model, tokenizer, sample)
        ground_truth = sample["correct_answer"]

        predictions.append(predicted)
        ground_truths.append(ground_truth)
        detailed.append({
            "uuid": sample.get("uuid", f"sample_{i}"),
            "predicted": predicted,
            "ground_truth": ground_truth,
            "correct": predicted == ground_truth,
            "source": sample.get("source", "unknown"),
        })

        if (i + 1) % 50 == 0:
            running_acc = sum(r["correct"] for r in detailed) / len(detailed)
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(mcq_samples) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(mcq_samples)}] acc={running_acc:.1%} "
                  f"({rate:.1f} samples/s, ETA {eta/60:.0f}m)")

    # ── Compute metrics ───────────────────────────────────────────────
    metrics = compute_when2call_metrics(predictions, ground_truths)

    result = {
        "model": model_tag,
        "benchmark": "When2Call",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "adapter": adapter or (pick_best_adapter() if not base_only else None),
        **metrics,
    }

    # ── Save results ──────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_json(result, f"{RESULTS_DIR}/when2call_results_{model_tag}.json")
    save_jsonl(detailed, f"{RESULTS_DIR}/when2call_detailed_{model_tag}.jsonl")
    volume.commit()

    print_when2call_summary(metrics, model_tag)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    return result


@app.local_entrypoint()
def main(
    adapter: str = "",
    base_only: bool = False,
    limit: int = 0,
):
    """
    Run When2Call MCQ benchmark on Modal.

    Flags:
      --adapter PATH   Override adapter path
      --base-only      Run on base Qwen3-4B-Instruct (no adapter)
      --limit N        Only score first N samples (smoke test)
    """
    result = run_when2call.remote(
        adapter=adapter,
        base_only=base_only,
        limit=limit,
    )

    if result:
        # Save locally too
        local_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(local_dir, exist_ok=True)
        tag = result["model"]
        # Summary
        local_path = os.path.join(local_dir, f"when2call_results_{tag}.json")
        with open(local_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nLocal copy saved: {local_path}")
    else:
        print("\nNo result returned. Downloading from Modal volume...")
        import subprocess
        for tag in ["finetuned", "base"]:
            for suffix in [f"when2call_results_{tag}.json", f"when2call_detailed_{tag}.jsonl"]:
                remote = f"results/{suffix}"
                local = os.path.join(os.path.dirname(__file__), "..", "data", suffix)
                subprocess.run(["modal", "volume", "get", VOLUME_NAME, remote, local], capture_output=True)
                if os.path.exists(local):
                    print(f"  Downloaded: {local}")
