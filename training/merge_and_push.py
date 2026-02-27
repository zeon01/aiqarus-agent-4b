"""
merge_and_push.py
=================
Merge LoRA adapter into base model, save fp32 locally (safekeeping),
convert to bf16, and push to HuggingFace.

Steps:
  1. Load base model in fp32 on CPU
  2. Load LoRA adapter and merge
  3. Save fp32 locally (~8-9 GB) — golden copy
  4. Convert to bf16 (~4.5 GB)
  5. Push bf16 to HuggingFace

Usage:
  python training/merge_and_push.py
  python training/merge_and_push.py --skip-push     # merge only, no HF upload
  python training/merge_and_push.py --fp32-only     # save fp32 only, stop there

Prerequisites:
  pip install torch transformers peft huggingface_hub
  huggingface-cli login   (or set HF_TOKEN in .env)
"""

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_DIR = "aiqarus-agent-4b"       # downloaded from Modal volume
FP32_DIR    = "aiqarus-agent-4b-fp32"  # local safekeeping (~8-9 GB)
BF16_DIR    = "aiqarus-agent-4b-bf16"  # for HF upload (~4.5 GB)
HF_REPO     = "zeon01/aiqarus-agent-4b"

# ─── HF token ─────────────────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

hf_token = os.environ.get("HF_TOKEN")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(skip_push: bool = False, fp32_only: bool = False):

    # ── 1. Load tokenizer ─────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # ── 2. Load base model in fp32 on CPU ────────────────────────────────────
    print(f"\nLoading base model (fp32, CPU): {BASE_MODEL}")
    print("  Expected RAM: ~8-9 GB")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    # ── 3. Load LoRA adapter and merge ────────────────────────────────────────
    if not Path(ADAPTER_DIR).exists():
        raise FileNotFoundError(
            f"Adapter not found at '{ADAPTER_DIR}'. "
            "Download it first:\n"
            "  modal volume get aiqarus-data adapter/aiqarus-agent-4b ."
        )

    print(f"\nLoading LoRA adapter: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()
    print("  Merge complete.")

    # ── 4. Save fp32 locally (golden copy) ────────────────────────────────────
    print(f"\nSaving fp32 → {FP32_DIR}  (~8-9 GB)")
    os.makedirs(FP32_DIR, exist_ok=True)
    model.save_pretrained(FP32_DIR, safe_serialization=True)
    tokenizer.save_pretrained(FP32_DIR)
    print(f"  Saved to {FP32_DIR}")

    if fp32_only:
        print("\nDone (--fp32-only). Skipping bf16 and HF push.")
        return

    # ── 5. Convert to bf16 ────────────────────────────────────────────────────
    print(f"\nConverting to bf16 → {BF16_DIR}  (~4.5 GB)")
    model = model.to(torch.bfloat16)
    os.makedirs(BF16_DIR, exist_ok=True)
    model.save_pretrained(BF16_DIR, safe_serialization=True)
    tokenizer.save_pretrained(BF16_DIR)
    print(f"  Saved to {BF16_DIR}")

    if skip_push:
        print("\nDone (--skip-push). Not uploading to HuggingFace.")
        return

    # ── 6. Push bf16 to HuggingFace ───────────────────────────────────────────
    print(f"\nPushing to HuggingFace: {HF_REPO}")
    if not hf_token:
        print("  HF_TOKEN not found in env — using cached credentials from huggingface-cli login")

    model.push_to_hub(
        HF_REPO,
        safe_serialization=True,
        token=hf_token,
    )
    tokenizer.push_to_hub(HF_REPO, token=hf_token)

    print(f"\n{'='*60}")
    print(f"Done! Model live at:")
    print(f"  https://huggingface.co/{HF_REPO}")
    print(f"\nLocal copies:")
    print(f"  fp32 (safekeeping): {FP32_DIR}/")
    print(f"  bf16 (HF copy):     {BF16_DIR}/")
    print(f"{'='*60}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and push to HuggingFace")
    parser.add_argument(
        "--skip-push", action="store_true",
        help="Merge and save locally, but don't push to HuggingFace"
    )
    parser.add_argument(
        "--fp32-only", action="store_true",
        help="Save fp32 golden copy only, skip bf16 conversion and push"
    )
    args = parser.parse_args()
    main(skip_push=args.skip_push, fp32_only=args.fp32_only)
