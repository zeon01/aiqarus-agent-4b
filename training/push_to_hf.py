"""
push_to_hf.py — Merge LoRA + push to HuggingFace from Modal
=============================================================
Runs on Modal (data center upload speeds). Merges adapter from volume,
pushes bf16 model directly to HF.

Usage:
  modal run training/push_to_hf.py
  modal run training/push_to_hf.py --dry-run   # merge only, skip push
"""

import modal

BASE_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"
VOLUME_NAME = "aiqarus-data"
ADAPTER_DIR = "/data/adapter/aiqarus-agent-4b"
HF_REPO     = "zeon01/aiqarus-agent-4b"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.4.0",
        "transformers>=5.0.0",
        "peft>=0.14.0",
        "accelerate>=1.0.0",
        "huggingface-hub>=0.25.0",
        "sentencepiece",
        "protobuf",
    ])
)

app    = modal.App("aiqarus-push-hf")
volume = modal.Volume.from_name(VOLUME_NAME)

try:
    hf_secret = modal.Secret.from_name("huggingface-secret")
    secrets = [hf_secret]
except Exception:
    secrets = []


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": volume},
    timeout=2 * 3600,
    secrets=secrets,
    memory=32768,
)
def push(dry_run: bool = False):
    import os
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not dry_run:
        raise RuntimeError(
            "HF_TOKEN not found. Create a Modal secret:\n"
            "  modal secret create huggingface-secret HF_TOKEN=hf_..."
        )

    # ── 1. Load tokenizer ─────────────────────────────────────────────
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # ── 2. Load base model in bf16 ────────────────────────────────────
    print(f"\nLoading base model (bf16): {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    # ── 3. Load LoRA adapter and merge ────────────────────────────────
    print(f"\nLoading LoRA adapter: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)

    print("Merging adapter into base model...")
    model = model.merge_and_unload()
    print("  Merge complete.")

    if dry_run:
        print("\n--dry-run set. Skipping push.")
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        return

    # ── 4. Push to HuggingFace ────────────────────────────────────────
    print(f"\nPushing to HuggingFace: {HF_REPO}")
    model.push_to_hub(HF_REPO, token=hf_token)
    tokenizer.push_to_hub(HF_REPO, token=hf_token)

    print(f"\n{'='*60}")
    print(f"Done! Model live at:")
    print(f"  https://huggingface.co/{HF_REPO}")
    print(f"{'='*60}")


@app.local_entrypoint()
def main(dry_run: bool = False):
    """
    Merge LoRA adapter and push to HuggingFace from Modal.

    Flags:
      --dry-run   Merge only, don't push (verify merge works)
    """
    push.remote(dry_run=dry_run)
