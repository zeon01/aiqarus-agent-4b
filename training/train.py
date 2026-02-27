"""
aiqarus-agent-4b: QLoRA Fine-Tuning on Modal
==============================================
Two-stage curriculum training of Qwen3-4B-Instruct-2507 via QLoRA on A100 40GB.

  Stage 1 (1 epoch):  Layer 1 foundation data only — builds stable tool-calling base
  Stage 2 (2 epochs): All layers, Layer 3 upsampled 3x — adds reasoning + enterprise behaviors

Config:
  Base:         Qwen/Qwen3-4B-Instruct-2507
  LoRA:         rank=32, alpha=64, dropout=0.05
  Quantize:     4-bit NF4, double quant, bfloat16 compute
  LR:           2e-4 (Stage 1), 1e-4 (Stage 2) — cosine schedule
  Batch:        4 per device × 4 grad accum = 16 effective
  NEFTune:      alpha=5
  Max seq len:  4096
  GPU:          A100 40GB (~3-4 hrs, ~$6-8)

Usage:
  modal run training/train.py
  modal run training/train.py --stage1-only   # debug Stage 1 only
"""

import json
import os
import random
from pathlib import Path

import modal

# ─── Hyperparameters ────────────────────────────────────────────────────────

BASE_MODEL    = "Qwen/Qwen3-4B-Instruct-2507"
VOLUME_NAME   = "aiqarus-data"
DATASET_DIR   = "/data/dataset"
OUTPUT_DIR    = "/data/checkpoints/aiqarus-agent-4b"
ADAPTER_DIR   = "/data/adapter/aiqarus-agent-4b"

LORA_RANK     = 32
LORA_ALPHA    = 64
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

STAGE1_LR     = 2e-4
STAGE2_LR     = 1e-4
LR_SCHEDULER  = "cosine"
WARMUP_RATIO  = 0.10

BATCH_SIZE    = 2
GRAD_ACCUM    = 8         # effective batch size = 16
MAX_SEQ_LEN   = 4096
NEFTUNE_ALPHA = 5

L3_UPSAMPLE   = 3
SEED          = 42

# ─── Modal setup ─────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.4.0",
        "transformers>=5.0.0",
        "peft>=0.14.0",
        "trl>=0.25.0",
        "bitsandbytes>=0.44.0",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "sentencepiece",
        "protobuf",
        "huggingface-hub>=0.25.0",
    ])
)

app    = modal.App("aiqarus-training")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

try:
    hf_secret = modal.Secret.from_name("huggingface-secret")
    secrets = [hf_secret]
except Exception:
    secrets = []


# ─── Data helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


def prepare_messages(sample: dict) -> list[dict] | None:
    """Prepare messages list for SFTTrainer. Embed tool schemas in system if present."""
    messages = list(sample.get("messages", []))
    if len(messages) < 2:
        return None

    tools = sample.get("tools")
    if tools:
        tool_text = "You have access to the following tools:\n\n" + json.dumps(tools, indent=2)
        if messages[0]["role"] == "system":
            content = messages[0]["content"]
            if "available tools" not in content.lower() and '"name"' not in content:
                messages[0] = {"role": "system", "content": content + "\n\n" + tool_text}
        else:
            messages.insert(0, {"role": "system", "content": tool_text})

    return messages


def build_stage1_dataset(train_samples: list[dict]):
    """Stage 1: Layer 1 foundation samples only."""
    from datasets import Dataset

    layer1 = [s for s in train_samples if s.get("layer") == 1]
    random.shuffle(layer1)

    msgs = [m for s in layer1 if (m := prepare_messages(s))]
    print(f"Stage 1: {len(layer1)} Layer 1 samples → {len(msgs)} with messages")
    return Dataset.from_dict({"messages": msgs})


def build_stage2_dataset(train_samples: list[dict]):
    """Stage 2: All layers with Layer 3 upsampled L3_UPSAMPLE×."""
    from datasets import Dataset

    layer3 = [s for s in train_samples if s.get("layer") == 3]
    other  = [s for s in train_samples if s.get("layer") != 3]

    all_samples = other + (layer3 * L3_UPSAMPLE)
    random.shuffle(all_samples)

    msgs = [m for s in all_samples if (m := prepare_messages(s))]
    print(
        f"Stage 2: {len(other)} base + {len(layer3)}×{L3_UPSAMPLE}={len(layer3)*L3_UPSAMPLE} "
        f"Layer 3 = {len(all_samples)} raw → {len(msgs)} with messages"
    )
    return Dataset.from_dict({"messages": msgs})


def build_val_dataset(val_samples: list[dict]):
    from datasets import Dataset
    msgs = [m for s in val_samples if (m := prepare_messages(s))]
    print(f"Val: {len(val_samples)} samples → {len(msgs)} with messages")
    return Dataset.from_dict({"messages": msgs})


# ─── Training function ────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/data": volume},
    timeout=36 * 3600,
    secrets=secrets,
)
def train(stage1_only: bool = False, stage2_only: bool = False):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    random.seed(SEED)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ── Load tokenizer ───────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load dataset ─────────────────────────────────────────────────────
    print(f"Loading dataset from {DATASET_DIR}")
    train_samples = load_jsonl(f"{DATASET_DIR}/train.jsonl")
    val_samples   = load_jsonl(f"{DATASET_DIR}/val.jsonl")
    print(f"  Train: {len(train_samples):,}  Val: {len(val_samples):,}")

    val_ds = build_val_dataset(val_samples)

    # ── QLoRA / BitsAndBytes config ──────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    if stage2_only:
        # ── Load Stage 1 checkpoint ──────────────────────────────────────
        import glob
        from peft import PeftModel
        checkpoints = sorted(glob.glob(f"{OUTPUT_DIR}/stage1/checkpoint-*"))
        if not checkpoints:
            raise FileNotFoundError(f"No Stage 1 checkpoints in {OUTPUT_DIR}/stage1/")
        latest = checkpoints[-1]
        print(f"\nLoading Stage 1 checkpoint: {latest}")
        model = PeftModel.from_pretrained(model, latest, is_trainable=True)
        model.print_trainable_parameters()
    else:
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if not stage2_only:
        # ────────────────────────────────────────────────────────────────────
        # STAGE 1 — Foundation (Layer 1 only, 1 epoch)
        # ────────────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STAGE 1: Foundation (Layer 1 only, 1 epoch, LR=2e-4)")
        print("=" * 60)

        stage1_ds = build_stage1_dataset(train_samples)

        stage1_config = SFTConfig(
            output_dir=f"{OUTPUT_DIR}/stage1",
            num_train_epochs=1,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=STAGE1_LR,
            lr_scheduler_type=LR_SCHEDULER,
            warmup_ratio=WARMUP_RATIO,
            bf16=True,
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=250,
            save_strategy="steps",
            save_steps=250,
            save_total_limit=2,
            load_best_model_at_end=False,
            report_to="none",
            seed=SEED,
            dataloader_num_workers=2,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # SFT-specific
            max_length=MAX_SEQ_LEN,
            packing=False,
            neftune_noise_alpha=NEFTUNE_ALPHA,
        )

        trainer1 = SFTTrainer(
            model=model,
            args=stage1_config,
            train_dataset=stage1_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
        )

        trainer1.train()
        print("Stage 1 complete.")
        volume.commit()

        if stage1_only:
            print("--stage1-only flag set. Stopping after Stage 1.")
            model.save_pretrained(f"{ADAPTER_DIR}-stage1")
            tokenizer.save_pretrained(f"{ADAPTER_DIR}-stage1")
            volume.commit()
            return

    # ────────────────────────────────────────────────────────────────────
    # STAGE 2 — Full curriculum (all layers, Layer 3 3× upsampled, 2 epochs)
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"STAGE 2: Full curriculum (all layers, L3 {L3_UPSAMPLE}× upsample, 2 epochs, LR=1e-4)")
    print("=" * 60)

    stage2_ds = build_stage2_dataset(train_samples)

    stage2_config = SFTConfig(
        output_dir=f"{OUTPUT_DIR}/stage2",
        num_train_epochs=2,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=STAGE2_LR,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # SFT-specific
        max_length=MAX_SEQ_LEN,
        packing=False,
        neftune_noise_alpha=NEFTUNE_ALPHA,
    )

    # Callback to commit volume after each checkpoint save
    from transformers import TrainerCallback

    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            volume.commit()
            print(f"  [volume committed at step {state.global_step}]")

    trainer2 = SFTTrainer(
        model=model,
        args=stage2_config,
        train_dataset=stage2_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[VolumeCommitCallback()],
    )

    trainer2.train()
    print("Stage 2 complete.")

    # ── Save final LoRA adapter ──────────────────────────────────────────
    print(f"\nSaving LoRA adapter → {ADAPTER_DIR}")
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    volume.commit()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Adapter: {ADAPTER_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Download adapter:  modal volume get aiqarus-data adapter/aiqarus-agent-4b .")
    print(f"  2. Merge + push:      python training/merge_and_push.py")
    print("=" * 60)


# ─── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(stage1_only: bool = False, stage2_only: bool = False):
    """
    Run QLoRA fine-tuning on Modal.

    Flags:
      --stage1-only   Run only Stage 1 (useful for quick validation)
      --stage2-only   Skip Stage 1, load Stage 1 checkpoint, run Stage 2 only

    Example:
      modal run training/train.py
      modal run training/train.py --stage1-only
      modal run training/train.py --stage2-only
    """
    train.remote(stage1_only=stage1_only, stage2_only=stage2_only)
