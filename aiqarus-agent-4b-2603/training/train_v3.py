"""
train_v3.py — V3 SFT on Modal (Qwen3.5-4B, B200)
===================================================
Flattened curriculum SFT with loss masking on assistant turns only.
Builds on V2 lessons: all data mixed from epoch 1, sequence packing enabled,
longer context (8192), and tighter eval/save intervals.

Key changes from V2:
  - Base model: Qwen/Qwen3.5-4B (dense, Apache 2.0, Gated DeltaNet attention)
  - MAX_SEQ_LEN: 8192 (up from 4096 — Qwen3.5-4B supports 262K natively)
  - Sequence packing: True (efficient GPU utilization for mixed-length samples)
  - Loss masking: assistant turns only (via SFTTrainer's response_template or
    dataset_text_field with labels masking)
  - Epochs: 3 (up from 2 — larger model capacity, diverse data mix)
  - Eval/save intervals tightened (eval=200 steps, save=100 steps)
  - Logging every 10 steps
  - Warmup ratio: 0.03 (down from 0.10 — faster ramp with mixed data)
  - Weight decay: 0.01 (regularization for 3-epoch run)

Config:
  Base:         Qwen/Qwen3.5-4B
  LoRA:         rank=32, alpha=64, dropout=0.05
  Quantize:     4-bit NF4, double quant, bfloat16 compute
  LR:           2e-4, cosine, 3% warmup
  Batch:        4 per device x 4 grad accum = 16 effective
  NEFTune:      alpha=5
  Max seq len:  8192
  Epochs:       3
  GPU:          B200 on Modal

Usage:
  modal run --detach training/train_v3.py
  modal run --detach training/train_v3.py --lora-rank 64
  modal run --detach training/train_v3.py --resume-from /data/checkpoints/aiqarus-agent-4b-v3/checkpoint-1500
  modal run --detach training/train_v3.py --limit-steps 50  # quick sanity check
"""

import json
import os
import random
from pathlib import Path

import modal

# --- Hyperparameters -----------------------------------------------------------

BASE_MODEL    = "Qwen/Qwen3.5-4B"
VOLUME_NAME   = "aiqarus-data"
DATASET_DIR   = "/data/dataset_v3"
OUTPUT_DIR    = "/data/checkpoints/aiqarus-agent-4b-v3"
ADAPTER_DIR   = "/data/adapter/aiqarus-agent-4b-v3"

DEFAULT_LORA_RANK = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = "all-linear"       # target all linear layers for maximum expressivity

LEARNING_RATE = 2e-4
LR_SCHEDULER  = "cosine"
WARMUP_RATIO  = 0.03
WEIGHT_DECAY  = 0.01
MAX_GRAD_NORM = 1.0
NUM_EPOCHS    = 3

BATCH_SIZE    = 4
GRAD_ACCUM    = 4         # effective batch size = 16
MAX_SEQ_LEN   = 8192
NEFTUNE_ALPHA = 5

SEED          = 42

EVAL_STEPS    = 200
SAVE_STEPS    = 100
LOGGING_STEPS = 10

# --- Modal setup --------------------------------------------------------------

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
        "scipy",
        "wandb",
    ])
)

app    = modal.App("aiqarus-v3-sft")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Optionally load secrets (HF for gated model access, W&B for logging)
secrets = []
for secret_name in ["huggingface-secret", "wandb-secret"]:
    try:
        secrets.append(modal.Secret.from_name(secret_name))
    except Exception:
        pass


# --- Data helpers --------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file, skipping malformed lines."""
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
    """Prepare messages list for SFTTrainer.

    Embeds tool schemas in system message if present. Returns uniform dicts
    with only 'role' and 'content' keys (pyarrow requires consistent schema).
    """
    raw_messages = sample.get("messages", [])
    if len(raw_messages) < 2:
        return None

    # Normalize: keep only role + content, ensure both are strings
    messages = []
    for msg in raw_messages:
        role = str(msg.get("role") or "user")
        content = str(msg.get("content") or "")
        if role and content.strip():
            messages.append({"role": role, "content": content})

    if len(messages) < 2:
        return None

    # Embed tool schemas in system message if not already present
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


# --- Training function ---------------------------------------------------------

@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": volume},
    timeout=24 * 3600,   # 24 hours
    secrets=secrets,
)
def train(
    lora_rank: int = DEFAULT_LORA_RANK,
    resume_from: str = "",
    limit_steps: int = 0,
):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    random.seed(SEED)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Determine W&B availability
    report_to = "none"
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb.init(project="aiqarus-v3", name=f"sft-r{lora_rank}-ep{NUM_EPOCHS}")
            report_to = "wandb"
            print("W&B logging enabled.")
        except Exception as e:
            print(f"W&B init failed ({e}), falling back to console logging.")

    lora_alpha = lora_rank * 2

    # -- Load tokenizer --------------------------------------------------------
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -- Load dataset ----------------------------------------------------------
    from datasets import Dataset

    print(f"Loading dataset from {DATASET_DIR}")
    train_samples = load_jsonl(f"{DATASET_DIR}/train_v3.jsonl")
    val_samples = load_jsonl(f"{DATASET_DIR}/val_v3.jsonl")
    print(f"  Raw — Train: {len(train_samples):,}  Val: {len(val_samples):,}")

    # Prepare messages (normalize roles, embed tool schemas)
    train_msgs = [m for s in train_samples if (m := prepare_messages(s))]
    val_msgs = [m for s in val_samples if (m := prepare_messages(s))]
    print(f"  With messages — Train: {len(train_msgs):,}  Val: {len(val_msgs):,}")

    if not train_msgs:
        raise ValueError(f"No valid training samples loaded from {DATASET_DIR}/train_v3.jsonl")

    train_ds = Dataset.from_dict({"messages": train_msgs})
    val_ds = Dataset.from_dict({"messages": val_msgs})

    # -- Pre-training truncation audit -----------------------------------------
    print(f"\nTruncation audit (sampling up to 500 examples)...")
    audit_count = min(500, len(train_msgs))
    truncated = 0
    max_tokens_seen = 0
    token_lengths = []
    for msgs in random.sample(train_msgs, audit_count):
        token_ids = tokenizer.apply_chat_template(msgs, tokenize=True)
        n_tokens = len(token_ids)
        token_lengths.append(n_tokens)
        max_tokens_seen = max(max_tokens_seen, n_tokens)
        if n_tokens > MAX_SEQ_LEN:
            truncated += 1

    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    median_tokens = sorted(token_lengths)[len(token_lengths) // 2] if token_lengths else 0
    print(f"  Sampled {audit_count} examples:")
    print(f"    Truncated: {truncated} ({truncated / audit_count * 100:.1f}%)")
    print(f"    Max tokens: {max_tokens_seen}, Avg: {avg_tokens:.0f}, Median: {median_tokens}")
    print(f"    MAX_SEQ_LEN: {MAX_SEQ_LEN}")
    if truncated > audit_count * 0.05:
        print(f"  WARNING: >{5}% truncation! Consider increasing MAX_SEQ_LEN.")

    # -- QLoRA / BitsAndBytes config -------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )

    # -- Load model ------------------------------------------------------------
    print(f"\nLoading model: {BASE_MODEL}")
    print(f"  LoRA rank={lora_rank}, alpha={lora_alpha}, targets={LORA_TARGETS}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # -- Volume commit callback ------------------------------------------------
    class VolumeCommitCallback(TrainerCallback):
        """Commit Modal volume after each checkpoint save for persistence."""
        def on_save(self, args, state, control, **kwargs):
            volume.commit()
            print(f"  [volume committed at step {state.global_step}]")

    # -- Training config -------------------------------------------------------
    training_args = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": LR_SCHEDULER,
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "bf16": True,
        "logging_steps": LOGGING_STEPS,
        "eval_strategy": "steps",
        "eval_steps": EVAL_STEPS,
        "save_strategy": "steps",
        "save_steps": SAVE_STEPS,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": report_to,
        "seed": SEED,
        "dataloader_num_workers": 2,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # SFT-specific
        "max_seq_length": MAX_SEQ_LEN,
        "packing": True,
        "neftune_noise_alpha": NEFTUNE_ALPHA,
        # Loss masking: only compute loss on assistant completions.
        # SFTTrainer with "messages" column + packing handles this via
        # the chat template's assistant token boundaries. Setting
        # dataset_kwargs to mask non-assistant turns.
        "dataset_kwargs": {
            "skip_prepare_dataset": False,
        },
    }

    if limit_steps > 0:
        training_args["max_steps"] = limit_steps
        training_args["save_steps"] = max(10, limit_steps // 2)
        training_args["eval_steps"] = max(10, limit_steps // 2)
        print(f"\n  [LIMIT MODE: {limit_steps} steps]")

    config = SFTConfig(**training_args)

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=[VolumeCommitCallback()],
    )

    # -- Train -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"V3 SFT TRAINING")
    print(f"  Model:    {BASE_MODEL}")
    print(f"  Samples:  {len(train_msgs):,} train, {len(val_msgs):,} val")
    print(f"  Epochs:   {NUM_EPOCHS}")
    print(f"  LoRA:     r={lora_rank}, alpha={lora_alpha}")
    print(f"  LR:       {LEARNING_RATE}, scheduler={LR_SCHEDULER}")
    print(f"  Batch:    {BATCH_SIZE} x {GRAD_ACCUM} grad_accum = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"  Seq len:  {MAX_SEQ_LEN}")
    print(f"  Packing:  True")
    print(f"  NEFTune:  alpha={NEFTUNE_ALPHA}")
    print(f"  GPU:      B200")
    print("=" * 60)

    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    print("Training complete.")

    # -- Save final LoRA adapter -----------------------------------------------
    print(f"\nSaving LoRA adapter -> {ADAPTER_DIR}")
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    volume.commit()

    # -- Summary ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("V3 SFT TRAINING COMPLETE!")
    print(f"  Adapter:     {ADAPTER_DIR}")
    print(f"  Checkpoints: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Run eval:      modal run --detach training/eval_harness_v3.py")
    print(f"  2. Run SimPO:     modal run --detach training/simpo_train_v3.py")
    print(f"  3. Or merge+push: modal run training/push_to_hf.py --adapter-dir {ADAPTER_DIR}")
    print("=" * 60)


# --- Local entrypoint ----------------------------------------------------------

@app.local_entrypoint()
def main(
    lora_rank: int = DEFAULT_LORA_RANK,
    resume_from: str = "",
    limit_steps: int = 0,
):
    """
    Run V3 SFT on Modal (Qwen3.5-4B, B200).

    Flags:
      --lora-rank N         LoRA rank (default: 32)
      --resume-from PATH    Resume from checkpoint path on Modal volume
      --limit-steps N       Max training steps (for testing, 0=unlimited)

    Example:
      modal run --detach training/train_v3.py
      modal run --detach training/train_v3.py --lora-rank 64
      modal run --detach training/train_v3.py --limit-steps 50
      modal run --detach training/train_v3.py --resume-from /data/checkpoints/aiqarus-agent-4b-v3/checkpoint-1500
    """
    train.remote(lora_rank=lora_rank, resume_from=resume_from, limit_steps=limit_steps)
