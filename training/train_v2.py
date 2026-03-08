"""
train_v2.py — Round 2 Flattened Curriculum SFT on Modal
========================================================
Single-pass training with all layers mixed from epoch 1.
No stages — fixes Round 1's "always call a tool" bias from staged curriculum.

Key changes from train.py:
  - No Stage 1/Stage 2 split — all data mixed from epoch 1
  - Layer 3 + customs upsampled 3x (baked into dataset by prepare_dataset_v2.py)
  - Configurable LoRA rank (32 or 64)
  - Resume from checkpoint support
  - A100-40GB for larger dataset

Config:
  Base:         Qwen/Qwen3-4B-Instruct-2507
  LoRA:         rank=32 (or 64), alpha=2*rank, dropout=0.05
  Quantize:     4-bit NF4, double quant, bfloat16 compute
  LR:           2e-4, cosine, 10% warmup
  Batch:        4 per device × 4 grad accum = 16 effective
  NEFTune:      alpha=5
  Max seq len:  4096
  Epochs:       2
  GPU:          B200 (~11 hrs, ~$70)

Usage:
  modal run --detach training/train_v2.py
  modal run --detach training/train_v2.py --lora-rank 64
  modal run --detach training/train_v2.py --resume-from /data/checkpoints/aiqarus-agent-4b-v2/checkpoint-1500
"""

import json
import os
import random
from pathlib import Path

import modal

# ─── Hyperparameters ────────────────────────────────────────────────────────

BASE_MODEL    = "Qwen/Qwen3-4B-Instruct-2507"
VOLUME_NAME   = "aiqarus-data"
DATASET_DIR   = "/data/dataset_v2"
OUTPUT_DIR    = "/data/checkpoints/aiqarus-agent-4b-v2"
ADAPTER_DIR   = "/data/adapter/aiqarus-agent-4b-v2"

DEFAULT_LORA_RANK = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

LEARNING_RATE = 2e-4
LR_SCHEDULER  = "cosine"
WARMUP_RATIO  = 0.10
NUM_EPOCHS    = 2         # 8x custom exposure (4x upsample × 2 epochs) — sufficient for 75K dataset

BATCH_SIZE    = 4         # better GPU utilization on B200 (192GB HBM3e)
GRAD_ACCUM    = 4         # effective batch size = 16
MAX_SEQ_LEN   = 4096      # 28/72K samples truncated (0.04%) — avoids quadratic attention cost of 8192
NEFTUNE_ALPHA = 5

SEED          = 42

EVAL_GENERATE_N          = 0      # disabled — run generation separately post-training to avoid overhead
EVAL_GENERATE_DIR        = "/data/eval_outputs/v2"
EVAL_GENERATE_MAX_TOKENS = 2048
EVAL_GENERATE_TEMP       = 0.7

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

app    = modal.App("aiqarus-training-v2")
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
    """Prepare messages list for SFTTrainer. Embed tool schemas in system if present.
    Returns uniform dicts with only 'role' and 'content' keys (pyarrow requires consistent schema)."""
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


def extract_prompt_and_reference(messages):
    """Split messages into prompt (before first assistant) and reference (first assistant response)."""
    prompt = []
    reference = ""
    for msg in messages:
        if msg["role"] == "assistant":
            reference = msg["content"]
            break
        prompt.append(msg)
    return prompt, reference


# ─── Training function ────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": volume},
    timeout=24 * 3600,
    secrets=secrets,
)
def train(lora_rank: int = DEFAULT_LORA_RANK, resume_from: str = "", limit_steps: int = 0, eval_generate: int = EVAL_GENERATE_N):
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

    lora_alpha = lora_rank * 2

    # ── Load tokenizer ───────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Load dataset ─────────────────────────────────────────────────────
    from datasets import Dataset

    print(f"Loading dataset from {DATASET_DIR}")
    train_samples = load_jsonl(f"{DATASET_DIR}/train_v2.jsonl")
    val_samples = load_jsonl(f"{DATASET_DIR}/val_v2.jsonl")
    print(f"  Train: {len(train_samples):,}  Val: {len(val_samples):,}")

    # Prepare messages
    train_msgs = [m for s in train_samples if (m := prepare_messages(s))]
    val_msgs = [m for s in val_samples if (m := prepare_messages(s))]
    print(f"  Train with messages: {len(train_msgs):,}  Val: {len(val_msgs):,}")

    train_ds = Dataset.from_dict({"messages": train_msgs})
    val_ds = Dataset.from_dict({"messages": val_msgs})

    # ── QLoRA / BitsAndBytes config ──────────────────────────────────────
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

    # ── Load model ────────────────────────────────────────────────────────
    print(f"\nLoading model: {BASE_MODEL}")
    print(f"  LoRA rank={lora_rank}, alpha={lora_alpha}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # ── Pre-training truncation audit ────────────────────────────────────
    print("\nTruncation audit (sampling 500 examples)...")
    audit_count = min(500, len(train_msgs))
    truncated = 0
    max_tokens_seen = 0
    for msgs in random.sample(train_msgs, audit_count):
        token_ids = tokenizer.apply_chat_template(msgs, tokenize=True)
        n_tokens = len(token_ids)
        max_tokens_seen = max(max_tokens_seen, n_tokens)
        if n_tokens > MAX_SEQ_LEN:
            truncated += 1
    print(f"  Sampled {audit_count} examples: {truncated} would be truncated ({truncated/audit_count*100:.1f}%)")
    print(f"  Max tokens seen: {max_tokens_seen}, MAX_SEQ_LEN: {MAX_SEQ_LEN}")
    if truncated > audit_count * 0.05:
        print(f"  WARNING: >{5}% truncation! Consider increasing MAX_SEQ_LEN.")

    # ── Volume commit callback ───────────────────────────────────────────
    class VolumeCommitCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            volume.commit()
            print(f"  [volume committed at step {state.global_step}]")

    # ── Generation callback (on-policy outputs for DPO) ──────────────────
    class GenerationCallback(TrainerCallback):
        """Generate on fixed prompts at each evaluation step — on-policy data for DPO."""

        def __init__(self, gen_model, gen_tokenizer, prompts, output_dir):
            self.gen_model = gen_model
            self.gen_tokenizer = gen_tokenizer
            self.prompts = prompts
            self.output_dir = output_dir
            os.makedirs(output_dir, exist_ok=True)

        def on_evaluate(self, args, state, control, **kwargs):
            if not self.prompts:
                return
            step = state.global_step
            print(f"\n  [GenCallback] Generating {len(self.prompts)} samples at step {step}...")

            self.gen_model.eval()
            results = []
            for prompt_data in self.prompts:
                try:
                    text = self.gen_tokenizer.apply_chat_template(
                        prompt_data["prompt"], tokenize=False, add_generation_prompt=True
                    )
                    ids = self.gen_tokenizer(text, return_tensors="pt").input_ids.to(self.gen_model.device)
                    with torch.no_grad():
                        out = self.gen_model.generate(
                            ids,
                            max_new_tokens=EVAL_GENERATE_MAX_TOKENS,
                            temperature=EVAL_GENERATE_TEMP,
                            do_sample=True,
                            top_p=0.9,
                            pad_token_id=self.gen_tokenizer.pad_token_id,
                        )
                    gen = self.gen_tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
                    results.append({
                        "step": step,
                        "idx": prompt_data["idx"],
                        "category": prompt_data["category"],
                        "source": prompt_data["source"],
                        "prompt": prompt_data["prompt"],
                        "reference": prompt_data["reference"],
                        "generated": gen,
                    })
                except Exception as e:
                    print(f"    [gen error]: {e}")

            path = os.path.join(self.output_dir, f"step_{step:05d}.jsonl")
            with open(path, "w") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            volume.commit()
            print(f"  [GenCallback] Saved {len(results)} → {path}")

    # ── Prepare generation prompts from val set ──────────────────────────
    gen_callback = None
    if eval_generate > 0:
        print(f"\nPreparing {eval_generate} generation prompts from val set...")
        prompt_pool = []
        for i, sample in enumerate(val_samples):
            msgs = prepare_messages(sample)
            if msgs is None:
                continue
            prompt, ref = extract_prompt_and_reference(msgs)
            if prompt and ref:
                prompt_pool.append({
                    "idx": i,
                    "category": sample.get("category", "unknown"),
                    "source": sample.get("source", "unknown"),
                    "prompt": prompt,
                    "reference": ref,
                })

        if len(prompt_pool) > eval_generate:
            random.shuffle(prompt_pool)
            prompt_pool = prompt_pool[:eval_generate]

        # Save selected prompts for reproducibility
        prompts_path = os.path.join(EVAL_GENERATE_DIR, "selected_prompts.jsonl")
        os.makedirs(EVAL_GENERATE_DIR, exist_ok=True)
        with open(prompts_path, "w") as f:
            for p in prompt_pool:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

        print(f"  Selected {len(prompt_pool)} prompts → {prompts_path}")
        gen_callback = GenerationCallback(model, tokenizer, prompt_pool, EVAL_GENERATE_DIR)

    # ── Training config ──────────────────────────────────────────────────
    training_args = {
        "output_dir": OUTPUT_DIR,
        "num_train_epochs": NUM_EPOCHS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_type": LR_SCHEDULER,
        "warmup_ratio": WARMUP_RATIO,
        "bf16": True,
        "logging_steps": 25,
        "eval_strategy": "steps",
        "eval_steps": 500,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "seed": SEED,
        "dataloader_num_workers": 2,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_length": MAX_SEQ_LEN,
        "packing": False,
        "neftune_noise_alpha": NEFTUNE_ALPHA,
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
        callbacks=[VolumeCommitCallback()] + ([gen_callback] if gen_callback else []),
    )

    # ── Train ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"TRAINING: {len(train_msgs):,} samples, {NUM_EPOCHS} epochs, "
          f"LoRA r={lora_rank}, LR={LEARNING_RATE}")
    print("=" * 60)

    if resume_from:
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    print("Training complete.")

    # ── Save final LoRA adapter ──────────────────────────────────────────
    print(f"\nSaving LoRA adapter → {ADAPTER_DIR}")
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    volume.commit()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"  Adapter: {ADAPTER_DIR}")
    print(f"  Checkpoints: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Run DPO:  modal run --detach training/dpo_train.py")
    print(f"  2. Or merge: modal run training/push_to_hf.py")
    print("=" * 60)


# ─── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(lora_rank: int = DEFAULT_LORA_RANK, resume_from: str = "", limit_steps: int = 0, eval_generate: int = EVAL_GENERATE_N):
    """
    Run Round 2 flattened curriculum SFT on Modal.

    Flags:
      --lora-rank N         LoRA rank (default: 32)
      --resume-from PATH    Resume from checkpoint path
      --limit-steps N       Max training steps (for testing, 0=unlimited)
      --eval-generate N     Val samples to generate on at each eval step (default: 100, 0=disable)

    Example:
      modal run --detach training/train_v2.py
      modal run --detach training/train_v2.py --lora-rank 64
      modal run --detach training/train_v2.py --eval-generate 50
    """
    train.remote(lora_rank=lora_rank, resume_from=resume_from, limit_steps=limit_steps, eval_generate=eval_generate)
