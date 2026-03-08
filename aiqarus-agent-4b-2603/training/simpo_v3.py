"""
simpo_v3.py — V3 SimPO Alignment on Modal (Qwen3.5-4B, B200)
==============================================================
Runs SimPO (Simple Preference Optimization) on top of the V3 SFT adapter.
Reference-free: no dual-adapter trick needed, simpler and more VRAM-efficient.

Uses on-policy preference pairs generated from V3 SFT model failures
paired with corrected responses.

Falls back to KTO (Kahneman-Tversky Optimization) if SimPO doesn't converge.
KTO doesn't require paired data — just good/bad labels.

Dataset format (in simpo_pairs.jsonl):
  {"prompt": [{"role": "system", ...}, {"role": "user", ...}],
   "chosen": [{"role": "assistant", "content": "..."}],
   "rejected": [{"role": "assistant", "content": "..."}]}

Config:
  Base:         Qwen/Qwen3.5-4B
  SFT adapter:  /data/adapter/aiqarus-agent-4b-v3/
  Beta:         1.0 (conservative start)
  Gamma:        0.3 (gentle target margin)
  LR:           5e-7
  Max grad norm: 0.5 (tight clipping)
  Epochs:       2-3
  Batch:        2 per device x 4 grad accum = 8 effective
  Max length:   4096
  GPU:          B200 (~1-2 hrs)

Usage:
  modal run --detach training/simpo_v3.py
  modal run --detach training/simpo_v3.py --beta 0.5 --gamma 0.5
  modal run --detach training/simpo_v3.py --limit-steps 20
  modal run --detach training/simpo_v3.py --use-kto  # KTO fallback
"""

import json
import os
import random
from pathlib import Path

import modal

# --- Hyperparameters ----------------------------------------------------------

BASE_MODEL    = "Qwen/Qwen3.5-4B"
VOLUME_NAME   = "aiqarus-data"
DATASET_DIR   = "/data/dataset_v3"
SFT_ADAPTER   = "/data/adapter/aiqarus-agent-4b-v3"
OUTPUT_DIR    = "/data/checkpoints/aiqarus-agent-4b-v3-simpo"
ADAPTER_DIR   = "/data/adapter/aiqarus-agent-4b-v3-simpo"

# SimPO
DEFAULT_BETA  = 1.0        # conservative start
DEFAULT_GAMMA = 0.3        # gentle target margin
LEARNING_RATE = 5e-7
LR_SCHEDULER  = "cosine"
WARMUP_RATIO  = 0.15       # longer warmup for stability
DEFAULT_EPOCHS = 3
MAX_GRAD_NORM = 0.5        # tight clipping to prevent NaN explosion

# KTO fallback
KTO_BETA      = 0.1

# Shared
BATCH_SIZE    = 2
GRAD_ACCUM    = 4           # effective batch size = 8
MAX_LENGTH    = 4096
LORA_RANK     = 32

SEED          = 42

# Early stopping: halt if reward margin stays negative for this many steps
NEGATIVE_MARGIN_PATIENCE = 10

# --- Modal setup -------------------------------------------------------------

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

app    = modal.App("aiqarus-v3-simpo")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Optionally load secrets (HF for gated model access)
secrets = []
for secret_name in ["huggingface-secret", "wandb-secret"]:
    try:
        secrets.append(modal.Secret.from_name(secret_name))
    except Exception:
        pass


# --- Data helpers -------------------------------------------------------------

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


def load_and_split_pairs(pairs_path: str, val_ratio: float = 0.05):
    """Load preference pairs from JSONL, validate, and split train/val.

    Returns (train_ds, val_ds) as HF Datasets.
    """
    from datasets import Dataset

    print(f"Loading preference pairs from {pairs_path}")
    raw_pairs = load_jsonl(pairs_path)
    print(f"  Loaded {len(raw_pairs)} pairs")

    if not raw_pairs:
        raise ValueError(f"No preference pairs found at {pairs_path}")

    # Validate and extract fields
    prompts = []
    chosen_list = []
    rejected_list = []

    for pair in raw_pairs:
        prompt = pair.get("prompt", [])
        chosen = pair.get("chosen", [])
        rejected = pair.get("rejected", [])

        if not prompt or not chosen or not rejected:
            continue

        prompts.append(prompt)
        chosen_list.append(chosen)
        rejected_list.append(rejected)

    print(f"  Valid pairs: {len(prompts)}")

    if not prompts:
        raise ValueError("Zero valid pairs after filtering!")

    # Split train/val
    n_val = max(1, int(len(prompts) * val_ratio))
    indices = list(range(len(prompts)))
    random.shuffle(indices)

    val_idx = set(indices[:n_val])
    train_prompts = [prompts[i] for i in range(len(prompts)) if i not in val_idx]
    train_chosen = [chosen_list[i] for i in range(len(prompts)) if i not in val_idx]
    train_rejected = [rejected_list[i] for i in range(len(prompts)) if i not in val_idx]

    val_prompts = [prompts[i] for i in val_idx]
    val_chosen = [chosen_list[i] for i in val_idx]
    val_rejected = [rejected_list[i] for i in val_idx]

    train_ds = Dataset.from_dict({
        "prompt": train_prompts,
        "chosen": train_chosen,
        "rejected": train_rejected,
    })
    val_ds = Dataset.from_dict({
        "prompt": val_prompts,
        "chosen": val_chosen,
        "rejected": val_rejected,
    })

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    return train_ds, val_ds


def load_kto_data(pairs_path: str, val_ratio: float = 0.05):
    """Convert preference pairs to KTO format (unpaired good/bad labels).

    Each pair produces two KTO examples:
      - chosen response with label=True  (desirable)
      - rejected response with label=False (undesirable)

    Returns (train_ds, val_ds) as HF Datasets.
    """
    from datasets import Dataset

    print(f"Loading preference pairs for KTO from {pairs_path}")
    raw_pairs = load_jsonl(pairs_path)
    print(f"  Loaded {len(raw_pairs)} pairs")

    if not raw_pairs:
        raise ValueError(f"No preference pairs found at {pairs_path}")

    prompts = []
    completions = []
    labels = []

    for pair in raw_pairs:
        prompt = pair.get("prompt", [])
        chosen = pair.get("chosen", [])
        rejected = pair.get("rejected", [])

        if not prompt or not chosen or not rejected:
            continue

        # Desirable: chosen response
        prompts.append(prompt)
        completions.append(chosen)
        labels.append(True)

        # Undesirable: rejected response
        prompts.append(prompt)
        completions.append(rejected)
        labels.append(False)

    print(f"  KTO examples: {len(prompts)} ({sum(labels)} desirable, {len(labels) - sum(labels)} undesirable)")

    if not prompts:
        raise ValueError("Zero valid KTO examples after filtering!")

    # Shuffle before split so desirable/undesirable are mixed
    combined = list(zip(prompts, completions, labels))
    random.shuffle(combined)
    prompts, completions, labels = zip(*combined)
    prompts, completions, labels = list(prompts), list(completions), list(labels)

    # Split train/val
    n_val = max(2, int(len(prompts) * val_ratio))
    train_ds = Dataset.from_dict({
        "prompt": prompts[n_val:],
        "completion": completions[n_val:],
        "label": labels[n_val:],
    })
    val_ds = Dataset.from_dict({
        "prompt": prompts[:n_val],
        "completion": completions[:n_val],
        "label": labels[:n_val],
    })

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
    return train_ds, val_ds


# --- Training function --------------------------------------------------------

@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": volume},
    timeout=12 * 3600,    # 12 hours
    secrets=secrets,
)
def train_simpo(
    adapter_path: str = SFT_ADAPTER,
    pairs_path: str = f"{DATASET_DIR}/simpo_pairs.jsonl",
    epochs: int = DEFAULT_EPOCHS,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    limit_steps: int = 0,
    use_kto: bool = False,
):
    import math

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from datasets import Dataset

    random.seed(SEED)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    mode = "KTO" if use_kto else "SimPO"

    # -- Log all hyperparameters -----------------------------------------------
    print("\n" + "=" * 60)
    print(f"V3 {mode} ALIGNMENT")
    print("=" * 60)
    print(f"  Base model:      {BASE_MODEL}")
    print(f"  SFT adapter:     {adapter_path}")
    print(f"  Pairs path:      {pairs_path}")
    print(f"  Mode:            {mode}")
    if use_kto:
        print(f"  KTO beta:        {KTO_BETA}")
    else:
        print(f"  SimPO beta:      {beta}")
        print(f"  SimPO gamma:     {gamma}")
    print(f"  LR:              {LEARNING_RATE}")
    print(f"  Scheduler:       {LR_SCHEDULER}")
    print(f"  Warmup ratio:    {WARMUP_RATIO}")
    print(f"  Max grad norm:   {MAX_GRAD_NORM}")
    print(f"  Epochs:          {epochs}")
    print(f"  Batch:           {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective")
    print(f"  Max length:      {MAX_LENGTH}")
    print(f"  LoRA rank:       {LORA_RANK}")
    print(f"  Seed:            {SEED}")
    if limit_steps > 0:
        print(f"  Limit steps:     {limit_steps}")
    print("=" * 60)

    # -- Load tokenizer --------------------------------------------------------
    print(f"\nLoading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # -- Load dataset ----------------------------------------------------------
    if use_kto:
        train_ds, val_ds = load_kto_data(pairs_path)
    else:
        train_ds, val_ds = load_and_split_pairs(pairs_path)

    # -- Load base model in 4-bit ----------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"\nLoading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    # -- Load SFT adapter (trainable) ------------------------------------------
    print(f"\nLoading SFT adapter from {adapter_path}")
    if not Path(adapter_path).exists():
        print(f"ERROR: SFT adapter not found at {adapter_path}")
        print("Run training/train_v3.py first!")
        return

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=True,
    )
    print(f"  Loaded SFT adapter (trainable, reference-free for {mode})")
    model.print_trainable_parameters()

    # -- Callbacks -------------------------------------------------------------
    class VolumeCommitCallback(TrainerCallback):
        """Commit Modal volume after each checkpoint save for persistence."""
        def on_save(self, args, state, control, **kwargs):
            volume.commit()
            print(f"  [volume committed at step {state.global_step}]")

    class NaNDetectionCallback(TrainerCallback):
        """Stop training immediately if NaN/Inf/diverged loss detected."""
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss_val = logs["loss"]
                if isinstance(loss_val, (int, float)) and (
                    math.isnan(loss_val) or math.isinf(loss_val) or loss_val > 10.0
                ):
                    print(f"\n  WARNING: STOPPING — loss={loss_val} at step {state.global_step} (NaN/diverged)")
                    print("  Saving emergency checkpoint before halting...")
                    control.should_training_stop = True
                    control.should_save = True

    class RewardMarginMonitor(TrainerCallback):
        """Monitor reward margins and trigger early stopping if negative too long.

        For SimPO: tracks rewards/margins logged by CPOTrainer.
        Prints chosen vs rejected log-prob means and warns on negative margins.
        """
        def __init__(self, patience: int = NEGATIVE_MARGIN_PATIENCE):
            self.patience = patience
            self.consecutive_negative = 0
            self.step_rewards = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return

            # CPOTrainer logs these keys
            margin = logs.get("rewards/margins", None)
            chosen = logs.get("rewards/chosen", None)
            rejected = logs.get("rewards/rejected", None)

            if margin is not None:
                self.step_rewards.append({
                    "step": state.global_step,
                    "margin": margin,
                    "chosen": chosen,
                    "rejected": rejected,
                })

                # Print comparison
                chosen_str = f"{chosen:.4f}" if chosen is not None else "N/A"
                rejected_str = f"{rejected:.4f}" if rejected is not None else "N/A"
                print(f"  [Step {state.global_step}] Margin: {margin:.4f} | Chosen: {chosen_str} | Rejected: {rejected_str}")

                if margin < 0:
                    self.consecutive_negative += 1
                    print(f"  WARNING: Negative reward margin ({self.consecutive_negative}/{self.patience} consecutive)")
                    if self.consecutive_negative >= self.patience:
                        print(f"\n  EARLY STOPPING: Reward margin negative for {self.patience} consecutive steps.")
                        print("  Saving checkpoint before halting...")
                        control.should_training_stop = True
                        control.should_save = True
                else:
                    self.consecutive_negative = 0

    class PeriodicVolumeCommitCallback(TrainerCallback):
        """Commit volume every N steps (independent of save_steps)."""
        def __init__(self, every_n_steps: int = 50):
            self.every_n_steps = every_n_steps

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step > 0 and state.global_step % self.every_n_steps == 0:
                volume.commit()
                print(f"  [periodic volume commit at step {state.global_step}]")

    callbacks = [
        VolumeCommitCallback(),
        NaNDetectionCallback(),
        PeriodicVolumeCommitCallback(every_n_steps=50),
    ]

    # Only add reward margin monitor for SimPO (not KTO — different log keys)
    if not use_kto:
        callbacks.append(RewardMarginMonitor(patience=NEGATIVE_MARGIN_PATIENCE))

    # -- Determine output paths ------------------------------------------------
    if use_kto:
        output_dir = OUTPUT_DIR.replace("-simpo", "-kto")
        final_adapter_dir = ADAPTER_DIR.replace("-simpo", "-kto")
    else:
        output_dir = OUTPUT_DIR
        final_adapter_dir = ADAPTER_DIR

    # -- Build trainer ---------------------------------------------------------
    if use_kto:
        from trl import KTOConfig, KTOTrainer

        training_args = {
            "output_dir": output_dir,
            "beta": KTO_BETA,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": BATCH_SIZE,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "lr_scheduler_type": LR_SCHEDULER,
            "warmup_ratio": WARMUP_RATIO,
            "bf16": True,
            "max_grad_norm": MAX_GRAD_NORM,
            "logging_steps": 2,
            "eval_strategy": "steps",
            "eval_steps": 30,
            "save_strategy": "steps",
            "save_steps": 30,
            "save_total_limit": 4,
            "report_to": "none",
            "seed": SEED,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "max_length": MAX_LENGTH,
            "max_completion_length": MAX_LENGTH // 2,
        }

        if limit_steps > 0:
            training_args["max_steps"] = limit_steps
            training_args["save_steps"] = max(5, limit_steps // 3)
            training_args["eval_steps"] = max(5, limit_steps // 3)
            print(f"\n  [LIMIT MODE: {limit_steps} steps]")

        config = KTOConfig(**training_args)

        trainer = KTOTrainer(
            model=model,
            args=config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

    else:
        from trl import CPOConfig, CPOTrainer

        training_args = {
            "output_dir": output_dir,
            "loss_type": "simpo",
            "cpo_alpha": 0.0,         # no BC regularizer for SimPO
            "beta": beta,
            "simpo_gamma": gamma,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": BATCH_SIZE,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "learning_rate": LEARNING_RATE,
            "lr_scheduler_type": LR_SCHEDULER,
            "warmup_ratio": WARMUP_RATIO,
            "bf16": True,
            "max_grad_norm": MAX_GRAD_NORM,
            "logging_steps": 2,
            "eval_strategy": "steps",
            "eval_steps": 30,
            "save_strategy": "steps",
            "save_steps": 30,
            "save_total_limit": 4,
            "report_to": "none",
            "seed": SEED,
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "max_length": MAX_LENGTH,
        }

        if limit_steps > 0:
            training_args["max_steps"] = limit_steps
            training_args["save_steps"] = max(5, limit_steps // 3)
            training_args["eval_steps"] = max(5, limit_steps // 3)
            print(f"\n  [LIMIT MODE: {limit_steps} steps]")

        config = CPOConfig(**training_args)

        trainer = CPOTrainer(
            model=model,
            args=config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

    # -- Train -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{mode} TRAINING: {len(train_ds)} {'examples' if use_kto else 'pairs'}, {epochs} epoch(s)")
    if use_kto:
        print(f"  kto_beta={KTO_BETA}, LR={LEARNING_RATE}")
    else:
        print(f"  beta={beta}, gamma={gamma}, LR={LEARNING_RATE}")
    print(f"  Reference-free: no dual adapter needed")
    print("=" * 60)

    trainer.train()
    print(f"{mode} training complete.")

    # -- Print reward margin summary (SimPO only) ------------------------------
    if not use_kto:
        for cb in callbacks:
            if isinstance(cb, RewardMarginMonitor) and cb.step_rewards:
                margins = [r["margin"] for r in cb.step_rewards]
                chosen_vals = [r["chosen"] for r in cb.step_rewards if r["chosen"] is not None]
                rejected_vals = [r["rejected"] for r in cb.step_rewards if r["rejected"] is not None]

                print(f"\n  Reward Margin Summary:")
                print(f"    Final margin:      {margins[-1]:.4f}")
                print(f"    Mean margin:       {sum(margins) / len(margins):.4f}")
                print(f"    Min margin:        {min(margins):.4f}")
                print(f"    Max margin:        {max(margins):.4f}")
                if chosen_vals:
                    print(f"    Mean chosen LP:    {sum(chosen_vals) / len(chosen_vals):.4f}")
                if rejected_vals:
                    print(f"    Mean rejected LP:  {sum(rejected_vals) / len(rejected_vals):.4f}")

    # -- Save aligned adapter --------------------------------------------------
    print(f"\nSaving {mode} adapter -> {final_adapter_dir}")
    os.makedirs(final_adapter_dir, exist_ok=True)

    model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    volume.commit()

    print("\n" + "=" * 60)
    print(f"V3 {mode} COMPLETE!")
    print(f"  Adapter:     {final_adapter_dir}")
    print(f"  Checkpoints: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Run eval:      modal run --detach training/eval_harness_v3.py --adapter {final_adapter_dir}")
    print(f"  2. Merge + push:  modal run training/push_to_hf.py --adapter-dir {final_adapter_dir}")
    if not use_kto:
        print(f"  3. If converge failed, try KTO:  modal run --detach training/simpo_v3.py --use-kto")
    print("=" * 60)


# --- Local entrypoint ---------------------------------------------------------

@app.local_entrypoint()
def main(
    adapter_path: str = SFT_ADAPTER,
    pairs_path: str = f"{DATASET_DIR}/simpo_pairs.jsonl",
    epochs: int = DEFAULT_EPOCHS,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    limit_steps: int = 0,
    use_kto: bool = False,
):
    """
    Run V3 SimPO/KTO alignment on Modal.

    Flags:
      --adapter-path PATH   SFT adapter to align (default: V3 SFT adapter)
      --pairs-path PATH     Preference pairs JSONL on Modal volume
      --epochs N            Number of epochs (default: 3)
      --beta FLOAT          SimPO beta parameter (default: 1.0)
      --gamma FLOAT         SimPO gamma margin (default: 0.3)
      --limit-steps N       Max training steps (for testing, 0=unlimited)
      --use-kto             Use KTO instead of SimPO (fallback)

    Examples:
      modal run --detach training/simpo_v3.py
      modal run --detach training/simpo_v3.py --beta 0.5 --gamma 0.5
      modal run --detach training/simpo_v3.py --limit-steps 20
      modal run --detach training/simpo_v3.py --use-kto
    """
    train_simpo.remote(
        adapter_path=adapter_path,
        pairs_path=pairs_path,
        epochs=epochs,
        beta=beta,
        gamma=gamma,
        limit_steps=limit_steps,
        use_kto=use_kto,
    )
