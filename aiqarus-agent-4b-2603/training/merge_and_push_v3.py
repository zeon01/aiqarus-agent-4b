"""
merge_and_push_v3.py — Merge LoRA, convert to GGUF, push to HuggingFace
========================================================================
V3 merge/push script supporting both local execution and Modal (data center
upload speeds). Updated for Qwen3.5-4B base model and date-based naming.

Steps:
  1. Load base model (Qwen3.5-4B) and LoRA adapter
  2. Merge adapter into base model using PEFT
  3. Save merged model in bf16 (and optionally fp32)
  4. Convert to GGUF using llama.cpp (Q4_K_M, Q5_K_M, Q8_0)
  5. Push to HuggingFace — model weights + GGUF files + model card

Local usage:
  python3 training/merge_and_push_v3.py --adapter ./adapter/aiqarus-agent-4b-v3-simpo
  python3 training/merge_and_push_v3.py --adapter ./adapter/aiqarus-agent-4b-v3-simpo --skip-push
  python3 training/merge_and_push_v3.py --adapter ./adapter/aiqarus-agent-4b-v3-simpo --skip-gguf
  python3 training/merge_and_push_v3.py --dry-run

Modal usage:
  modal run training/merge_and_push_v3.py
  modal run training/merge_and_push_v3.py --dry-run
  modal run training/merge_and_push_v3.py --skip-gguf

Prerequisites:
  pip install torch transformers peft huggingface_hub
  huggingface-cli login   (or set HF_TOKEN in .env)
  # For GGUF: clone llama.cpp and build llama-quantize
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# ─── Defaults ────────────────────────────────────────────────────────────────

BASE_MODEL       = "Qwen/Qwen3.5-4B"
DEFAULT_ADAPTER  = "adapter/aiqarus-agent-4b-v3-simpo"
DEFAULT_OUTPUT   = "merged/aiqarus-agent-4b-2603"
HF_REPO          = "zeon01/aiqarus-agent-4b-2603"
MODEL_NAME       = "aiqarus-agent-4b-2603"
DEFAULT_GGUF_TYPES = "Q4_K_M,Q5_K_M,Q8_0"

# Modal paths (used when running on Modal)
VOLUME_NAME      = "aiqarus-data"
MODAL_ADAPTER    = "/data/adapter/aiqarus-agent-4b-v3-simpo"
MODAL_OUTPUT     = "/data/merged/aiqarus-agent-4b-2603"

# ─── Model card ──────────────────────────────────────────────────────────────

MODEL_CARD = textwrap.dedent("""\
    ---
    language: en
    license: apache-2.0
    base_model: Qwen/Qwen3.5-4B
    tags:
      - tool-calling
      - enterprise
      - agent
      - function-calling
      - qwen3.5
    datasets:
      - custom
    pipeline_tag: text-generation
    ---

    # Aiqarus Agent 4B (March 2026)

    Enterprise AI agent model fine-tuned from Qwen3.5-4B for tool-calling, routing, escalation, and multi-step planning.

    ## Key Capabilities
    - 14-category enterprise agent behavior
    - 200+ enterprise tool schema generalization
    - Tool restraint (knows when NOT to call tools)
    - Multi-turn conversation with tool simulation
    - Prompt injection resistance

    ## Training
    - Base: Qwen/Qwen3.5-4B (dense, 4B params, Apache 2.0, Gated DeltaNet attention)
    - Method: QLoRA (rank 32) + SimPO alignment
    - Data: ~70K SFT samples + 1,000+ on-policy preference pairs
    - GPU: B200 on Modal

    ## Eval Results
    [To be filled with actual V3 results]

    ## Usage

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "zeon01/aiqarus-agent-4b-2603"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)

    messages = [
        {"role": "system", "content": "You are an enterprise AI agent with access to tools."},
        {"role": "user", "content": "Look up the account status for Acme Corp."},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=1024)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    ```

    ## GGUF Variants
    - `aiqarus-agent-4b-2603-Q4_K_M.gguf` — 4-bit quantized (~2.5 GB), best for constrained devices
    - `aiqarus-agent-4b-2603-Q5_K_M.gguf` — 5-bit quantized (~3 GB), good balance
    - `aiqarus-agent-4b-2603-Q8_0.gguf` — 8-bit quantized (~4.5 GB), near-lossless

    ## License
    Apache 2.0 (same as base model)
""")


# ─── HF token (local mode) ──────────────────────────────────────────────────

def _load_hf_token() -> str | None:
    """Load HF token from env or dotenv."""
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
    except ImportError:
        pass
    return os.environ.get("HF_TOKEN")


# ─── Auto-detect adapter ────────────────────────────────────────────────────

def _find_adapter(adapter_arg: str | None) -> str:
    """Resolve adapter path. Tries explicit arg, then common locations."""
    candidates = []
    if adapter_arg:
        candidates.append(adapter_arg)
    else:
        # Common local paths
        candidates.extend([
            DEFAULT_ADAPTER,
            "aiqarus-agent-4b-v3-simpo",
            "adapter/aiqarus-agent-4b-v3",
            "aiqarus-agent-4b-v3",
        ])

    for path in candidates:
        p = Path(path)
        if p.exists() and (p / "adapter_config.json").exists():
            return str(p)

    raise FileNotFoundError(
        f"LoRA adapter not found. Searched: {candidates}\n"
        "Download it first:\n"
        "  modal volume get aiqarus-data adapter/aiqarus-agent-4b-v3-simpo .\n"
        "Or specify explicitly:\n"
        "  python3 training/merge_and_push_v3.py --adapter /path/to/adapter"
    )


# ─── Core merge logic ───────────────────────────────────────────────────────

def merge_adapter(
    base_model: str,
    adapter_dir: str,
    output_dir: str,
    save_fp32: bool = False,
    dry_run: bool = False,
) -> str:
    """Load base model + adapter, merge, save bf16 (and optionally fp32).

    Returns path to bf16 output directory.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bf16_dir = os.path.join(output_dir, "bf16")
    fp32_dir = os.path.join(output_dir, "fp32")

    # ── 1. Load tokenizer ────────────────────────────────────────────────
    print(f"\n[1/4] Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # ── 2. Load base model ───────────────────────────────────────────────
    # Load in fp32 first for clean merge, then convert to bf16
    load_dtype = torch.float32 if save_fp32 else torch.bfloat16
    dtype_label = "fp32" if save_fp32 else "bf16"
    print(f"\n[2/4] Loading base model ({dtype_label}, CPU): {base_model}")
    if save_fp32:
        print("  Loading in fp32 (will save both fp32 and bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=load_dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    # ── 3. Load LoRA adapter and merge ───────────────────────────────────
    print(f"\n[3/4] Loading LoRA adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(model, adapter_dir)

    print("  Merging adapter into base model...")
    model = model.merge_and_unload()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Merge complete. Total params: {param_count:,}")

    if dry_run:
        print("\n  --dry-run: skipping save.")
        return bf16_dir

    # ── 4. Save ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving merged model...")

    if save_fp32:
        print(f"  Saving fp32 -> {fp32_dir}")
        os.makedirs(fp32_dir, exist_ok=True)
        model.save_pretrained(fp32_dir, safe_serialization=True)
        tokenizer.save_pretrained(fp32_dir)

        # Convert to bf16 for primary output
        print(f"  Converting to bf16...")
        model = model.to(torch.bfloat16)

    print(f"  Saving bf16 -> {bf16_dir}")
    os.makedirs(bf16_dir, exist_ok=True)
    model.save_pretrained(bf16_dir, safe_serialization=True)
    tokenizer.save_pretrained(bf16_dir)

    # Write model card
    model_card_path = os.path.join(bf16_dir, "README.md")
    with open(model_card_path, "w") as f:
        f.write(MODEL_CARD)
    print(f"  Model card written to {model_card_path}")

    return bf16_dir


# ─── GGUF conversion ────────────────────────────────────────────────────────

def convert_to_gguf(
    merged_dir: str,
    output_dir: str,
    gguf_types: list[str],
    dry_run: bool = False,
) -> list[str]:
    """Convert merged HF model to GGUF format using llama.cpp.

    Requires llama.cpp to be cloned and built locally:
      git clone https://github.com/ggerganov/llama.cpp
      cd llama.cpp && cmake -B build && cmake --build build

    Returns list of GGUF file paths created.
    """
    # Find llama.cpp
    llama_cpp_dir = _find_llama_cpp()
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    quantize_bin = _find_quantize_binary(llama_cpp_dir)

    if not os.path.isfile(convert_script):
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {convert_script}\n"
            "Clone llama.cpp:\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  pip install -r llama.cpp/requirements.txt"
        )

    if not quantize_bin:
        raise FileNotFoundError(
            "llama-quantize binary not found. Build llama.cpp:\n"
            "  cd llama.cpp && cmake -B build && cmake --build build"
        )

    gguf_dir = os.path.join(output_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

    f16_gguf = os.path.join(gguf_dir, f"{MODEL_NAME}-f16.gguf")
    created_files = []

    if dry_run:
        print(f"\n  --dry-run: would convert {merged_dir} to GGUF")
        print(f"    f16 intermediate: {f16_gguf}")
        for qtype in gguf_types:
            print(f"    quantize: {qtype} -> {MODEL_NAME}-{qtype}.gguf")
        return []

    # Step 1: Convert HF -> GGUF f16
    print(f"\n[GGUF] Converting to f16 GGUF...")
    print(f"  Source: {merged_dir}")
    print(f"  Output: {f16_gguf}")

    result = subprocess.run(
        [
            sys.executable, convert_script,
            merged_dir,
            "--outfile", f16_gguf,
            "--outtype", "f16",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: convert_hf_to_gguf.py failed:\n{result.stderr}")
        raise RuntimeError("GGUF conversion failed")
    print(f"  f16 GGUF created: {f16_gguf}")

    # Step 2: Quantize to each requested type
    for qtype in gguf_types:
        quantized_path = os.path.join(gguf_dir, f"{MODEL_NAME}-{qtype}.gguf")
        print(f"\n[GGUF] Quantizing to {qtype}...")
        print(f"  Output: {quantized_path}")

        result = subprocess.run(
            [
                quantize_bin,
                f16_gguf,
                quantized_path,
                qtype,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  ERROR: quantization to {qtype} failed:\n{result.stderr}")
            continue

        size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"  {qtype} created: {size_mb:.0f} MB")
        created_files.append(quantized_path)

    # Optionally remove intermediate f16 GGUF to save disk space
    if created_files and os.path.exists(f16_gguf):
        f16_size_mb = os.path.getsize(f16_gguf) / (1024 * 1024)
        print(f"\n[GGUF] Removing intermediate f16 GGUF ({f16_size_mb:.0f} MB)")
        os.remove(f16_gguf)

    return created_files


def _find_llama_cpp() -> str:
    """Search for llama.cpp directory in common locations."""
    candidates = [
        "llama.cpp",
        "../llama.cpp",
        os.path.expanduser("~/llama.cpp"),
        "/opt/llama.cpp",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "llama.cpp directory not found. Searched:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + "\nClone it:\n  git clone https://github.com/ggerganov/llama.cpp"
    )


def _find_quantize_binary(llama_cpp_dir: str) -> str | None:
    """Find the llama-quantize binary (varies by build system)."""
    candidates = [
        os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize"),
        os.path.join(llama_cpp_dir, "build", "llama-quantize"),
        os.path.join(llama_cpp_dir, "llama-quantize"),
        os.path.join(llama_cpp_dir, "quantize"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


# ─── HuggingFace push ───────────────────────────────────────────────────────

def push_to_hf(
    bf16_dir: str,
    hf_repo: str,
    gguf_files: list[str] | None = None,
    hf_token: str | None = None,
    dry_run: bool = False,
):
    """Push merged model + GGUF files to HuggingFace Hub."""
    from huggingface_hub import HfApi

    if dry_run:
        print(f"\n  --dry-run: would push to {hf_repo}")
        if gguf_files:
            for f in gguf_files:
                print(f"    GGUF: {os.path.basename(f)}")
        return

    if not hf_token:
        hf_token = _load_hf_token()

    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    print(f"\n[Push] Ensuring repo exists: {hf_repo}")
    api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="model")

    # Push the bf16 model directory (includes model card)
    print(f"[Push] Uploading bf16 model from {bf16_dir}...")
    api.upload_folder(
        folder_path=bf16_dir,
        repo_id=hf_repo,
        token=hf_token,
    )
    print(f"  bf16 model uploaded.")

    # Push GGUF files individually
    if gguf_files:
        print(f"[Push] Uploading {len(gguf_files)} GGUF file(s)...")
        for gguf_path in gguf_files:
            filename = os.path.basename(gguf_path)
            print(f"  Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=gguf_path,
                path_in_repo=filename,
                repo_id=hf_repo,
                token=hf_token,
            )
        print(f"  GGUF files uploaded.")

    print(f"\n{'=' * 60}")
    print(f"Done! Model live at:")
    print(f"  https://huggingface.co/{hf_repo}")
    print(f"{'=' * 60}")


# ─── Local entrypoint ───────────────────────────────────────────────────────

def main_local():
    """Run merge, GGUF conversion, and push locally."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into Qwen3.5-4B, convert to GGUF, push to HuggingFace"
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help=f"Path to LoRA adapter directory (default: auto-detect, tries {DEFAULT_ADAPTER})"
    )
    parser.add_argument(
        "--base-model", type=str, default=BASE_MODEL,
        help=f"Base model name or path (default: {BASE_MODEL})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT,
        help=f"Output directory for merged model (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--hf-repo", type=str, default=HF_REPO,
        help=f"HuggingFace repo to push to (default: {HF_REPO})"
    )
    parser.add_argument(
        "--skip-push", action="store_true",
        help="Merge and save locally, don't push to HuggingFace"
    )
    parser.add_argument(
        "--skip-gguf", action="store_true",
        help="Skip GGUF conversion"
    )
    parser.add_argument(
        "--fp32", action="store_true",
        help="Save fp32 copy in addition to bf16"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without doing anything"
    )
    parser.add_argument(
        "--gguf-types", type=str, default=DEFAULT_GGUF_TYPES,
        help=f"Comma-separated GGUF quant types (default: {DEFAULT_GGUF_TYPES})"
    )
    args = parser.parse_args()

    gguf_types = [t.strip() for t in args.gguf_types.split(",") if t.strip()]

    # Print plan
    adapter_dir = args.adapter or DEFAULT_ADAPTER
    print(f"\n{'=' * 60}")
    print(f"  Aiqarus V3 — Merge & Push")
    print(f"{'=' * 60}")
    print(f"  Base model:  {args.base_model}")
    print(f"  Adapter:     {adapter_dir}")
    print(f"  Output:      {args.output_dir}")
    print(f"  HF repo:     {args.hf_repo}")
    print(f"  Save fp32:   {args.fp32}")
    print(f"  GGUF:        {'skip' if args.skip_gguf else ', '.join(gguf_types)}")
    print(f"  Push to HF:  {'no' if args.skip_push else 'yes'}")
    print(f"  Dry run:     {args.dry_run}")
    print(f"{'=' * 60}")

    if args.dry_run:
        # Validate adapter exists even in dry-run
        try:
            adapter_dir = _find_adapter(args.adapter)
            print(f"\n  Adapter found: {adapter_dir}")
        except FileNotFoundError as e:
            print(f"\n  Adapter check: {e}")
        print("\n  Dry run complete. No files written.")
        return

    # Step 1: Find adapter
    adapter_dir = _find_adapter(args.adapter)
    print(f"\n  Using adapter: {adapter_dir}")

    # Step 2: Merge
    bf16_dir = merge_adapter(
        base_model=args.base_model,
        adapter_dir=adapter_dir,
        output_dir=args.output_dir,
        save_fp32=args.fp32,
        dry_run=False,
    )

    # Step 3: GGUF conversion
    gguf_files = []
    if not args.skip_gguf:
        print(f"\n{'─' * 60}")
        print(f"  GGUF Conversion")
        print(f"{'─' * 60}")
        try:
            gguf_files = convert_to_gguf(
                merged_dir=bf16_dir,
                output_dir=args.output_dir,
                gguf_types=gguf_types,
                dry_run=False,
            )
        except FileNotFoundError as e:
            print(f"\n  WARNING: Skipping GGUF conversion — {e}")
            print("  You can convert later with --skip-push, or use --skip-gguf to skip.")
    else:
        print("\n  GGUF conversion skipped (--skip-gguf)")

    # Step 4: Push
    if not args.skip_push:
        print(f"\n{'─' * 60}")
        print(f"  Push to HuggingFace")
        print(f"{'─' * 60}")
        push_to_hf(
            bf16_dir=bf16_dir,
            hf_repo=args.hf_repo,
            gguf_files=gguf_files if gguf_files else None,
        )
    else:
        print(f"\n  Push skipped (--skip-push)")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    print(f"  bf16 model:   {bf16_dir}/")
    if args.fp32:
        fp32_dir = os.path.join(args.output_dir, "fp32")
        print(f"  fp32 model:   {fp32_dir}/")
    if gguf_files:
        gguf_dir = os.path.join(args.output_dir, "gguf")
        print(f"  GGUF files:   {gguf_dir}/")
        for f in gguf_files:
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"    {os.path.basename(f)} ({size_mb:.0f} MB)")
    if not args.skip_push:
        print(f"  HF repo:      https://huggingface.co/{args.hf_repo}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════════
# Modal mode — run on data center for fast upload
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import modal

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

    app    = modal.App("aiqarus-v3-push")
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

    # Load HF secret for push
    secrets = []
    try:
        secrets.append(modal.Secret.from_name("huggingface-secret"))
    except Exception:
        pass

    @app.function(
        image=image,
        gpu="T4",                     # T4 is enough for CPU merge + push
        volumes={"/data": volume},
        timeout=3 * 3600,             # 3 hrs — merge + upload can be slow
        secrets=secrets,
        memory=32768,                 # 32 GB RAM for fp32 merge
    )
    def modal_push(
        dry_run: bool = False,
        skip_gguf: bool = True,       # GGUF not available on Modal by default
        adapter_dir: str = MODAL_ADAPTER,
        base_model: str = BASE_MODEL,
        hf_repo: str = HF_REPO,
    ):
        """Merge LoRA adapter and push to HuggingFace from Modal."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token and not dry_run:
            raise RuntimeError(
                "HF_TOKEN not found. Create a Modal secret:\n"
                "  modal secret create huggingface-secret HF_TOKEN=hf_..."
            )

        output_dir = MODAL_OUTPUT
        bf16_dir   = os.path.join(output_dir, "bf16")

        # ── 1. Load tokenizer ────────────────────────────────────────────
        print(f"\n[1/4] Loading tokenizer: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

        # ── 2. Load base model in bf16 ───────────────────────────────────
        print(f"\n[2/4] Loading base model (bf16, CPU): {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )

        # ── 3. Load LoRA adapter and merge ───────────────────────────────
        if not Path(adapter_dir).exists():
            # List available adapters for debugging
            adapter_parent = Path(adapter_dir).parent
            available = list(adapter_parent.glob("*")) if adapter_parent.exists() else []
            raise FileNotFoundError(
                f"Adapter not found at '{adapter_dir}'\n"
                f"Available in {adapter_parent}: {[p.name for p in available]}\n"
                "Check volume contents with:\n"
                "  modal volume ls aiqarus-data adapter/"
            )

        print(f"\n[3/4] Loading LoRA adapter: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir)

        print("  Merging adapter into base model...")
        model = model.merge_and_unload()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Merge complete. Total params: {param_count:,}")

        if dry_run:
            print(f"\n  --dry-run: merge verified. Params: {param_count:,}")
            print("  Skipping save and push.")
            return

        # ── 4. Save bf16 on volume ───────────────────────────────────────
        print(f"\n[4/4] Saving bf16 -> {bf16_dir}")
        os.makedirs(bf16_dir, exist_ok=True)
        model.save_pretrained(bf16_dir, safe_serialization=True)
        tokenizer.save_pretrained(bf16_dir)

        # Write model card
        model_card_path = os.path.join(bf16_dir, "README.md")
        with open(model_card_path, "w") as f:
            f.write(MODEL_CARD)
        print(f"  Model card written.")

        # Commit to volume so files persist
        volume.commit()
        print(f"  Saved to volume.")

        # ── 5. Push to HuggingFace ───────────────────────────────────────
        from huggingface_hub import HfApi

        print(f"\n[Push] Pushing to HuggingFace: {hf_repo}")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=hf_repo, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=bf16_dir,
            repo_id=hf_repo,
            token=hf_token,
        )

        print(f"\n{'=' * 60}")
        print(f"Done! Model live at:")
        print(f"  https://huggingface.co/{hf_repo}")
        print(f"{'=' * 60}")

    @app.local_entrypoint()
    def modal_main(
        dry_run: bool = False,
        skip_gguf: bool = True,
    ):
        """
        Merge LoRA adapter and push to HuggingFace from Modal.

        Flags:
          --dry-run    Merge only, verify it works (don't push)
          --skip-gguf  Skip GGUF conversion (default: True on Modal)
        """
        modal_push.remote(dry_run=dry_run, skip_gguf=skip_gguf)

except ImportError:
    # Modal not installed — local-only mode
    pass


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # When run directly with python, use local mode.
    # When run with `modal run`, Modal handles the entrypoint via @app.local_entrypoint.
    main_local()
