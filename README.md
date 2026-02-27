# aiqarus-agent-4b

Fine-tuned [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) for enterprise AI agent tasks — tool-calling, multi-step planning, risk escalation, confidence calibration, and multi-agent handoff.

**Model:** [huggingface.co/zeon01/aiqarus-agent-4b](https://huggingface.co/zeon01/aiqarus-agent-4b)

## What's Here

| Script | Purpose |
|---|---|
| `prepare_dataset.py` | Filter, merge, and format 51K training samples from multiple sources into Qwen3 chat format |
| `training/train.py` | QLoRA fine-tuning on Modal.com (2-stage curriculum, A10G GPU) |
| `training/test_harness.py` | Evaluation harness — 230 test cases scoring action accuracy, tool compliance, adversarial robustness |
| `training/llm_judge.py` | LLM-as-judge — re-scores test results using gpt-5-codex-mini for reasoning/response quality |
| `training/merge_and_push.py` | Merge LoRA adapter into base model locally |
| `training/push_to_hf.py` | Merge + push to HuggingFace from Modal (data center upload speeds) |

## Training

- **Base model:** Qwen/Qwen3-4B-Instruct-2507
- **Method:** QLoRA (4-bit NF4, rank=32, alpha=64)
- **Dataset:** 51,642 samples across 3 layers (foundation tool-calling, reasoning traces, enterprise agent scenarios)
- **Curriculum:** Stage 1 (Layer 1 only, 1 epoch) → Stage 2 (all layers, 2 epochs, Layer 3 upsampled 3x)
- **Hardware:** NVIDIA A10G on Modal.com (~17 hours)
- **Final loss:** 0.3761 | **Token accuracy:** ~90.9%

## Round 1 Eval Results (230 test cases)

Evaluated with both keyword/regex heuristics and LLM-as-judge (gpt-5-codex-mini).

| Metric | Heuristic | LLM Judge |
|---|---|---|
| Overall action accuracy | 53.0% | **38.7%** |
| Avg reasoning quality | N/A | **2.1/5** |
| Avg response quality | N/A | **1.9/5** |

**By category (LLM judge, 25 cases each):**

| Category | Heuristic | LLM Judge |
|---|---|---|
| cost_aware_routing | 88% | **84%** |
| audit_trail | 80% | **52%** |
| confidence_calibration | 28% | **40%** |
| adversarial | 46.7% | **40%** |
| tool_routing | 40% | **40%** |
| multi_agent_handoff | 100% | **36%** |
| multi_step | 52% | **36%** |
| graceful_failure | 36% | **16%** |
| risk_escalation | 8% | **4%** |

**Key finding:** Token accuracy (90.9%) ≠ decision quality (38.7%). The heuristic eval (53%) was also misleading — keyword matching inflated scores, especially for multi_agent_handoff (100% → 36%). The model formats tool calls perfectly but can't decide when to use them and barely reasons (2.1/5). Round 2 addresses this with reasoning-quality training, negative examples, adversarial data, DPO alignment, and targeted data for weak categories.

## Try It

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "zeon01/aiqarus-agent-4b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("zeon01/aiqarus-agent-4b", trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are an enterprise AI agent with access to tools."},
    {"role": "user", "content": "Find all customers with contracts expiring this quarter."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.6, do_sample=True)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
```

## License

Apache 2.0
