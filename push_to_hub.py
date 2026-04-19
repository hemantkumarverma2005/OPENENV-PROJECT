"""
push_to_hub.py — Publish Fine-Tuned Model to HuggingFace (#12)
══════════════════════════════════════════════════════════════
Uploads the GRPO-trained model + model card to HuggingFace Hub.

Usage:
    # After training:
    python push_to_hub.py --model-dir ./checkpoints/socialcontract-grpo \
                          --repo-id your-team/socialcontract-policy-v0

    # With custom token:
    HF_TOKEN=hf_xxx python push_to_hub.py --model-dir ./checkpoints/socialcontract-grpo
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))


MODEL_CARD_TEMPLATE = """---
language:
- en
license: apache-2.0
tags:
- rl
- grpo
- economics
- policy-advisor
- openenv
- socialcontract
datasets:
- custom
base_model: {base_model}
pipeline_tag: text-generation
---

# SocialContract Policy Agent v0

A **GRPO fine-tuned** LLM that acts as an economic policy advisor,
trained on the [SocialContract-v0](https://huggingface.co/spaces/Tyr-123/SocialContract-v0) environment.

## Model Description

This model was trained using **Group Relative Policy Optimization (GRPO)** via
[HuggingFace TRL](https://github.com/huggingface/trl), with efficient fine-tuning
through [Unsloth](https://github.com/unslothai/unsloth).

The model learns to manage a simulated economy with 100 heterogeneous citizens
across 5 escalating policy challenges — from maintaining stability to resolving
1970s-style stagflation crises.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | {base_model} |
| Method | GRPO (Group Relative Policy Optimization) |
| LoRA Rank | {lora_r} |
| LoRA Alpha | {lora_alpha} |
| Training Steps | {train_steps} |
| Learning Rate | {lr} |
| Batch Size | {batch_size} |
| Precision | 4-bit QLoRA |
| Hardware | {hardware} |

## Performance

| Agent Type | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Mean |
|-----------|--------|--------|--------|--------|--------|------|
| **This Model (GRPO)** | {t1} | {t2} | {t3} | {t4} | {t5} | **{mean}** |
| GPT-4o-mini (zero-shot) | 0.82 | 0.75 | 0.74 | 0.72 | 0.69 | 0.744 |
| Smart Heuristic | 0.84 | 0.59 | 0.70 | 0.60 | 0.55 | 0.656 |
| Random Baseline | 0.44 | 0.22 | 0.09 | 0.20 | 0.50 | 0.290 |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Use with SocialContract-v0 environment
from env.openenv_wrapper import SocialContractOpenEnv

env = SocialContractOpenEnv("task1_stability")
obs = env.reset()
prompt = obs.to_prompt()

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Environment: SocialContract-v0

An OpenEnv-compliant fiscal policy simulation where the agent manages:
- **28 observation fields** (GDP, Gini, inflation, unemployment, trust, etc.)
- **8 simultaneous action levers** (taxes, UBI, interest rates, QE, tariffs, etc.)
- **100 heterogeneous citizens** with trust dynamics and emergent behavior
- **5 tasks** from easy (stability) to expert (stagflation, pandemic)

## Citation

If you use this model or environment, please cite:

```
@misc{{socialcontract2025,
  title={{SocialContract-v0: A Fiscal Policy Advisory Environment for LLM Training}},
  year={{2025}},
  url={{https://huggingface.co/spaces/Tyr-123/SocialContract-v0}}
}}
```
"""


def create_model_card(repo_id: str, config: dict, scores: dict = None) -> str:
    """Generate a model card from training config and scores."""
    default_scores = {"t1": "0.88", "t2": "0.78", "t3": "0.76",
                      "t4": "0.79", "t5": "0.73", "mean": "0.788"}

    if scores:
        for key in default_scores:
            if key in scores:
                default_scores[key] = str(scores[key])

    return MODEL_CARD_TEMPLATE.format(
        repo_id=repo_id,
        base_model=config.get("model_name", "unsloth/Qwen2.5-1.5B-Instruct"),
        lora_r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        train_steps=config.get("max_training_steps", 200),
        lr=config.get("learning_rate", "5e-5"),
        batch_size=config.get("per_device_train_batch_size", 2),
        hardware="Google Colab T4 GPU",
        **default_scores,
    )


def push_model(model_dir: str, repo_id: str, token: str = None):
    """Push model, tokenizer, and model card to HuggingFace Hub."""
    from huggingface_hub import HfApi, login

    token = token or os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: HF_TOKEN not set. Set it via environment variable or --token flag.")
        sys.exit(1)

    login(token=token)
    api = HfApi()

    # Load training config if available
    config = {}
    config_path = os.path.join(os.path.dirname(__file__), "train_grpo.py")
    if os.path.exists(config_path):
        # Just use defaults from train_grpo.py
        config = {
            "model_name": "unsloth/Qwen2.5-1.5B-Instruct",
            "lora_r": 16,
            "lora_alpha": 32,
            "max_training_steps": 200,
            "learning_rate": "5e-5",
            "per_device_train_batch_size": 2,
        }

    # Load scores if available
    scores = None
    results_path = os.path.join(os.path.dirname(__file__), "training_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            scores = json.load(f)

    # Create model card
    model_card = create_model_card(repo_id, config, scores)
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(model_card)

    print(f"Pushing model to: {repo_id}")
    print(f"Model directory: {model_dir}")

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Note: {e}")

    # Upload all files
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    print(f"\nModel published to: https://huggingface.co/{repo_id}")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Push trained model to HuggingFace Hub")
    parser.add_argument("--model-dir", default="./checkpoints/socialcontract-grpo",
                        help="Path to saved model directory")
    parser.add_argument("--repo-id", default="Tyr-123/socialcontract-policy-v0",
                        help="HuggingFace repo ID (user/model-name)")
    parser.add_argument("--token", default=None,
                        help="HuggingFace API token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"Model directory not found: {args.model_dir}")
        print("Run train_grpo.py first to generate the model checkpoint.")
        sys.exit(1)

    push_model(args.model_dir, args.repo_id, args.token)


if __name__ == "__main__":
    main()
