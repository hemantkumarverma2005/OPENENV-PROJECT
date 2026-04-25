"""
train_grpo.py — Minimal GRPO Training Script for SocialContract-v0
═══════════════════════════════════════════════════════════════════
Uses HuggingFace TRL's GRPOTrainer + Unsloth for efficient fine-tuning.
Designed to run on Colab (T4 GPU) or with HuggingFace compute credits.

Usage (local):
    pip install unsloth trl datasets transformers accelerate
    python train_grpo.py

Usage (Colab — see train_colab.ipynb):
    # Cell 1: !pip install unsloth trl datasets
    # Cell 2: Run this script

Environment: SocialContract-v0 (OpenEnv-compliant)
Training objective: Maximize episode reward via GRPO (Group Relative Policy Optimization)
"""

import os
import sys
import json
import re
import time
import random
import numpy as np
import torch
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from env.openenv_wrapper import SocialContractOpenEnv
from env.pydantic_models import PolicyAction, EconomicObservation
from graders.graders import run_grader

# ── Configuration ────────────────────────────────────────────────────────────

TRAINING_CONFIG = {
    "model_name": "unsloth/Qwen2.5-7B-Instruct",   # 7B — strong reasoning, fits A100 80GB in 4-bit
    "max_seq_length": 3072,
    "lora_r": 32,                # Higher rank = more expressiveness
    "lora_alpha": 64,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 3,
    "learning_rate": 2e-5,       # Lower LR for larger model stability
    "num_generations": 6,        # GRPO: generate N completions per prompt (more = better signal)
    "max_new_tokens": 384,
    "training_tasks": ["task1_stability", "task2_recession", "task3_crisis",
                       "task4_stagflation", "task5_pandemic"],  # ALL 5 tasks
    "seeds": [42, 99, 7, 13, 55],
    "max_training_steps": 500,    # 2.5x more training for convergence
    "save_dir": "./checkpoints/socialcontract-grpo-7b",
    "log_file": "./training_log.jsonl",
}


# ── Prompt Generation ────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert economic policy advisor. You make precise, data-driven "
    "decisions to optimize economic outcomes across multiple competing objectives. "
    "You must respond ONLY with valid JSON matching the exact format requested. "
    "No markdown, no extra text — only the JSON object."
)


def generate_training_prompts(n_prompts: int = 200) -> list[dict]:
    """
    Generate diverse training prompts by running partial episodes.
    Each prompt captures a unique economic state the agent must respond to.
    """
    prompts = []
    tasks = TRAINING_CONFIG["training_tasks"]
    seeds = list(range(1, n_prompts // len(tasks) + 2))

    for task_id in tasks:
        for seed in seeds:
            if len(prompts) >= n_prompts:
                break

            env = SocialContractOpenEnv(task_id, seed=seed)
            obs = env.reset()

            # Run 0-N random warmup steps to get diverse states
            warmup = random.randint(0, env.task_cfg["max_steps"] // 2)
            rng = np.random.default_rng(seed)
            for _ in range(warmup):
                action = PolicyAction(
                    tax_delta=float(rng.uniform(-0.03, 0.03)),
                    ubi_delta=float(rng.uniform(-2.0, 2.0)),
                    public_good_delta=float(rng.uniform(-0.03, 0.03)),
                    interest_rate_delta=float(rng.uniform(-0.01, 0.01)),
                    stimulus_package=float(rng.uniform(0, 50)),
                    import_tariff_delta=float(rng.uniform(-0.02, 0.02)),
                    money_supply_delta=float(rng.uniform(-50, 50)),
                    minimum_wage_delta=float(rng.uniform(-0.5, 0.5)),
                    reasoning="warmup",
                )
                obs = env.step(action)
                if obs.done:
                    break

            if not env.is_done:
                prompt_text = obs.to_prompt()
                prompts.append({
                    "prompt": prompt_text,
                    "task_id": task_id,
                    "seed": seed,
                    "warmup_steps": warmup,
                    "step": obs.step,
                    # Store env state for reward evaluation
                    "_env_state": {
                        "task_id": task_id,
                        "seed": seed,
                        "warmup_steps": warmup,
                    }
                })

    random.shuffle(prompts)
    return prompts[:n_prompts]


# ── Action Parsing ───────────────────────────────────────────────────────────

def parse_action_from_text(text: str) -> Optional[PolicyAction]:
    """Parse a PolicyAction from raw LLM text output. Robust to formatting."""
    try:
        # Try to extract JSON from the text
        # Handle markdown code blocks
        if "```" in text:
            match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
            if match:
                text = match.group(1)

        # Find the JSON object
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        parsed = json.loads(text)

        return PolicyAction(
            tax_delta=float(np.clip(parsed.get("tax_delta", 0.0), -0.10, 0.10)),
            ubi_delta=float(np.clip(parsed.get("ubi_delta", 0.0), -10.0, 10.0)),
            public_good_delta=float(np.clip(parsed.get("public_good_delta", 0.0), -0.10, 0.10)),
            interest_rate_delta=float(np.clip(parsed.get("interest_rate_delta", 0.0), -0.03, 0.03)),
            stimulus_package=float(np.clip(parsed.get("stimulus_package", 0.0), 0.0, 500.0)),
            import_tariff_delta=float(np.clip(parsed.get("import_tariff_delta", 0.0), -0.05, 0.05)),
            money_supply_delta=float(np.clip(parsed.get("money_supply_delta", 0.0), -500.0, 500.0)),
            minimum_wage_delta=float(np.clip(parsed.get("minimum_wage_delta", 0.0), -2.0, 2.0)),
            reasoning=str(parsed.get("reasoning", ""))[:200],
            policy_speech=str(parsed.get("policy_speech", ""))[:150] if "policy_speech" in parsed else None,
        )
    except Exception:
        return None


# ── Reward Function for GRPO ─────────────────────────────────────────────────

def compute_environment_reward(
    completion: str,
    task_id: str,
    seed: int,
    warmup_steps: int,
) -> float:
    """
    Run a full episode from the given state, using the LLM's action for the
    current step and a simple heuristic for remaining steps.
    Returns the grader score [0.0, 1.0] as the reward signal.
    """
    try:
        action = parse_action_from_text(completion)
        if action is None:
            return 0.0  # Unparseable output gets zero reward

        env = SocialContractOpenEnv(task_id, seed=seed)
        obs = env.reset()

        # Replay warmup steps
        rng = np.random.default_rng(seed)
        for _ in range(warmup_steps):
            warmup_action = PolicyAction(
                tax_delta=float(rng.uniform(-0.03, 0.03)),
                ubi_delta=float(rng.uniform(-2.0, 2.0)),
                public_good_delta=float(rng.uniform(-0.03, 0.03)),
                interest_rate_delta=float(rng.uniform(-0.01, 0.01)),
                stimulus_package=float(rng.uniform(0, 50)),
                import_tariff_delta=float(rng.uniform(-0.02, 0.02)),
                money_supply_delta=float(rng.uniform(-50, 50)),
                minimum_wage_delta=float(rng.uniform(-0.5, 0.5)),
                reasoning="warmup",
            )
            obs = env.step(warmup_action)
            if obs.done:
                return 0.0

        # Apply the LLM's action
        obs = env.step(action)

        if obs.done:
            grade = run_grader(task_id, env._history)
            return grade["score"]

        # Run remaining steps with a simple heuristic (to evaluate full episode)
        while not env.is_done:
            # Use a mild follow-through policy (not random, to isolate LLM value)
            heuristic = PolicyAction(
                tax_delta=action.tax_delta * 0.5,
                ubi_delta=action.ubi_delta * 0.3,
                public_good_delta=action.public_good_delta * 0.5,
                interest_rate_delta=action.interest_rate_delta * 0.3,
                stimulus_package=0.0,
                import_tariff_delta=0.0,
                money_supply_delta=action.money_supply_delta * 0.2,
                minimum_wage_delta=action.minimum_wage_delta * 0.3,
                reasoning="heuristic follow-through",
            )
            obs = env.step(heuristic)

        grade = run_grader(task_id, env._history)
        return grade["score"]

    except Exception as e:
        print(f"Reward computation error: {e}")
        return 0.0


# ── Dataset Creation ─────────────────────────────────────────────────────────

def create_training_dataset(n_prompts: int = 100):
    """Create a HuggingFace Dataset of training prompts."""
    from datasets import Dataset

    raw_prompts = generate_training_prompts(n_prompts)

    dataset_entries = []
    for p in raw_prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": p["prompt"]},
        ]
        dataset_entries.append({
            "prompt": messages,
            "task_id": p["task_id"],
            "seed": p["seed"],
            "warmup_steps": p["warmup_steps"],
        })

    return Dataset.from_list(dataset_entries)


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, tasks=None, seeds=None):
    """Run full evaluation episodes and return per-task scores."""
    if tasks is None:
        tasks = ["task1_stability", "task2_recession", "task3_crisis",
                 "task4_stagflation", "task5_pandemic"]
    if seeds is None:
        seeds = [42]

    results = {}
    for task_id in tasks:
        task_scores = []
        for seed in seeds:
            env = SocialContractOpenEnv(task_id, seed=seed)
            obs = env.reset()

            while not env.is_done:
                prompt = obs.to_prompt()
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]

                # Tokenize and generate
                inputs = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=256,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

                response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                action = parse_action_from_text(response)

                if action is None:
                    # Fallback: do nothing
                    action = PolicyAction(
                        tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.01,
                        interest_rate_delta=0.0, stimulus_package=0.0,
                        import_tariff_delta=0.0, money_supply_delta=0.0,
                        minimum_wage_delta=0.0, reasoning="parse failure fallback",
                    )

                obs = env.step(action)

            grade = run_grader(task_id, env._history)
            task_scores.append(grade["score"])

        results[task_id] = {
            "mean": float(np.mean(task_scores)),
            "std": float(np.std(task_scores)),
            "scores": task_scores,
        }

    overall_mean = float(np.mean([r["mean"] for r in results.values()]))
    results["overall_mean"] = overall_mean
    return results


# ── Main Training Loop ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  SocialContract-v0 — GRPO Training with Unsloth + TRL")
    print("=" * 70)

    # ── Step 1: Load model with Unsloth ──────────────────────────────────
    print("\n[1/5] Loading model with Unsloth...")
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=TRAINING_CONFIG["model_name"],
        max_seq_length=TRAINING_CONFIG["max_seq_length"],
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=TRAINING_CONFIG["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=TRAINING_CONFIG["lora_alpha"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model: {TRAINING_CONFIG['model_name']}")
    print(f"  LoRA rank: {TRAINING_CONFIG['lora_r']}")

    # ── Step 2: Pre-training evaluation ──────────────────────────────────
    print("\n[2/5] Pre-training evaluation...")
    pre_scores = evaluate_model(model, tokenizer,
                                tasks=TRAINING_CONFIG["training_tasks"],
                                seeds=[42])
    print(f"  Pre-training mean score: {pre_scores['overall_mean']:.4f}")
    for task_id, data in pre_scores.items():
        if task_id != "overall_mean":
            print(f"    {task_id}: {data['mean']:.4f}")

    # ── Step 3: Create training dataset ──────────────────────────────────
    print("\n[3/5] Generating training prompts...")
    dataset = create_training_dataset(n_prompts=200)
    print(f"  Created {len(dataset)} training prompts across "
          f"{len(TRAINING_CONFIG['training_tasks'])} tasks")

    # ── Step 4: GRPO Training ────────────────────────────────────────────
    print("\n[4/5] Starting GRPO training...")
    from trl import GRPOTrainer, GRPOConfig


    def reward_function(completions, **kwargs):
        """
        GRPO reward: run the environment step with parsed action,
        return the grader score as the reward signal.
        """
        rewards = []
        task_ids = kwargs.get("task_id", ["task1_stability"] * len(completions))
        seeds_list = kwargs.get("seed", [42] * len(completions))
        warmups = kwargs.get("warmup_steps", [0] * len(completions))

        for i, completion in enumerate(completions):
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            else:
                text = str(completion)

            task_id = task_ids[i] if i < len(task_ids) else "task1_stability"
            seed = seeds_list[i] if i < len(seeds_list) else 42
            warmup = warmups[i] if i < len(warmups) else 0

            r = compute_environment_reward(text, task_id, seed, warmup)
            rewards.append(r)

            # Print live sample for every completion
            action = parse_action_from_text(text)
            if action:
                action_json = json.dumps({
                    "tax_delta": round(action.tax_delta, 3),
                    "interest_rate_delta": round(action.interest_rate_delta, 3),
                    "stimulus_package": round(action.stimulus_package, 1),
                    "ubi_delta": round(action.ubi_delta, 2),
                    "public_good_delta": round(action.public_good_delta, 3),
                    "reasoning": (action.reasoning or "")[:80] + "...",
                })
                print(f"[Live Sample] Task: {task_id} | Grader Score: {r:.4f}")
                print(f"  AI Decision -> {action_json}")

        return rewards

    training_args = GRPOConfig(
        output_dir=TRAINING_CONFIG["save_dir"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        max_completion_length=TRAINING_CONFIG["max_new_tokens"],
        num_generations=TRAINING_CONFIG["num_generations"],
        logging_steps=5,
        save_steps=50,
        max_steps=TRAINING_CONFIG["max_training_steps"],
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        temperature=0.7,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_function],
    )

    print("  Training started...")
    train_result = trainer.train()
    print(f"  Training complete. Total steps: {train_result.global_step}")

    # ── Step 5: Post-training evaluation ─────────────────────────────────
    print("\n[5/5] Post-training evaluation...")
    post_scores = evaluate_model(model, tokenizer,
                                 tasks=TRAINING_CONFIG["training_tasks"],
                                 seeds=[42])
    print(f"  Post-training mean score: {post_scores['overall_mean']:.4f}")

    # ── Results Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  {'Task':<25} {'Pre':>8} {'Post':>8} {'Delta':>8}")
    print("  " + "-" * 52)

    for task_id in TRAINING_CONFIG["training_tasks"]:
        pre = pre_scores.get(task_id, {}).get("mean", 0.0)
        post = post_scores.get(task_id, {}).get("mean", 0.0)
        delta = post - pre
        marker = "↑" if delta > 0 else ("↓" if delta < 0 else "—")
        print(f"  {task_id:<25} {pre:>8.4f} {post:>8.4f} {delta:>+8.4f} {marker}")

    pre_mean = pre_scores["overall_mean"]
    post_mean = post_scores["overall_mean"]
    print(f"\n  Overall Mean:           {pre_mean:>8.4f} {post_mean:>8.4f} "
          f"{post_mean - pre_mean:>+8.4f}")
    print("=" * 70)

    # Save results for plotting
    results_log = {
        "pre_training": {k: v for k, v in pre_scores.items() if k != "overall_mean"},
        "post_training": {k: v for k, v in post_scores.items() if k != "overall_mean"},
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "config": TRAINING_CONFIG,
        "train_steps": train_result.global_step,
    }

    log_path = os.path.join(os.path.dirname(__file__), "training_results.json")
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\n  Results saved to: {log_path}")

    # Save model
    print(f"  Saving model to: {TRAINING_CONFIG['save_dir']}")
    model.save_pretrained(TRAINING_CONFIG["save_dir"])
    tokenizer.save_pretrained(TRAINING_CONFIG["save_dir"])
    print("  Done!")


if __name__ == "__main__":
    main()
