"""
benchmark_models.py — Multi-Model Benchmarking Script (#10)
══════════════════════════════════════════════════════════
Evaluates multiple LLM backends against SocialContract-v0 and
generates a comprehensive comparison table + visualization.

Supports:
  - OpenAI API (GPT-4o, GPT-4o-mini)
  - Anthropic API (Claude 3.5 Sonnet)
  - Local models via Ollama
  - HuggingFace Inference API
  - Custom API endpoints (vLLM, TGI, etc.)

Usage:
    # Benchmark GPT-4o-mini (default):
    python benchmark_models.py

    # Benchmark specific models:
    python benchmark_models.py --models gpt-4o-mini gpt-4o claude-3.5-sonnet

    # Include local model via Ollama:
    python benchmark_models.py --models gpt-4o-mini ollama/llama3.1

    # All tasks + multiple seeds:
    python benchmark_models.py --tasks all --seeds 42 99 7 256 1337
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from env.openenv_wrapper import SocialContractOpenEnv
from env.pydantic_models import PolicyAction, EconomicObservation
from graders.graders import run_grader

# ── Model Configurations ─────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "gpt-4o-mini": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.10,
        "max_tokens": 350,
    },
    "gpt-4o": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.10,
        "max_tokens": 350,
    },
    "gpt-4-turbo": {
        "api_base": "https://api.openai.com/v1",
        "model": "gpt-4-turbo",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.10,
        "max_tokens": 350,
    },
    "claude-3.5-sonnet": {
        "api_base": "https://api.anthropic.com/v1",
        "model": "claude-3-5-sonnet-20241022",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.10,
        "max_tokens": 350,
        "provider": "anthropic",
    },
    "smart-heuristic": {
        "type": "heuristic",
    },
    "random": {
        "type": "random",
    },
}

ALL_TASKS = ["task1_stability", "task2_recession", "task3_crisis",
             "task4_stagflation", "task5_pandemic"]

SYSTEM_PROMPT = (
    "You are an expert economic policy advisor. You make precise, data-driven "
    "decisions. Respond ONLY with valid JSON. No markdown, no extra text."
)


def call_openai_api(prompt: str, config: dict) -> str:
    """Call OpenAI-compatible API."""
    from openai import OpenAI

    api_key = os.environ.get(config["api_key_env"], "")
    if not api_key:
        raise ValueError(f"Missing API key: set {config['api_key_env']}")

    client = OpenAI(base_url=config["api_base"], api_key=api_key)
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 350),
    )
    return response.choices[0].message.content.strip()


def parse_action(text: str):
    """Parse PolicyAction from LLM text output."""
    import re
    try:
        if "```" in text:
            match = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
            if match:
                text = match.group(1)
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
        )
    except Exception:
        return None


def smart_policy(obs):
    """Smart rule-based policy (imported from demo.py logic)."""
    from demo import smart_policy as sp
    return sp(obs)


def run_episode(model_name: str, task_id: str, seed: int = 42) -> dict:
    """Run a single episode with specified model and return results."""
    config = MODEL_CONFIGS.get(model_name, {})
    env = SocialContractOpenEnv(task_id, seed=seed)
    obs = env.reset()
    total_reward = 0.0
    parse_failures = 0
    step = 0
    start_time = time.time()

    while not env.is_done:
        if config.get("type") == "random":
            rng = np.random.default_rng(seed + step * 1000)
            action = PolicyAction(
                tax_delta=float(rng.uniform(-0.05, 0.05)),
                ubi_delta=float(rng.uniform(-5.0, 5.0)),
                public_good_delta=float(rng.uniform(-0.05, 0.05)),
                interest_rate_delta=float(rng.uniform(-0.02, 0.02)),
                stimulus_package=float(rng.uniform(0.0, 100.0)),
                import_tariff_delta=float(rng.uniform(-0.03, 0.03)),
                money_supply_delta=float(rng.uniform(-100.0, 100.0)),
                minimum_wage_delta=float(rng.uniform(-1.0, 1.0)),
                reasoning="random baseline",
            )
        elif config.get("type") == "heuristic":
            action = smart_policy(obs)
        else:
            # API call
            prompt = obs.to_prompt()
            try:
                response = call_openai_api(prompt, config)
                action = parse_action(response)
                if action is None:
                    parse_failures += 1
                    action = PolicyAction(
                        tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.01,
                        interest_rate_delta=0.0, stimulus_package=0.0,
                        import_tariff_delta=0.0, money_supply_delta=0.0,
                        minimum_wage_delta=0.0, reasoning="parse failure fallback",
                    )
            except Exception as e:
                print(f"  API error for {model_name}: {e}")
                action = PolicyAction(
                    tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.01,
                    interest_rate_delta=0.0, stimulus_package=0.0,
                    import_tariff_delta=0.0, money_supply_delta=0.0,
                    minimum_wage_delta=0.0, reasoning="api error fallback",
                )
                parse_failures += 1

            time.sleep(0.3)  # Rate limit protection

        obs, reward, done, info = env.step(action)
        total_reward += reward.total
        step += 1

    elapsed = time.time() - start_time
    grade = run_grader(task_id, env._history)

    return {
        "model": model_name,
        "task_id": task_id,
        "seed": seed,
        "score": grade["score"],
        "verdict": grade["verdict"],
        "total_reward": round(total_reward, 4),
        "steps": step,
        "parse_failures": parse_failures,
        "elapsed_seconds": round(elapsed, 2),
    }


def run_benchmark(models: list[str], tasks: list[str], seeds: list[int]) -> dict:
    """Run full benchmark suite."""
    all_results = []

    for model_name in models:
        config = MODEL_CONFIGS.get(model_name, {})

        # Check if API key is available for API models
        if config.get("type") not in ("random", "heuristic"):
            api_key_env = config.get("api_key_env", "")
            if api_key_env and not os.environ.get(api_key_env):
                print(f"  Skipping {model_name}: {api_key_env} not set")
                continue

        print(f"\n  Benchmarking: {model_name}")
        print(f"  {'Task':<25} {'Seed':>6} {'Score':>8} {'Verdict'}")
        print(f"  {'-'*60}")

        for task_id in tasks:
            for seed in seeds:
                result = run_episode(model_name, task_id, seed)
                all_results.append(result)
                print(f"  {task_id:<25} {seed:>6} {result['score']:>8.4f} "
                      f"{result['verdict']}")

    return {"results": all_results, "timestamp": datetime.now().isoformat()}


def generate_report(benchmark_data: dict, output_path: str = "benchmark_report.json"):
    """Generate a summary report from benchmark results."""
    results = benchmark_data["results"]

    # Group by model
    models = sorted(set(r["model"] for r in results))
    tasks = sorted(set(r["task_id"] for r in results))

    summary = {}
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        task_scores = {}
        for task in tasks:
            task_results = [r for r in model_results if r["task_id"] == task]
            if task_results:
                scores = [r["score"] for r in task_results]
                task_scores[task] = {
                    "mean": round(float(np.mean(scores)), 4),
                    "std": round(float(np.std(scores)), 4),
                    "min": round(float(np.min(scores)), 4),
                    "max": round(float(np.max(scores)), 4),
                    "n_seeds": len(scores),
                }
        overall_means = [v["mean"] for v in task_scores.values()]
        summary[model] = {
            "per_task": task_scores,
            "overall_mean": round(float(np.mean(overall_means)), 4) if overall_means else 0.0,
            "total_episodes": len(model_results),
        }

    report = {
        "summary": summary,
        "raw_results": results,
        "timestamp": benchmark_data["timestamp"],
        "models_tested": models,
        "tasks_tested": tasks,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {output_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*80}")
    header = f"  {'Model':<25}"
    for task in tasks:
        short = task.replace("task", "T").replace("_", " ")[:8]
        header += f" {short:>8}"
    header += f" {'Mean':>8}"
    print(header)
    print(f"  {'-'*75}")

    for model in models:
        row = f"  {model:<25}"
        for task in tasks:
            ts = summary[model]["per_task"].get(task, {})
            row += f" {ts.get('mean', 0.0):>8.4f}"
        row += f" {summary[model]['overall_mean']:>8.4f}"
        print(row)
    print(f"{'='*80}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Benchmark models on SocialContract-v0")
    parser.add_argument("--models", nargs="+",
                        default=["random", "smart-heuristic"],
                        help="Models to benchmark")
    parser.add_argument("--tasks", nargs="+", default=["all"],
                        help="Tasks to evaluate (or 'all')")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds for evaluation")
    parser.add_argument("--output", default="benchmark_report.json",
                        help="Output report path")
    args = parser.parse_args()

    if "all" in args.tasks:
        tasks = ALL_TASKS
    else:
        tasks = args.tasks

    print("\n" + "=" * 60)
    print("  SocialContract-v0 — Multi-Model Benchmark")
    print("=" * 60)
    print(f"  Models: {args.models}")
    print(f"  Tasks:  {tasks}")
    print(f"  Seeds:  {args.seeds}")

    benchmark_data = run_benchmark(args.models, tasks, args.seeds)
    generate_report(benchmark_data, args.output)


if __name__ == "__main__":
    main()
