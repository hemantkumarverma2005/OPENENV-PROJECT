"""
demo.py — Offline evaluation demo for SocialContract-v0
Runs all 5 tasks with a task-aware rule-based policy, shows grader scores.
No API key or server required — runs fully locally.
Also demonstrates multi-seed generalization testing.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from env.openenv_wrapper import SocialContractOpenEnv
from env.pydantic_models import PolicyAction
from graders.graders import run_grader

ALL_TASKS = ["task1_stability", "task2_recession", "task3_crisis", "task4_stagflation", "task5_pandemic"]


def smart_policy(obs) -> PolicyAction:
    """
    Task-aware rule-based policy using all 8 policy levers.
    Academic-grade strategy:
    - Phillips Curve awareness (inflation-unemployment tradeoff)
    - Taylor Rule for interest rates
    - Targeted QE/tightening for monetary conditions
    - Minimum wage for redistributive objectives
    - Phase-aware strategy for expert tasks
    """
    tax_d = 0.0; ubi_d = 0.0; pg_d  = 0.0
    ir_d  = 0.0; stim  = 0.0; tariff_d = 0.0
    ms_d  = 0.0; mw_d  = 0.0
    task  = obs.task_id

    if task == "task1_stability":
        if obs.gov_budget < 100:
            tax_d += 0.02; ubi_d -= 0.2
        elif obs.gov_budget > 800:
            tax_d -= 0.01; ubi_d += 0.2
        if obs.gini > 0.42:
            ubi_d += 0.3; mw_d = 0.3
        if obs.unrest > 0.10:
            pg_d += 0.02; ubi_d += 0.2
        if obs.gdp_delta < 0:
            pg_d += 0.01; ir_d -= 0.005; ms_d = 20.0

    elif task == "task2_recession":
        pg_d += 0.05; tax_d += 0.02; ir_d -= 0.01
        ms_d = 60.0  # Mild QE for stimulus
        if obs.step < 10:
            stim = 50.0
        if obs.ubi_amount > 1.5:
            ubi_d -= 0.3
        if obs.unrest > 0.25:
            pg_d += 0.02
        if obs.gov_budget > -500:
            stim = 0.0; ms_d = 0.0
        mw_d = 0.2  # Small minimum wage boost

    elif task == "task3_crisis":
        if obs.gini > 0.55:
            tax_d += 0.05; ubi_d += 3.0; tariff_d += 0.02; mw_d = 1.0
        elif obs.gini > 0.48:
            tax_d += 0.03; ubi_d += 1.5; mw_d = 0.5
        elif obs.gini > 0.43:
            tax_d += 0.01; ubi_d += 0.5
        if obs.unrest > 0.08:
            pg_d += 0.04; ubi_d += 1.0
        if obs.gov_budget < -800:
            tax_d += 0.02; ubi_d -= 1.0
        pg_d += 0.02
        if obs.gdp_delta < 0:
            ir_d -= 0.005; ms_d = 30.0

    elif task == "task4_stagflation":
        max_steps = 35
        if obs.step < max_steps // 2:
            # PHASE 1: Inflation control (Volcker + monetary tightening)
            if obs.inflation > 0.08:
                ir_d += 0.02; tax_d += 0.03; ubi_d -= 2.0; pg_d -= 0.01
                ms_d = -150.0  # Tight money
            elif obs.inflation > 0.04:
                ir_d += 0.01; tax_d += 0.02; ubi_d -= 1.0
                ms_d = -80.0
            else:
                ir_d += 0.005; ms_d = -40.0
        else:
            # PHASE 2: Growth recovery
            if obs.inflation < 0.05:
                ir_d -= 0.01; pg_d += 0.03; ms_d = 40.0
                if obs.unemployment > 0.12:
                    pg_d += 0.02; stim = 30.0; mw_d = 0.3
                elif obs.unemployment > 0.08:
                    pg_d += 0.01
            if obs.gov_budget < -800:
                tax_d += 0.01; ubi_d -= 0.5

    elif task == "task5_pandemic":
        max_steps = 30
        p1 = max_steps // 3; p2 = 2 * max_steps // 3
        if obs.step < p1:
            # PHASE 1: Emergency — spend + QE
            pg_d += 0.05; ir_d -= 0.01; tax_d -= 0.01
            stim = 100.0 if obs.unemployment > 0.15 else 50.0
            ubi_d += 1.0 if obs.unemployment > 0.15 else 0.0
            ms_d = 150.0  # Aggressive QE
            mw_d = 0.5
        elif obs.step < p2:
            # PHASE 2: Recovery — fight inflation
            if obs.inflation > 0.05:
                ir_d += 0.01; ubi_d -= 1.5; tax_d += 0.02
                ms_d = -80.0  # Start tightening
            pg_d += 0.02
            if obs.unemployment > 0.10:
                pg_d += 0.02; stim = 30.0
        else:
            # PHASE 3: Consolidation — restore budget
            tax_d += 0.03; ubi_d -= 2.0; pg_d -= 0.01; ir_d += 0.005
            stim = 0.0; ms_d = -50.0  # Continue tightening
            if obs.gov_budget > -500:
                tax_d -= 0.01

    return PolicyAction(
        tax_delta=float(np.clip(tax_d, -0.10, 0.10)),
        ubi_delta=float(np.clip(ubi_d, -10.0, 10.0)),
        public_good_delta=float(np.clip(pg_d, -0.10, 0.10)),
        interest_rate_delta=float(np.clip(ir_d, -0.03, 0.03)),
        stimulus_package=float(np.clip(stim, 0.0, 500.0)),
        import_tariff_delta=float(np.clip(tariff_d, -0.05, 0.05)),
        money_supply_delta=float(np.clip(ms_d, -500.0, 500.0)),
        minimum_wage_delta=float(np.clip(mw_d, -2.0, 2.0)),
        reasoning=f"Task-aware 8-lever policy for {task}",
    )


def run_task_demo(task_id: str, seed: int = 42) -> dict:
    env = SocialContractOpenEnv(task_id, seed=seed)
    obs = env.reset()
    total_reward = 0.0
    while not env.is_done:
        action = smart_policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward.total
    grade = run_grader(task_id, env._history)
    return {
        "steps": env._step_count, "total_reward": round(total_reward, 4),
        "score": grade["score"], "verdict": grade["verdict"],
    }


def run_random_baseline(task_id: str, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed + 999)
    env = SocialContractOpenEnv(task_id, seed=seed)
    env.reset()
    total_reward = 0.0
    while not env.is_done:
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
        _, reward, _, _ = env.step(action)
        total_reward += reward.total
    grade = run_grader(task_id, env._history)
    return {"score": grade["score"], "verdict": grade["verdict"]}


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  SocialContract-v0  —  Offline Evaluation Demo")
    print("  8-lever task-aware smart policy vs random baseline")
    print("=" * 70)

    print("\n── Single-Seed Evaluation (seed=42) ────────────────────────")
    print(f"  {'Task':<25} {'Smart':>8} {'Random':>8} {'Gap':>8}  Status")
    print("  " + "-" * 60)

    results = {}
    for task_id in ALL_TASKS:
        smart  = run_task_demo(task_id, seed=42)
        random = run_random_baseline(task_id, seed=42)
        results[task_id] = smart
        gap = smart["score"] - random["score"]
        status = "✓ OK" if gap > 0.03 else ("⚠ Small" if gap > 0 else "✗ BROKEN")
        print(f"  {task_id:<25} {smart['score']:>8.3f} {random['score']:>8.3f} "
              f"{gap:>+8.3f}  {status}")

    mean = sum(r["score"] for r in results.values()) / len(results)
    print(f"\n  Smart Mean Score: {mean:.4f}")

    print("\n── Multi-Seed Generalization Test ───────────────────────────")
    seeds = [42, 99, 7, 256, 1337]
    all_means = []
    for task_id in ALL_TASKS:
        scores = [run_task_demo(task_id, seed=s)["score"] for s in seeds]
        avg = np.mean(scores); std = np.std(scores)
        all_means.append(avg)
        print(f"  {task_id:25s}  avg={avg:.4f}  std={std:.4f}  "
              f"range=[{min(scores):.4f}, {max(scores):.4f}]")

    print(f"\n  Overall Mean (5 seeds × 5 tasks): {np.mean(all_means):.4f}")
    print("\n" + "=" * 70)
    print("  Demo complete! Smart policy should beat random on all tasks.")
    print("=" * 70 + "\n")
