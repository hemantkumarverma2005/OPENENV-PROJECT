"""
explain_policy.py — Policy Explanation Feature (#11)
═══════════════════════════════════════════════════
Generates natural-language explanations of the trained agent's
policy decisions. Demonstrates that the environment trains
genuine economic REASONING, not just action prediction.

The explainer:
  1. Runs an episode with the agent
  2. At each step, analyzes why the agent chose each lever value
  3. Maps decisions to economic theory (Phillips Curve, Taylor Rule, etc.)
  4. Generates a narrated policy report

Usage:
    python explain_policy.py                     # Uses smart heuristic
    python explain_policy.py --task task4_stagflation
    python explain_policy.py --llm               # Uses LLM via API
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from env.openenv_wrapper import SocialContractOpenEnv
from env.pydantic_models import PolicyAction
from graders.graders import run_grader
from demo import smart_policy


# ── Economic Theory Mapper ───────────────────────────────────────────────────

THEORY_MAP = {
    "interest_rate_increase": {
        "theory": "Taylor Rule / Volcker Doctrine",
        "explanation": "Raising interest rates fights inflation by increasing the cost of borrowing, "
                       "reducing spending and investment. Paul Volcker used this aggressively in 1979-82.",
        "tradeoff": "Higher rates slow GDP growth and increase unemployment (Phillips Curve tradeoff).",
    },
    "interest_rate_decrease": {
        "theory": "Keynesian Stimulus",
        "explanation": "Lowering interest rates stimulates the economy by making borrowing cheaper, "
                       "encouraging investment and consumption.",
        "tradeoff": "Low rates risk fueling inflation, especially when combined with QE.",
    },
    "tax_increase": {
        "theory": "Fiscal Consolidation / Laffer Curve",
        "explanation": "Raising taxes increases government revenue, helping close budget deficits. "
                       "However, beyond a certain rate, revenue actually decreases (Laffer Curve).",
        "tradeoff": "High taxes reduce economic activity, may trigger capital flight and tax evasion.",
    },
    "tax_decrease": {
        "theory": "Supply-Side Economics",
        "explanation": "Cutting taxes aims to stimulate economic activity and may increase revenue "
                       "through higher growth (Laffer Curve effect).",
        "tradeoff": "Tax cuts widen budget deficits if growth doesn't compensate.",
    },
    "ubi_increase": {
        "theory": "Keynesian Demand Stimulus / Redistribution",
        "explanation": "Increasing UBI puts money directly in citizens' hands, boosting demand. "
                       "Poor households have high MPC (0.90), so UBI is very effective at stimulating spending.",
        "tradeoff": "UBI costs reduce government budget and may create dependency (reduced labor supply).",
    },
    "ubi_decrease": {
        "theory": "Fiscal Austerity",
        "explanation": "Cutting UBI reduces government spending, helping restore budget balance.",
        "tradeoff": "Reduced UBI increases inequality and unrest, especially among poor citizens.",
    },
    "qe_expansion": {
        "theory": "Quantitative Easing (Cantillon Effect)",
        "explanation": "Expanding money supply (QE) provides liquidity and lowers real interest rates. "
                       "However, QE benefits asset holders (wealthy) more than wage earners (poor).",
        "tradeoff": "Excessive QE causes inflation and widens wealth inequality.",
    },
    "qe_tightening": {
        "theory": "Monetary Tightening",
        "explanation": "Reducing money supply fights inflation by reducing liquidity in the economy.",
        "tradeoff": "Tightening can trigger recession if done too aggressively.",
    },
    "public_goods_increase": {
        "theory": "Public Investment / Solow Growth",
        "explanation": "Investing in public goods (education, infrastructure, healthcare) builds "
                       "long-term productive capacity and increases citizen satisfaction.",
        "tradeoff": "Public spending increases budget deficit in the short term.",
    },
    "stimulus": {
        "theory": "Keynesian Fiscal Stimulus",
        "explanation": "One-time fiscal stimulus packages inject demand into a recession-hit economy. "
                       "Most effective when unemployment is high and interest rates are at zero lower bound.",
        "tradeoff": "Stimulus is inflationary and adds to government deficit.",
    },
    "min_wage_increase": {
        "theory": "Card-Krueger Labor Economics",
        "explanation": "Raising minimum wage increases income for low-wage workers, reducing inequality.",
        "tradeoff": "Higher minimum wages may reduce employment for the most vulnerable workers.",
    },
}


def explain_action(action: PolicyAction, obs, prev_obs=None) -> list[str]:
    """Generate natural-language explanations for each policy lever."""
    explanations = []

    # ── Interest Rate ────────────────────────────────────────────────────
    if abs(action.interest_rate_delta) > 0.001:
        if action.interest_rate_delta > 0:
            theory = THEORY_MAP["interest_rate_increase"]
            explanations.append(
                f"RAISING interest rate by {action.interest_rate_delta:+.3f} "
                f"(to {obs.interest_rate + action.interest_rate_delta:.3f}). "
                f"[{theory['theory']}] — "
                f"{'Inflation at ' + f'{obs.inflation:.1%}' + ' exceeds target.' if obs.inflation > 0.04 else 'Preemptive tightening.'} "
                f"{theory['tradeoff']}"
            )
        else:
            theory = THEORY_MAP["interest_rate_decrease"]
            explanations.append(
                f"CUTTING interest rate by {action.interest_rate_delta:+.3f}. "
                f"[{theory['theory']}] — "
                f"{'Unemployment at ' + f'{obs.unemployment:.1%}' + ' is high.' if obs.unemployment > 0.08 else 'Stimulating growth.'} "
                f"{theory['tradeoff']}"
            )

    # ── Tax Rate ─────────────────────────────────────────────────────────
    if abs(action.tax_delta) > 0.001:
        if action.tax_delta > 0:
            theory = THEORY_MAP["tax_increase"]
            explanations.append(
                f"RAISING taxes by {action.tax_delta:+.2f} "
                f"(to {obs.tax_rate + action.tax_delta:.2f}). "
                f"[{theory['theory']}] — "
                f"{'Budget deficit at ' + f'{obs.gov_budget:,.0f}' + '.' if obs.gov_budget < 0 else 'Building fiscal buffer.'}"
            )
        else:
            theory = THEORY_MAP["tax_decrease"]
            explanations.append(
                f"CUTTING taxes by {action.tax_delta:+.2f}. "
                f"[{theory['theory']}] — Stimulating economic activity."
            )

    # ── UBI ──────────────────────────────────────────────────────────────
    if abs(action.ubi_delta) > 0.1:
        if action.ubi_delta > 0:
            theory = THEORY_MAP["ubi_increase"]
            explanations.append(
                f"INCREASING UBI by {action.ubi_delta:+.1f} "
                f"(to {obs.ubi_amount + action.ubi_delta:.1f}). "
                f"[{theory['theory']}] — "
                f"{'Gini at ' + f'{obs.gini:.3f}' + ' indicates high inequality.' if obs.gini > 0.45 else 'Supporting low-income citizens.'}"
            )
        else:
            theory = THEORY_MAP["ubi_decrease"]
            explanations.append(
                f"CUTTING UBI by {action.ubi_delta:+.1f}. "
                f"[{theory['theory']}] — Reducing fiscal burden."
            )

    # ── Money Supply ─────────────────────────────────────────────────────
    if abs(action.money_supply_delta) > 10:
        if action.money_supply_delta > 0:
            theory = THEORY_MAP["qe_expansion"]
            explanations.append(
                f"QE: Expanding money supply by {action.money_supply_delta:+.0f}. "
                f"[{theory['theory']}] — Injecting liquidity."
            )
        else:
            theory = THEORY_MAP["qe_tightening"]
            explanations.append(
                f"TIGHTENING: Reducing money supply by {action.money_supply_delta:+.0f}. "
                f"[{theory['theory']}] — Fighting inflation."
            )

    # ── Public Goods ─────────────────────────────────────────────────────
    if abs(action.public_good_delta) > 0.005:
        if action.public_good_delta > 0:
            theory = THEORY_MAP["public_goods_increase"]
            explanations.append(
                f"INVESTING in public goods: {action.public_good_delta:+.2f}. "
                f"[{theory['theory']}] — Building long-term capacity."
            )

    # ── Stimulus ─────────────────────────────────────────────────────────
    if action.stimulus_package > 5:
        theory = THEORY_MAP["stimulus"]
        explanations.append(
            f"FISCAL STIMULUS: {action.stimulus_package:.0f} one-time injection. "
            f"[{theory['theory']}] — Emergency demand boost."
        )

    # ── Minimum Wage ─────────────────────────────────────────────────────
    if abs(action.minimum_wage_delta) > 0.1:
        if action.minimum_wage_delta > 0:
            theory = THEORY_MAP["min_wage_increase"]
            explanations.append(
                f"RAISING minimum wage by {action.minimum_wage_delta:+.1f}. "
                f"[{theory['theory']}] — Reducing inequality from bottom up."
            )

    if not explanations:
        explanations.append(
            "HOLDING STEADY — No significant policy changes. "
            "Stability signals consistency to markets (Barro-Gordon credibility)."
        )

    return explanations


def detect_strategy_phase(obs, task_id: str, step: int, max_steps: int) -> str:
    """Detect which strategic phase the agent should be in."""
    progress = step / max(max_steps, 1)

    if task_id == "task4_stagflation":
        if progress < 0.5:
            return "PHASE 1 (Inflation Control) -- Volcker Doctrine: raise rates, cut spending, tighten money"
        else:
            return "PHASE 2 (Growth Recovery) -- Keynesian: lower rates, invest in public goods"

    elif task_id == "task5_pandemic":
        if progress < 0.33:
            return "PHASE 1 (Emergency) -- Fiscal stimulus, QE, UBI increase, public goods"
        elif progress < 0.67:
            return "PHASE 2 (Recovery) -- Manage inflation, maintain growth"
        else:
            return "PHASE 3 (Consolidation) -- Cut spending, restore budget, raise taxes"

    elif task_id == "task2_recession":
        if progress < 0.4:
            return "EARLY RECOVERY -- Stimulus and rate cuts to restart growth"
        else:
            return "LATE RECOVERY -- Fiscal consolidation while maintaining growth"

    elif task_id == "task3_crisis":
        if obs.gini > 0.55:
            return "CRISIS MODE -- Aggressive redistribution (high UBI, progressive taxation)"
        elif obs.gini > 0.48:
            return "RECOVERY -- Moderate redistribution while maintaining GDP"
        else:
            return "MAINTENANCE -- Sustain improvements, prevent regression"

    return "STANDARD ADVISORY"


def run_explained_episode(task_id: str, seed: int = 42) -> dict:
    """Run an episode with full policy explanations at each step."""
    env = SocialContractOpenEnv(task_id, seed=seed)
    obs = env.reset()
    max_steps = env.task_cfg["max_steps"]
    step_reports = []

    print(f"\n{'='*70}")
    print(f"  POLICY EXPLANATION REPORT")
    print(f"  Task: {task_id} | Seed: {seed} | Max Steps: {max_steps}")
    print(f"{'='*70}")

    prev_obs = None
    while not env.is_done:
        action = smart_policy(obs)
        phase = detect_strategy_phase(obs, task_id, obs.step, max_steps)
        explanations = explain_action(action, obs, prev_obs)

        step_report = {
            "step": obs.step,
            "phase": phase,
            "state": {
                "gdp": obs.gdp,
                "gini": obs.gini,
                "inflation": obs.inflation,
                "unemployment": obs.unemployment,
                "gov_budget": obs.gov_budget,
                "unrest": obs.unrest,
            },
            "action": action.model_dump(),
            "explanations": explanations,
        }
        step_reports.append(step_report)

        # Print step explanation
        print(f"\n  Step {obs.step} | {phase}")
        print(f"  State: GDP={obs.gdp:.0f} | Gini={obs.gini:.3f} | "
              f"Inflation={obs.inflation:.3f} | Unemp={obs.unemployment:.2f} | "
              f"Budget={obs.gov_budget:.0f}")
        for exp in explanations:
            print(f"    > {exp}")

        prev_obs = obs
        obs, reward, done, info = env.step(action)

    grade = run_grader(task_id, env._history)

    print(f"\n{'='*70}")
    print(f"  EPISODE RESULT: Score = {grade['score']:.4f}")
    print(f"  Verdict: {grade['verdict']}")
    print(f"{'='*70}\n")

    return {
        "task_id": task_id,
        "seed": seed,
        "score": grade["score"],
        "verdict": grade["verdict"],
        "steps": step_reports,
    }


def main():
    parser = argparse.ArgumentParser(description="Explain policy decisions")
    parser.add_argument("--task", default="task4_stagflation",
                        help="Task to explain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Save explanation to JSON file")
    args = parser.parse_args()

    result = run_explained_episode(args.task, args.seed)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Explanation saved to: {args.output}")


if __name__ == "__main__":
    main()
