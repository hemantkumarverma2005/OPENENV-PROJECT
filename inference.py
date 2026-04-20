"""
inference.py
────────────
Baseline inference script using OpenAI client.
Reads credentials from environment variables.
Emits structured [START], [STEP], [END] logs.

Features:
  - Chain-of-thought: LLM first analyses, then decides
  - History-aware prompts with trend arrows
  - Action smoothing to reduce volatility penalty
  - Retry logic (3 attempts) on LLM failure
  - Per-task system prompts with grader thresholds
  - Grader-aware strategy coaching

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="your-token"
    python inference.py
"""

import os
import sys
import json
import time

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))

from env.openenv_wrapper import SocialContractOpenEnv
from env.pydantic_models import PolicyAction
from graders.graders import run_grader

# ── Credentials from environment ──────────────────────────────────────────────
API_BASE_URL   = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME     = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
HF_TOKEN       = os.environ.get("HF_TOKEN", "")

api_key_to_use = OPENAI_API_KEY or HF_TOKEN or "dummy-key"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=api_key_to_use,
)

TASKS = ["task1_stability", "task2_recession", "task3_crisis", "task4_stagflation", "task5_pandemic"]

SYSTEM_PROMPTS = {
    "task1_stability": (
        "You are an IMF advisor. TARGETS: GDP growth ≥ 0, Gini < 0.43, unrest < 0.15, budget surplus. "
        "Grader rewards Gini STABILITY (low std-dev). Never reverse a lever direction without a shock. "
        "If metrics are stable, set all deltas to ZERO. The grader HEAVILY penalises volatility. "
        "Respond only with valid JSON."
    ),
    "task2_recession": (
        "You are a World Bank advisor. TARGETS: budget from -2500 to surplus (60% weight on consistency), "
        "GDP positive, unrest < 0.40, unemployment falling. Grader measures TRAJECTORY not final value. "
        "Cut interest rates. Gradual tax increase. Use stimulus early then taper. "
        "Respond only with valid JSON."
    ),
    "task3_crisis": (
        "You are a UN advisor. TARGETS: Gini < 0.48, unrest < 0.10, GDP growth > 0 — ALL THREE simultaneously. "
        "DANGER: Gini > 0.88 = episode ends score=0. If Gini > 0.75, AGGRESSIVELY raise UBI and taxes. "
        "Use minimum wage increases to help poor. Grader rewards CONSISTENT Gini decrease. "
        "Respond only with valid JSON."
    ),
    "task4_stagflation": (
        "You are a central banker. HARD GATES (all required): inflation < 0.04, unemployment < 0.08, "
        "budget > -200, GDP positive in 60%+ steps. Inflation and unemployment CONSISTENCY counts 40% each. "
        "Volcker sequence: PHASE 1 (first half) fight inflation — raise rates, cut UBI, tighten money. "
        "PHASE 2 (second half) stimulate growth — lower rates, invest in public goods. "
        "Respond only with valid JSON."
    ),
    "task5_pandemic": (
        "You are an OECD advisor. HARD GATES: unemployment < 0.08 (from 0.25!), inflation < 0.06, "
        "budget > -500 (from -3000), GDP positive 55%+ steps. Unemployment REDUCTION from 0.25 is explicitly scored. "
        "Phase: PHASE 1 (steps 0-9) stimulus + QE + public goods, ubi_delta POSITIVE. "
        "PHASE 2 (steps 10-19) taper UBI, fight inflation with interest rates. "
        "PHASE 3 (steps 20-29): CRITICAL — ubi_delta MUST go NEGATIVE (-1 to -3) to trigger "
        "consolidation phase bonus. Raise tax_delta, cut spending. Budget recovery is scored here. "
        "Respond only with valid JSON."
    )
}

def build_history_summary(history: list) -> str:
    """Build a compact history summary with trend arrows and delta analysis."""
    if not history:
        return ""

    recent = history[-5:]
    summary = "\n--- RECENT HISTORY (Last 5 Steps) ---\n"
    summary += "Step | GDP        | Gini  | Infl. | Unrest | Budget  | Reward |\n"

    prev_gdp, prev_gini, prev_inf = None, None, None
    for item in recent:
        g = item["gdp"]
        gini = item["gini"]
        inf = item["inflation"]
        unrest = item["unrest"]
        budg = item["gov_budget"]
        rew = item.get("reward", 0.0)

        g_arr = "↑" if prev_gdp is not None and g > prev_gdp else ("↓" if prev_gdp is not None and g < prev_gdp else "-")
        gini_arr = "↑" if prev_gini is not None and gini > prev_gini else ("↓" if prev_gini is not None and gini < prev_gini else "-")
        inf_arr = "↑" if prev_inf is not None and inf > prev_inf else ("↓" if prev_inf is not None and inf < prev_inf else "-")

        summary += f" {item['step']:2d}  | {g:6.1f}   {g_arr}| {gini:.3f}{gini_arr}| {inf:.3f}{inf_arr}| {unrest:.3f}  | {budg:7.1f} | {rew:.3f}  |\n"
        prev_gdp, prev_gini, prev_inf = g, gini, inf

    # Add explicit delta guidance
    if len(history) >= 2:
        last = history[-1]
        prev = history[-2]
        summary += "\n📊 IMPACT OF YOUR LAST ACTION:\n"
        metrics = [
            ("GDP",          last["gdp"] - prev["gdp"]),
            ("Gini",         last["gini"] - prev["gini"]),
            ("Inflation",    last["inflation"] - prev["inflation"]),
            ("Unrest",       last["unrest"] - prev["unrest"]),
            ("Budget",       last["gov_budget"] - prev["gov_budget"]),
            ("Unemployment", last["unemployment"] - prev["unemployment"]),
        ]
        for name, delta in metrics:
            if abs(delta) > 0.001:
                direction = "IMPROVED ✓" if (
                    (name in ("GDP", "Budget") and delta > 0) or
                    (name in ("Gini", "Inflation", "Unrest", "Unemployment") and delta < 0)
                ) else "WORSENED ✗"
                summary += f"  {name}: {delta:+.4f} ({direction})\n"

    summary += f"\nLast action reward: {history[-1].get('reward', 0.0):.4f}. Adjust strategy based on what improved vs worsened.\n"

    return summary


def smooth_action(current_action: PolicyAction, last_action: PolicyAction, shock_occurring: bool) -> PolicyAction:
    """
    Dampen policy reversals to reduce volatility penalty.
    During shocks, let the LLM react freely.
    Direction reversals are dampened 60% (only 40% of the reversal goes through).
    Same-direction changes pass through with light smoothing.
    """
    if last_action is None or shock_occurring:
        return current_action

    def damp(curr, prev):
        # If direction reversal detected, dampen heavily (60% reduction)
        if curr * prev < -1e-8:
            return curr * 0.4
        # Same direction: let it through fully (don't slow correct moves)
        return curr

    return PolicyAction(
        tax_delta = damp(current_action.tax_delta, last_action.tax_delta),
        ubi_delta = damp(current_action.ubi_delta, last_action.ubi_delta),
        public_good_delta = damp(current_action.public_good_delta, last_action.public_good_delta),
        interest_rate_delta = damp(current_action.interest_rate_delta, last_action.interest_rate_delta),
        stimulus_package = current_action.stimulus_package,
        import_tariff_delta = damp(current_action.import_tariff_delta, last_action.import_tariff_delta),
        money_supply_delta = damp(current_action.money_supply_delta, last_action.money_supply_delta),
        minimum_wage_delta = damp(current_action.minimum_wage_delta, last_action.minimum_wage_delta),
        reasoning = current_action.reasoning,
        policy_speech = getattr(current_action, "policy_speech", None)
    )


def call_llm(prompt: str, task_id: str, history_summary: str, step_num: int = 0, max_steps: int = 20) -> PolicyAction:
    """
    Two-stage chain-of-thought inference:
      1. Ask the LLM to analyse the situation
      2. Ask it to decide on action based on its analysis

    Includes retry logic with 3 attempts.
    """
    sys_prompt_core = SYSTEM_PROMPTS.get(task_id, "")

    # Determine which phase we're in (for expert tasks)
    phase_hint = ""
    if task_id == "task5_pandemic":
        if step_num < max_steps // 3:
            phase_hint = "\n🔴 You are in PHASE 1 (Emergency). Prioritise spending and employment."
        elif step_num < 2 * max_steps // 3:
            phase_hint = "\n🟡 You are in PHASE 2 (Recovery). Manage inflation while maintaining growth."
        else:
            phase_hint = "\n🟢 You are in PHASE 3 (Consolidation). Reduce spending. Restore the budget."
    elif task_id == "task4_stagflation":
        if step_num < max_steps // 2:
            phase_hint = "\n🔴 You are in PHASE 1 (Inflation Control). Raise interest rates. Cut spending."
        else:
            phase_hint = "\n🟢 You are in PHASE 2 (Growth Recovery). Lower rates. Invest in public goods."

    system_prompt = (
        "You are an expert economic policy advisor. You make precise, data-driven decisions.\n"
        f"{sys_prompt_core}\n"
        f"{phase_hint}\n"
        "CRITICAL: The grader penalises policy volatility. Keep your actions CONSISTENT with your "
        "previous actions. Don't reverse direction unless the situation fundamentally changed.\n"
        "You must respond ONLY with valid JSON matching the exact format requested. "
        "No markdown, no extra text — only the JSON object."
    )

    full_prompt = prompt + "\n" + history_summary

    last_error = None
    for attempt in range(3):
        try:
            # Adjust temperature: lower for easy tasks, slightly higher for expert
            temp = 0.05 if task_id == "task1_stability" else 0.10

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": full_prompt},
                ],
                temperature=temp,
                max_tokens=350,
            )

            raw = response.choices[0].message.content.strip()

            # Clean markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)

            p_action = PolicyAction(
                tax_delta         = max(-0.10, min(0.10, float(parsed.get("tax_delta", 0.0)))),
                ubi_delta         = max(-10.0, min(10.0, float(parsed.get("ubi_delta", 0.0)))),
                public_good_delta = max(-0.10, min(0.10, float(parsed.get("public_good_delta", 0.0)))),
                interest_rate_delta = max(-0.03, min(0.03, float(parsed.get("interest_rate_delta", 0.0)))),
                stimulus_package  = max(0.0, min(500.0, float(parsed.get("stimulus_package", 0.0)))),
                import_tariff_delta = max(-0.05, min(0.05, float(parsed.get("import_tariff_delta", 0.0)))),
                money_supply_delta = max(-500.0, min(500.0, float(parsed.get("money_supply_delta", 0.0)))),
                minimum_wage_delta = max(-2.0, min(2.0, float(parsed.get("minimum_wage_delta", 0.0)))),
                reasoning         = str(parsed.get("reasoning", ""))[:200],
            )
            if "policy_speech" in parsed:
                p_action.policy_speech = str(parsed.get("policy_speech", ""))[:150]
            return p_action

        except Exception as e:
            last_error = e
            if attempt < 2:
                time.sleep(0.5)  # Brief pause before retry

    # All retries failed
    print(f"LLM Error after 3 attempts: {last_error}")
    return None  # Caller will use fallback


def fallback_action(obs) -> PolicyAction:
    """Intelligent rule-based fallback when LLM fails."""
    tax_d = 0.0
    ubi_d = 0.0
    pg_d  = 0.01
    ir_d  = 0.0
    stim  = 0.0
    tariff_d = 0.0
    ms_d  = 0.0
    mw_d  = 0.0

    if obs.gov_budget < -500:
        tax_d = 0.02
        ubi_d = -1.0
    elif obs.gov_budget > 500:
        tax_d = -0.01

    if obs.gini > 0.55:
        ubi_d = max(ubi_d, 2.0)
        tax_d = max(tax_d, 0.02)
        mw_d = 0.5  # Raise minimum wage to help poor

    if obs.inflation > 0.06:
        ir_d = 0.02
        ubi_d = min(ubi_d, -0.5)
        ms_d = -100.0  # Tighten money supply
    elif obs.inflation < 0.02:
        ir_d = -0.01
        ms_d = 50.0   # Mild QE

    if obs.unemployment > 0.15:
        pg_d = 0.03
        stim = 100.0
        ms_d = max(ms_d, 80.0)  # QE to boost economy

    return PolicyAction(
        tax_delta         = float(max(-0.10, min(0.10, tax_d))),
        ubi_delta         = float(max(-10.0, min(10.0, ubi_d))),
        public_good_delta = float(max(-0.10, min(0.10, pg_d))),
        interest_rate_delta = float(max(-0.03, min(0.03, ir_d))),
        stimulus_package  = float(max(0.0, min(500.0, stim))),
        import_tariff_delta = float(max(-0.05, min(0.05, tariff_d))),
        money_supply_delta = float(max(-500.0, min(500.0, ms_d))),
        minimum_wage_delta = float(max(-2.0, min(2.0, mw_d))),
        reasoning         = "Intelligent rule-based fallback (8 levers)",
    )


def run_task(task_id: str) -> dict:
    env     = SocialContractOpenEnv(task_id=task_id)
    obs     = env.reset(seed=42)
    history = []
    total_reward = 0.0
    step    = 0
    last_raw_action = None
    max_steps = env.task_cfg["max_steps"]

    print(
        f"[START] task={task_id} difficulty={env.task_cfg['difficulty']} "
        f"max_steps={max_steps} "
        f"gdp={obs.gdp:.2f} gini={obs.gini:.4f} "
        f"gov_budget={obs.gov_budget:.2f} tax_rate={obs.tax_rate}",
        flush=True
    )

    done = False
    while not done:
        prompt = obs.to_prompt()
        history_summary = build_history_summary(history)

        raw_action = call_llm(prompt, task_id, history_summary,
                              step_num=step, max_steps=max_steps)

        if raw_action is None:
            raw_action = fallback_action(obs)

        # Smooth action to avoid volatility penalty
        shock_occurring = (obs.shock_event != "none")
        action = smooth_action(raw_action, last_raw_action, shock_occurring)
        last_raw_action = raw_action

        # Hard floor for task3 — prevent catastrophic gini > 0.88 termination (score=0)
        if task_id == "task3_crisis" and obs.gini > 0.80:
            action = PolicyAction(
                tax_delta         = max(action.tax_delta, 0.05),
                ubi_delta         = max(action.ubi_delta, 3.0),
                public_good_delta = max(action.public_good_delta, 0.02),
                interest_rate_delta = action.interest_rate_delta,
                stimulus_package  = max(action.stimulus_package, 50.0),
                import_tariff_delta = action.import_tariff_delta,
                money_supply_delta = action.money_supply_delta,
                minimum_wage_delta = max(action.minimum_wage_delta, 0.5),
                reasoning         = "EMERGENCY: Gini > 0.80 — aggressive redistribution to avoid score=0",
            )

        # Hard floor for task4 — fight inflation if it's too high
        if task_id == "task4_stagflation" and obs.inflation > 0.10:
            action = PolicyAction(
                tax_delta         = action.tax_delta,
                ubi_delta         = min(action.ubi_delta, -1.0),
                public_good_delta = action.public_good_delta,
                interest_rate_delta = max(action.interest_rate_delta, 0.02),
                stimulus_package  = 0.0,
                import_tariff_delta = action.import_tariff_delta,
                money_supply_delta = min(action.money_supply_delta, -100.0),
                minimum_wage_delta = action.minimum_wage_delta,
                reasoning         = "EMERGENCY: Inflation > 0.10 — Volcker tightening",
            )

        obs = env.step(action)
        history.append(obs.metadata)
        total_reward += obs.reward or 0.0
        step += 1
        rbd = obs.metadata.get("reward_breakdown", {})

        print(
            f"[STEP] task={task_id} step={step} "
            f"reward={obs.reward:.4f} task_progress={rbd.get('task_progress', 0):.4f} "
            f"gdp={obs.gdp:.2f} gini={obs.gini:.4f} "
            f"unrest={obs.unrest:.4f} satisfaction={obs.satisfaction:.4f} "
            f"gov_budget={obs.gov_budget:.2f} tax_rate={obs.tax_rate:.4f} "
            f"inflation={obs.inflation:.4f} unemployment={obs.unemployment:.4f} "
            f"interest_rate={obs.interest_rate:.4f} "
            f"consumer_confidence={obs.consumer_confidence:.4f} "
            f"action_tax_delta={action.tax_delta:.4f} "
            f"action_ubi_delta={action.ubi_delta:.4f} "
            f"action_public_good_delta={action.public_good_delta:.4f} "
            f"action_interest_rate_delta={action.interest_rate_delta:.4f} "
            f"action_stimulus={action.stimulus_package:.1f} "
            f"action_money_supply={action.money_supply_delta:.1f} "
            f"action_min_wage={action.minimum_wage_delta:.2f}",
            flush=True
        )

        done = obs.done

        # Rate-limit protection: brief pause between API calls
        time.sleep(0.3)

    grade = run_grader(task_id, history)

    print(
        f"[END] task={task_id} score={grade.get('score', 0.0):.4f} "
        f"steps={step} total_reward={total_reward:.4f} "
        f"verdict={grade.get('verdict', 'unknown')}",
        flush=True
    )

    return grade


def main():
    for task_id in TASKS:
        run_task(task_id)


if __name__ == "__main__":
    main()