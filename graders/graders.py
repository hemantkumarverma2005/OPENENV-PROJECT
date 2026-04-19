"""
graders.py
──────────
Programmatic graders for all 5 tasks.
Each grader takes the full episode history and returns a score in [0.0, 1.0].
Scores are deterministic given the same history.

Grading principles:
  - Coherent policies (consistent direction) score higher than chaotic ones
  - Sustained improvement across trajectory is rewarded over lucky final values
  - Policy volatility/oscillation is heavily penalised
  - All tasks require trajectory-level consistency, not just snapshot metrics
  - Expert tasks include PHASE DETECTION: did the agent correctly shift
    strategy at the right time? (e.g., pandemic: stabilise → grow → consolidate)
"""

import numpy as np
from typing import Any


# ── Shared helpers ────────────────────────────────────────────────────────────

def _compute_volatility(history: list[dict[str, Any]]) -> float:
    """
    Volatility + oscillation penalty targeting random/chaotic policies.

    Measures three signals:
      1. Direction reversal frequency — penalises flip-flopping
      2. Action magnitude shifts      — penalises erratic deltas
      3. Action variance              — penalises inconsistent strategy

    Returns penalty in [0.0, 0.35].
    """
    if len(history) < 3:
        return 0.0

    actions = [step.get("action", {}) for step in history]
    tax_d = [a.get("tax_delta", 0.0) for a in actions]
    ubi_d = [a.get("ubi_delta", 0.0) for a in actions]
    pg_d  = [a.get("public_good_delta", 0.0) for a in actions]
    ir_d  = [a.get("interest_rate_delta", 0.0) for a in actions]
    ms_d  = [a.get("money_supply_delta", 0.0) for a in actions]
    mw_d  = [a.get("minimum_wage_delta", 0.0) for a in actions]

    # 1. Direction reversal frequency
    def _rev_rate(vals):
        revs = sum(1 for i in range(1, len(vals)) if vals[i] * vals[i - 1] < -1e-8)
        return revs / max(len(vals) - 1, 1)

    rev = (_rev_rate(tax_d) + _rev_rate(ubi_d) + _rev_rate(pg_d)
           + _rev_rate(ir_d) + _rev_rate(ms_d) + _rev_rate(mw_d)) / 6.0

    # 2. Consecutive action magnitude shifts
    shifts = []
    for i in range(1, len(actions)):
        s = (abs(tax_d[i] - tax_d[i - 1])
             + abs(ubi_d[i] - ubi_d[i - 1]) / 10.0
             + abs(pg_d[i] - pg_d[i - 1])
             + abs(ir_d[i] - ir_d[i - 1]) / 0.03 * 0.3
             + abs(ms_d[i] - ms_d[i - 1]) / 500.0 * 0.2
             + abs(mw_d[i] - mw_d[i - 1]) / 2.0 * 0.15)
        shifts.append(s)
    mean_shift = float(np.mean(shifts)) if shifts else 0.0

    # 3. Action variance (std-dev normalised)
    tax_var = float(np.std(tax_d)) * 5.0
    ubi_var = float(np.std(ubi_d)) / 5.0
    pg_var  = float(np.std(pg_d)) * 5.0
    ir_var  = float(np.std(ir_d)) * 10.0
    ms_var  = float(np.std(ms_d)) / 200.0
    mw_var  = float(np.std(mw_d)) * 2.0
    act_var = (tax_var + ubi_var + pg_var + ir_var + ms_var + mw_var) / 6.0

    penalty = float(np.clip(
        rev * 0.18 + mean_shift * 0.35 + act_var * 0.30,
        0.0, 0.35,
    ))
    return penalty


def _metric_consistency(values: list[float], direction: str) -> float:
    """
    Fraction of steps where the metric moves in the desired direction.
    direction: 'decrease' or 'increase'.
    """
    if len(values) < 2:
        return 0.5
    good = 0
    for i in range(1, len(values)):
        if direction == "decrease" and values[i] <= values[i - 1] + 1e-6:
            good += 1
        elif direction == "increase" and values[i] >= values[i - 1] - 1e-6:
            good += 1
    return good / (len(values) - 1)


def _trend_bonus(values: list[float], direction: str) -> float:
    """Bonus in [0.0, 0.12] for consistently improving trends (>55 %)."""
    rate = _metric_consistency(values, direction)
    if rate > 0.55:
        return float(np.clip((rate - 0.55) * 0.28, 0.0, 0.12))
    return 0.0


def _detect_phase_strategy(history: list[dict[str, Any]],
                            n_phases: int = 3) -> dict:
    """
    Detect whether the agent executed a coherent phased strategy.
    Splits history into n_phases equal segments and checks whether
    the policy direction shifted appropriately between phases.

    Returns:
        phase_score: float in [0.0, 0.15]
        phase_details: dict with per-phase analysis
    """
    n = len(history)
    if n < n_phases * 3:
        return {"phase_score": 0.0, "phase_details": {}}

    phase_size = n // n_phases
    phases = []
    for i in range(n_phases):
        start = i * phase_size
        end = (i + 1) * phase_size if i < n_phases - 1 else n
        phase_actions = [h.get("action", {}) for h in history[start:end]]
        avg_tax   = float(np.mean([a.get("tax_delta", 0) for a in phase_actions]))
        avg_ubi   = float(np.mean([a.get("ubi_delta", 0) for a in phase_actions]))
        avg_pg    = float(np.mean([a.get("public_good_delta", 0) for a in phase_actions]))
        avg_stim  = float(np.mean([a.get("stimulus_package", 0) for a in phase_actions]))
        phases.append({
            "avg_tax_delta": avg_tax,
            "avg_ubi_delta": avg_ubi,
            "avg_pg_delta": avg_pg,
            "avg_stimulus": avg_stim,
        })

    # Check for strategy shift between phases
    # A good phased strategy should show DIFFERENT policy directions in each phase
    shifts = 0
    total_checks = 0
    for i in range(1, len(phases)):
        # Tax direction changed?
        if abs(phases[i]["avg_tax_delta"] - phases[i-1]["avg_tax_delta"]) > 0.005:
            shifts += 1
        total_checks += 1
        # UBI direction changed?
        if abs(phases[i]["avg_ubi_delta"] - phases[i-1]["avg_ubi_delta"]) > 0.3:
            shifts += 1
        total_checks += 1
        # PG direction changed?
        if abs(phases[i]["avg_pg_delta"] - phases[i-1]["avg_pg_delta"]) > 0.005:
            shifts += 1
        total_checks += 1

    shift_rate = shifts / max(total_checks, 1)
    # Phase score: reward having distinct phases, but not too many reversals
    phase_score = float(np.clip(shift_rate * 0.20, 0.0, 0.15))

    return {
        "phase_score": phase_score,
        "phase_details": {f"phase_{i+1}": p for i, p in enumerate(phases)},
        "shift_rate": round(shift_rate, 4),
    }


# ── Task 1 — Easy: Maintain Economic Stability ───────────────────────────────

def grade_task1_stability(history: list[dict[str, Any]]) -> dict:
    """Task 1 — Easy: Maintain a healthy economy."""
    if not history:
        return {"score": 0.0, "breakdown": {}, "verdict": "No steps taken"}

    gdp_scores, gini_scores, unrest_scores, budget_scores = [], [], [], []
    prev_gdp = None
    for step in history:
        gdp = step.get("gdp", 0)
        if prev_gdp is not None:
            g = (gdp - prev_gdp) / max(prev_gdp, 1)
            gdp_scores.append(1.0 if g >= 0 else max(0.0, 1.0 + g * 10))
        prev_gdp = gdp
        gini = step.get("gini", 1.0)
        gini_scores.append(float(np.clip(1.0 - max(0, gini - 0.43) / 0.22, 0, 1)))
        unrest = step.get("unrest", 1.0)
        unrest_scores.append(float(np.clip(1.0 - unrest / 0.30, 0, 1)))
        budget_scores.append(1.0 if step.get("gov_budget", -1) >= 0 else 0.0)

    gdp_s    = float(np.mean(gdp_scores))    if gdp_scores else 0.0
    gini_s   = float(np.mean(gini_scores))
    unrest_s = float(np.mean(unrest_scores))
    budget_s = float(np.mean(budget_scores))

    # Consistency bonuses (smart agents keep metrics stable)
    gini_vals = [s.get("gini", 1.0) for s in history]
    gini_stable = 1.0 - float(np.std(gini_vals)) * 5.0
    gini_stable = float(np.clip(gini_stable, 0.0, 1.0))

    vol_pen = _compute_volatility(history)

    raw = (0.35 * gdp_s + 0.25 * gini_s + 0.20 * unrest_s
           + 0.10 * budget_s + 0.10 * gini_stable)
    final = float(np.clip(raw - vol_pen, 0.0, 1.0))

    verdict = (
        "Excellent — economy fully maintained" if final >= 0.80 else
        "Good — minor deterioration"           if final >= 0.60 else
        "Partial — significant issues arose"   if final >= 0.35 else
        "Failed — economy deteriorated badly"
    )
    return {
        "score":     round(final, 4),
        "breakdown": {
            "gdp_growth_score":   round(gdp_s, 4),
            "gini_score":         round(gini_s, 4),
            "unrest_score":       round(unrest_s, 4),
            "budget_score":       round(budget_s, 4),
            "gini_stability":     round(gini_stable, 4),
            "volatility_penalty": round(vol_pen, 4),
        },
        "verdict":   verdict,
        "steps_run": len(history),
    }


# ── Task 2 — Medium: Recession Recovery ──────────────────────────────────────

def grade_task2_recession(history: list[dict[str, Any]]) -> dict:
    """Task 2 — Medium: Recover from recession."""
    if not history:
        return {"score": 0.0, "breakdown": {}, "verdict": "No steps taken"}

    n = len(history)

    # Budget: reward CONSISTENT improvement, not lucky final values
    budgets = [s.get("gov_budget", -2500) for s in history]
    budget_improving = sum(1 for i in range(1, len(budgets)) if budgets[i] > budgets[i - 1])
    budget_consistency = budget_improving / max(len(budgets) - 1, 1)
    final_budget   = budgets[-1]
    final_budget_ok = float(np.clip((final_budget + 2500) / 2500, 0, 1))
    budget_score    = 0.60 * budget_consistency + 0.40 * final_budget_ok

    # GDP: growth trajectory + absolute recovery
    gdp_vals   = [s.get("gdp", 0) for s in history]
    gdp_growth = [1.0 if gdp_vals[i] > gdp_vals[i - 1] else 0.0
                  for i in range(1, len(gdp_vals))]
    growth_frac  = float(np.mean(gdp_growth)) if gdp_growth else 0.0
    abs_recovery = float(np.clip((gdp_vals[-1] - gdp_vals[0]) / max(gdp_vals[0], 1), 0, 1))
    gdp_score    = 0.6 * growth_frac + 0.4 * abs_recovery

    # Unrest control
    unrest_vals  = [s.get("unrest", 1.0) for s in history]
    unrest_score = float(np.mean([np.clip(1.0 - u / 0.40, 0, 1) for u in unrest_vals]))

    # Speed bonus
    recovery_step = next((i for i, b in enumerate(budgets) if b >= 0), n)
    speed_bonus   = float(np.clip(1.0 - recovery_step / n, 0, 0.10))

    # Trend bonuses (smart agents show consistent improvement)
    budget_trend = _trend_bonus(budgets, "increase")
    gdp_trend    = _trend_bonus(gdp_vals, "increase")

    vol_pen = _compute_volatility(history)

    raw = (0.35 * budget_score + 0.30 * gdp_score + 0.25 * unrest_score
           + speed_bonus + budget_trend + gdp_trend)
    final = float(np.clip(raw - vol_pen, 0.0, 1.0))

    verdict = (
        "Excellent — full recovery achieved"      if final >= 0.75 else
        "Good — substantial recovery made"        if final >= 0.55 else
        "Partial — some improvement but incomplete" if final >= 0.30 else
        "Failed — recession not addressed"
    )
    return {
        "score":     round(final, 4),
        "breakdown": {
            "budget_score":       round(budget_score, 4),
            "gdp_score":          round(gdp_score, 4),
            "unrest_score":       round(unrest_score, 4),
            "speed_bonus":        round(speed_bonus, 4),
            "budget_trend":       round(budget_trend, 4),
            "gdp_trend":          round(gdp_trend, 4),
            "volatility_penalty": round(vol_pen, 4),
        },
        "verdict":   verdict,
        "steps_run": n,
    }


# ── Task 3 — Hard: Inequality Crisis Resolution ──────────────────────────────

def grade_task3_crisis(history: list[dict[str, Any]]) -> dict:
    """Task 3 — Hard: Resolve inequality crisis."""
    if not history:
        return {"score": 0.0, "breakdown": {}, "verdict": "No steps taken"}

    n = len(history)

    gini_vals = [s.get("gini", 1.0) for s in history]
    f_gini    = gini_vals[-1]
    avg_gini  = float(np.mean(gini_vals))
    gini_score = float(np.clip(1.0 - max(0, avg_gini - 0.38) / 0.32, 0, 1))
    gini_gate  = f_gini < 0.48

    sustain_w  = max(5, n // 4)
    recent_unr = [s.get("unrest", 1.0) for s in history[-sustain_w:]]
    unrest_sustained = all(u < 0.10 for u in recent_unr)
    unrest_score = float(np.mean([
        np.clip(1.0 - u / 0.10, 0, 1) for u in [s.get("unrest", 1.0) for s in history]
    ]))

    gdp_vals   = [s.get("gdp", 0) for s in history]
    gdp_growth = sum(1 for i in range(1, len(gdp_vals)) if gdp_vals[i] > gdp_vals[i - 1])
    gdp_frac   = gdp_growth / max(len(gdp_vals) - 1, 1)
    gdp_score  = float(np.clip((gdp_frac - 0.30) / 0.60, 0, 1))
    gdp_gate   = gdp_frac > 0.70

    # Consistency: Gini should be decreasing
    gini_consistency = _metric_consistency(gini_vals, "decrease")
    gini_trend       = _trend_bonus(gini_vals, "decrease")

    all_achieved       = gini_gate and unrest_sustained and gdp_gate
    simultaneous_bonus = 0.10 if all_achieved else 0.0
    vol_pen = _compute_volatility(history)

    raw = (0.35 * gini_score + 0.30 * unrest_score + 0.20 * gdp_score
           + 0.15 * gini_consistency + gini_trend)
    cap   = 1.0 if all_achieved else 0.70
    score = float(np.clip(raw + simultaneous_bonus - vol_pen, 0.0, cap))

    verdict = (
        "Excellent — crisis fully resolved and sustained" if all_achieved and score >= 0.85 else
        "Good — all three metrics resolved"               if all_achieved else
        "Partial — metrics improved but not sustained"    if score >= 0.35 else
        "Failed — crisis unresolved"
    )
    return {
        "score":     round(score, 4),
        "breakdown": {
            "gini_score":          round(gini_score, 4),
            "unrest_score":        round(unrest_score, 4),
            "gdp_score":           round(gdp_score, 4),
            "gini_consistency":    round(gini_consistency, 4),
            "gini_trend":          round(gini_trend, 4),
            "simultaneous_bonus":  round(simultaneous_bonus, 4),
            "volatility_penalty":  round(vol_pen, 4),
            "all_achieved":        all_achieved,
        },
        "verdict":   verdict,
        "steps_run": n,
    }


# ── Task 4 — Expert: Stagflation Crisis ──────────────────────────────────────

def grade_task4_stagflation(history: list[dict[str, Any]]) -> dict:
    """
    Task 4 — Expert: Resolve stagflation crisis.

    Every conventional tool has conflicting effects — raising taxes fights
    inflation but deepens recession; cutting UBI reduces deficit but raises
    unrest.  Scoring rewards *consistent, sustained* improvement rather than
    lucky final snapshots.

    Phase detection: expects 2-phase strategy:
      Phase 1 (early): Inflation control (higher interest rates, tighter fiscal)
      Phase 2 (later): Growth recovery (lower rates, public goods investment)
    """
    if not history:
        return {"score": 0.0, "breakdown": {}, "verdict": "No steps taken"}

    n = len(history)

    # GDP
    gdp_vals   = [s.get("gdp", 0) for s in history]
    gdp_growth = sum(1 for i in range(1, len(gdp_vals)) if gdp_vals[i] > gdp_vals[i - 1])
    gdp_frac   = gdp_growth / max(len(gdp_vals) - 1, 1)
    abs_rec    = float(np.clip((gdp_vals[-1] - gdp_vals[0]) / max(gdp_vals[0], 1), 0, 1))
    gdp_score  = 0.5 * float(np.clip((gdp_frac - 0.30) / 0.50, 0, 1)) + 0.5 * abs_rec

    # Inflation — consistency of reduction matters most
    inf_vals  = [s.get("inflation", 0.12) for s in history]
    inf_final = inf_vals[-1]
    inf_avg   = float(np.mean(inf_vals))
    inf_traj  = float(np.clip(1.0 - max(0, inf_avg - 0.03) / 0.12, 0, 1))
    inf_end   = float(np.clip(1.0 - max(0, inf_final - 0.03) / 0.12, 0, 1))
    inf_cons  = _metric_consistency(inf_vals, "decrease")
    inflation_score = 0.25 * inf_traj + 0.35 * inf_end + 0.40 * inf_cons
    inflation_gate  = inf_final < 0.04

    # Unemployment — consistency of reduction
    unemp_vals  = [s.get("unemployment", 0.22) for s in history]
    unemp_final = unemp_vals[-1]
    unemp_avg   = float(np.mean(unemp_vals))
    unemp_traj  = float(np.clip(1.0 - max(0, unemp_avg - 0.06) / 0.18, 0, 1))
    unemp_end   = float(np.clip(1.0 - max(0, unemp_final - 0.06) / 0.18, 0, 1))
    unemp_cons  = _metric_consistency(unemp_vals, "decrease")
    unemp_score = 0.25 * unemp_traj + 0.35 * unemp_end + 0.40 * unemp_cons
    unemp_gate  = unemp_final < 0.08

    # Budget
    budgets      = [s.get("gov_budget", -1200) for s in history]
    final_budget = budgets[-1]
    budget_cons  = _metric_consistency(budgets, "increase")
    budget_val   = float(np.clip((final_budget + 1200) / 1200, 0, 1))
    budget_score = 0.50 * budget_cons + 0.50 * budget_val
    budget_gate  = final_budget > -200

    gdp_gate = gdp_frac > 0.60

    all_achieved        = inflation_gate and unemp_gate and budget_gate and gdp_gate
    simultaneous_bonus  = 0.15 if all_achieved else 0.0

    # Phase detection: 2-phase strategy (inflation control → growth recovery)
    phase_info = _detect_phase_strategy(history, n_phases=2)
    phase_bonus = phase_info["phase_score"]

    # Trend bonuses
    inf_trend   = _trend_bonus(inf_vals, "decrease")
    unemp_trend = _trend_bonus(unemp_vals, "decrease")

    vol_pen = _compute_volatility(history)

    raw = (0.25 * gdp_score + 0.25 * inflation_score
           + 0.25 * unemp_score + 0.15 * budget_score
           + 0.10 * (inf_cons + unemp_cons) / 2.0
           + inf_trend + unemp_trend + phase_bonus)
    cap   = 1.0 if all_achieved else 0.60
    score = float(np.clip(raw + simultaneous_bonus - vol_pen, 0.0, cap))

    verdict = (
        "Masterful — stagflation fully resolved" if all_achieved and score >= 0.85 else
        "Excellent — all four metrics achieved"  if all_achieved else
        "Partial — some metrics improved"        if score >= 0.25 else
        "Failed — stagflation persists"
    )
    return {
        "score":     round(score, 4),
        "breakdown": {
            "gdp_score":          round(gdp_score, 4),
            "inflation_score":    round(inflation_score, 4),
            "unemployment_score": round(unemp_score, 4),
            "budget_score":       round(budget_score, 4),
            "inflation_consistency": round(inf_cons, 4),
            "unemployment_consistency": round(unemp_cons, 4),
            "simultaneous_bonus": round(simultaneous_bonus, 4),
            "phase_strategy_bonus": round(phase_bonus, 4),
            "volatility_penalty": round(vol_pen, 4),
            "all_achieved":       all_achieved,
            "inflation_gate":     inflation_gate,
            "unemployment_gate":  unemp_gate,
            "budget_gate":        budget_gate,
            "gdp_gate":           gdp_gate,
            "final_inflation":    round(inf_final, 4),
            "final_unemployment": round(unemp_final, 4),
            "final_budget":       round(final_budget, 1),
        },
        "verdict":   verdict,
        "steps_run": n,
    }


# ── Task 5 — Expert: Pandemic Economic Response ──────────────────────────────

def grade_task5_pandemic(history: list[dict[str, Any]]) -> dict:
    """
    Task 5 — Expert: Pandemic economic response (COVID-19-style).

    Requires phased recovery: emergency stabilisation → growth restart →
    fiscal consolidation.  Scoring rewards *active reduction* of unemployment
    and *consistent trajectory improvement*, not just lucky final values.

    Phase detection: expects 3-phase strategy:
      Phase 1 (steps 0-9): Emergency spending (high UBI, public goods, stimulus)
      Phase 2 (steps 10-19): Recovery (grow GDP, manage inflation)
      Phase 3 (steps 20-29): Consolidation (reduce deficit, tighten policy)
    """
    if not history:
        return {"score": 0.0, "breakdown": {}, "verdict": "No steps taken"}

    n = len(history)

    # Unemployment — must show ACTIVE reduction from 0.25
    unemp_vals  = [s.get("unemployment", 0.25) for s in history]
    unemp_final = unemp_vals[-1]
    unemp_avg   = float(np.mean(unemp_vals))
    unemp_traj  = float(np.clip(1.0 - max(0, unemp_avg - 0.06) / 0.22, 0, 1))
    unemp_end   = float(np.clip(1.0 - max(0, unemp_final - 0.06) / 0.22, 0, 1))
    unemp_reduction = float(np.clip((0.25 - unemp_final) / 0.20, 0, 1))
    unemp_cons  = _metric_consistency(unemp_vals, "decrease")
    unemp_score = (0.20 * unemp_traj + 0.25 * unemp_end
                   + 0.25 * unemp_reduction + 0.30 * unemp_cons)
    unemp_gate  = unemp_final < 0.08

    # GDP
    gdp_vals   = [s.get("gdp", 0) for s in history]
    gdp_growth = sum(1 for i in range(1, len(gdp_vals)) if gdp_vals[i] > gdp_vals[i - 1])
    gdp_frac   = gdp_growth / max(len(gdp_vals) - 1, 1)
    abs_rec    = float(np.clip((gdp_vals[-1] - gdp_vals[0]) / max(gdp_vals[0], 1), 0, 1))
    gdp_score  = 0.5 * float(np.clip((gdp_frac - 0.25) / 0.55, 0, 1)) + 0.5 * abs_rec

    # Inflation — tricky because stimulus causes delayed inflation
    inf_vals  = [s.get("inflation", 0.01) for s in history]
    inf_final = inf_vals[-1]
    inf_avg   = float(np.mean(inf_vals))
    inf_traj  = float(np.clip(1.0 - max(0, inf_avg - 0.04) / 0.10, 0, 1))
    inf_end   = float(np.clip(1.0 - max(0, inf_final - 0.04) / 0.10, 0, 1))
    inf_cons  = _metric_consistency(inf_vals, "decrease")
    inflation_score = 0.30 * inf_traj + 0.30 * inf_end + 0.40 * inf_cons
    inflation_gate  = inf_final < 0.06

    # Budget
    budgets      = [s.get("gov_budget", -3000) for s in history]
    final_budget = budgets[-1]
    budget_cons  = _metric_consistency(budgets, "increase")
    budget_val   = float(np.clip((final_budget + 3000) / 3000, 0, 1))
    budget_score = 0.50 * budget_cons + 0.50 * budget_val
    budget_gate  = final_budget > -500

    gdp_gate = gdp_frac > 0.55

    all_achieved        = unemp_gate and inflation_gate and budget_gate and gdp_gate
    simultaneous_bonus  = 0.15 if all_achieved else 0.0

    # Phase detection: 3-phase strategy (emergency → recovery → consolidation)
    phase_info = _detect_phase_strategy(history, n_phases=3)
    phase_bonus = phase_info["phase_score"]

    # Check specific phase behavior for pandemic
    pandemic_phase_bonus = 0.0
    if n >= 9:
        # Phase 1 (early): should use stimulus and increase spending
        early_actions = [h.get("action", {}) for h in history[:n//3]]
        early_spending = float(np.mean([
            a.get("ubi_delta", 0) + a.get("public_good_delta", 0) * 10 + a.get("stimulus_package", 0) / 50
            for a in early_actions
        ]))
        # Phase 3 (late): should consolidate (reduce spending, increase taxes)
        late_actions = [h.get("action", {}) for h in history[-(n//3):]]
        late_consolidation = float(np.mean([
            a.get("tax_delta", 0) - a.get("ubi_delta", 0) / 10
            for a in late_actions
        ]))
        if early_spending > 0.5:  # Was spending in early phase
            pandemic_phase_bonus += 0.03
        if late_consolidation > 0:  # Was consolidating in late phase
            pandemic_phase_bonus += 0.03
    pandemic_phase_bonus = float(np.clip(pandemic_phase_bonus, 0.0, 0.06))

    # Trend bonuses
    unemp_trend = _trend_bonus(unemp_vals, "decrease")
    budget_trend = _trend_bonus(budgets, "increase")

    vol_pen = _compute_volatility(history)

    raw = (0.30 * unemp_score + 0.20 * gdp_score
           + 0.20 * inflation_score + 0.15 * budget_score
           + 0.15 * (unemp_cons + budget_cons) / 2.0
           + unemp_trend + budget_trend + phase_bonus + pandemic_phase_bonus)
    cap   = 1.0 if all_achieved else 0.55
    score = float(np.clip(raw + simultaneous_bonus - vol_pen, 0.0, cap))

    verdict = (
        "Masterful — pandemic recovery fully achieved" if all_achieved and score >= 0.85 else
        "Excellent — all four metrics achieved"        if all_achieved else
        "Partial — some recovery achieved"             if score >= 0.20 else
        "Failed — economy still in pandemic crisis"
    )
    return {
        "score":     round(score, 4),
        "breakdown": {
            "unemployment_score":       round(unemp_score, 4),
            "gdp_score":                round(gdp_score, 4),
            "inflation_score":          round(inflation_score, 4),
            "budget_score":             round(budget_score, 4),
            "unemployment_consistency": round(unemp_cons, 4),
            "budget_consistency":       round(budget_cons, 4),
            "simultaneous_bonus":       round(simultaneous_bonus, 4),
            "phase_strategy_bonus":     round(phase_bonus, 4),
            "pandemic_phase_bonus":     round(pandemic_phase_bonus, 4),
            "volatility_penalty":       round(vol_pen, 4),
            "all_achieved":             all_achieved,
            "unemployment_gate":        unemp_gate,
            "inflation_gate":           inflation_gate,
            "budget_gate":              budget_gate,
            "gdp_gate":                 gdp_gate,
            "final_unemployment":       round(unemp_final, 4),
            "final_inflation":          round(inf_final, 4),
            "final_budget":             round(final_budget, 1),
        },
        "verdict":   verdict,
        "steps_run": n,
    }


# ── Registry ─────────────────────────────────────────────────────────────────

GRADERS = {
    "task1_stability":   grade_task1_stability,
    "task2_recession":   grade_task2_recession,
    "task3_crisis":      grade_task3_crisis,
    "task4_stagflation": grade_task4_stagflation,
    "task5_pandemic":    grade_task5_pandemic,
}


def run_grader(task_id: str, history: list[dict]) -> dict:
    grader = GRADERS.get(task_id)
    if grader is None:
        return {"score": 0.0, "verdict": f"Unknown task: {task_id}"}
    result = grader(history)
    assert 0.0 <= result["score"] <= 1.0, "Grader must return score in [0.0, 1.0]"
    return result