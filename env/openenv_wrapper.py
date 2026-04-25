"""
openenv_wrapper.py
──────────────────
OpenEnv-compliant interface for SocialContract-v0.
Implements: reset(), step(), state()
All inputs/outputs use typed Pydantic models.

Academic foundations:
  - Phillips Curve: inflation = f(unemployment, expected_inflation, money_supply)
    (Phillips 1958, augmented by Phelps 1967 and Friedman 1968)
  - Okun's Law: GDP gap ≈ -2 × unemployment gap (Okun 1962)
  - Taylor Rule: interest rate targets inflation and output gaps (Taylor 1993)
  - Solow Growth Model: savings → investment → capital accumulation (Solow 1956)
  - Laffer Curve: tax revenue peaks at interior rate (Trabandt & Uhlig 2011)
  - Barro-Gordon: policy credibility affects outcomes (Barro & Gordon 1983)
  - Cantillon Effect: money supply expansion reaches rich first (Cantillon 1755)
  - Stackelberg Game: rich invest → poor respond with labor (Von Stackelberg 1934)

Calibration sources:
  - OECD Economic Outlook (2023), IMF World Economic Outlook (April 2024)
  - World Bank Development Indicators, Federal Reserve FRED Database
  - BLS Productivity Statistics, Case-Shiller Home Price Index
"""

import numpy as np
from typing import Any, Optional
import base64
import io
from env.citizens import (CitizenGroup, sample_shock, compute_shock_effect,
                           phillips_curve_inflation, okuns_law_gdp_effect, NAIRU)
from env.pydantic_models import EconomicObservation, PolicyAction, StepReward
from env.market_agent import MarketConsortium
from env.curriculum import AdaptiveCurriculum, DifficultyConfig, DIFFICULTY_LEVELS

from openenv.core.env_server.interfaces import Environment as _SDKEnvironment
from openenv.core.env_server.types import State

MAX_STEPS   = 40
MAX_BUDGET  = 5000.0
MAX_UBI     = 50.0


class SocialContractOpenEnv(_SDKEnvironment):
    """
    OpenEnv-compliant Economic Policy Advisory Environment.
    Inherits from openenv-core SDK Environment base class.

    An LLM agent acts as a government policy advisor.
    Each step it receives an economic report and must recommend
    policy adjustments across 8 levers to achieve the current task objective.

    Inter-group dynamics:
      - Wealthy classes invest → creates jobs for poor/middle (Stackelberg)
      - Ultra-rich capital flight when trust < 0.6 or taxes > 0.50 (Collier 2001)
      - Poor/middle coordinate strikes when satisfaction < 0.25 (Olson 1965)
      - Money supply (QE) inflates rich assets but erodes poor purchasing power
      - Minimum wage raises floor income but may reduce employment
    """

    # SDK: each WebSocket session gets its own isolated env instance
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    TASKS = {
        "task1_stability": {
            "description": (
                "IMF ARTICLE IV CONSULTATION — Steady-State Maintenance: "
                "You are advising a small open economy (comparable to Estonia or New Zealand, "
                "GDP ~$30B, Gini ~0.33). Current indicators are ostensibly healthy, but the "
                "fiscal balance is fragile — similar to pre-2008 Ireland. Recommend quarterly "
                "adjustments to tax rate, UBI, public spending, interest rate, trade policy, "
                "money supply, and minimum wage to keep GDP growth positive, the Gini "
                "coefficient below 0.43, social unrest below 0.15, and the government budget "
                "in surplus across all 20 review periods. "
                "Avoid overcorrecting — the IMF's Article IV assessment penalises unnecessary "
                "policy volatility, as real-world markets react negatively to erratic fiscal signals."
            ),
            "max_steps": 20,
            "difficulty": "easy",
        },
        "task2_recession": {
            "description": (
                "WORLD BANK STRUCTURAL ADJUSTMENT — Recession Recovery Programme: "
                "The economy has entered recession comparable to Greece 2010-2012. GDP is "
                "contracting, the fiscal deficit has reached -2500 (5% of GDP), and unemployment "
                "& social unrest are rising. The IMF-World Bank joint mission requires a "
                "credible recovery roadmap. You have access to monetary policy (interest rates, "
                "QE), fiscal stimulus, trade policy, and minimum wage in addition to standard "
                "levers. Recommend policy adjustments each quarter to restore positive GDP "
                "growth, bring the budget back to surplus, and contain unrest below 0.40 — "
                "all within 30 review periods. "
                "Speed of recovery will be assessed: historically, successful recoveries "
                "(e.g., South Korea 1998) achieved turnaround in 6-8 quarters."
            ),
            "max_steps": 30,
            "difficulty": "medium",
        },
        "task3_crisis": {
            "description": (
                "UN DEVELOPMENT PROGRAMME — Inequality Shock Response: "
                "A sudden wealth shock (comparable to post-Soviet Russia 1992 or post-apartheid "
                "South Africa) has tripled ultra-high-net-worth holdings while halving low-income "
                "household wealth. The Gini index has surged to crisis levels (>0.60, exceeding "
                "Brazil's historic peak). Social unrest mirrors Arab Spring conditions. "
                "The Finance Ministry requires a redistribution strategy that simultaneously "
                "reduces the Gini below 0.48, brings unrest below 0.10, and preserves positive "
                "GDP growth — all three must be achieved concurrently within 40 review periods. "
                "CAUTION: Ultra-rich capital flight is a risk — aggressive taxation without trust "
                "building may trigger wealth exodus. Minimum wage adjustments can help the poor. "
                "History shows that addressing inequality without destroying growth is the "
                "central challenge of development economics (Kuznets curve)."
            ),
            "max_steps": 40,
            "difficulty": "hard",
        },
        "task4_stagflation": {
            "description": (
                "CENTRAL BANK EMERGENCY SESSION — Stagflation Crisis (1970s-style): "
                "The economy faces stagflation comparable to the US/UK 1973-1975 oil crisis. "
                "GDP is contracting while inflation has surged to 12% (exceeding the ECB's 2% "
                "target by 6x) and unemployment has spiked to 22%. Central bank orthodoxy "
                "(Volcker doctrine) says fighting inflation requires raising interest rates and "
                "tight monetary policy (QE reversal). But austerity deepens the recession. "
                "The PM demands you simultaneously restore GDP growth, bring inflation below "
                "0.04, reduce unemployment below 0.08, AND keep the budget deficit manageable — "
                "all within 35 review periods. "
                "Use interest rates AND money supply strategically: the Taylor Rule suggests "
                "rates should respond to both inflation and output gaps. Phase your approach."
            ),
            "max_steps": 35,
            "difficulty": "expert",
        },
        "task5_pandemic": {
            "description": (
                "WHO-IMF JOINT EMERGENCY BRIEF — Pandemic Economic Response (COVID-19-style): "
                "A global pandemic has triggered simultaneous supply and demand shocks. GDP has "
                "crashed 15% (comparable to Q2-2020 in most OECD countries), unemployment has "
                "spiked to 25% as businesses close, and emergency fiscal spending has created "
                "a massive deficit. Unlike a normal recession, the pandemic creates recurring "
                "waves that disrupt recovery — modelled on real COVID-19 wave patterns. "
                "You must design a phased recovery using ALL available levers: "
                "(1) stabilise employment via targeted stimulus, UBI, QE, and public goods, "
                "(2) manage the inevitable inflation from stimulus and QE using interest rates, "
                "(3) restore GDP growth as lockdowns ease, "
                "(4) begin fiscal consolidation without triggering a double-dip recession. "
                "BEWARE: Excessive QE will inflate asset prices (Cantillon effect) and may "
                "trigger a worker strike if poor feel left behind."
            ),
            "max_steps": 30,
            "difficulty": "expert",
        },
    }

    def __init__(self, task_id: str = "task1_stability", seed: int = 42,
                 enable_market_agent: bool = False,
                 enable_curriculum: bool = False,
                 difficulty_level: int = 0):
        # Initialise SDK base class first
        try:
            super().__init__()
        except TypeError:
            pass  # fallback when SDK not installed
        assert task_id in self.TASKS, f"Unknown task: {task_id}. Choose from {list(self.TASKS)}"
        self.task_id    = task_id
        self.task_cfg   = self.TASKS[task_id]
        self._seed      = seed
        self._rng       = np.random.default_rng(seed)
        self._step_count = 0
        self._done      = False
        self._history   = []
        self.citizen_groups: list[CitizenGroup] = []
        self.tax_rate          = 0.25
        self.ubi_amount        = 5.0
        self.public_good_level = 0.3
        self.gov_budget        = 500.0
        self.prev_gdp          = 0.0

        self.inflation         = 0.02
        self.unemployment      = 0.05
        self.interest_rate     = 0.05
        self.import_tariff     = 0.0
        self.consumer_confidence = 0.5
        self.money_supply      = 0.0
        self.minimum_wage      = 5.0

        # Persistent shock state
        self.current_shock     = None
        self.shock_remaining   = 0
        self.shock_total_duration = 0

        self.last_action       = None
        self.speeches_used     = 0

        # Multi-agent: Market speculator consortium (#8)
        self.enable_market_agent = enable_market_agent
        self.market_consortium: Optional[MarketConsortium] = None
        self._last_market_response: Optional[dict] = None

        # Curriculum learning: adaptive difficulty (#9)
        self.enable_curriculum = enable_curriculum
        self.curriculum = AdaptiveCurriculum() if enable_curriculum else None
        self.difficulty = DIFFICULTY_LEVELS[min(difficulty_level, len(DIFFICULTY_LEVELS) - 1)]

    @property
    def is_done(self) -> bool:
        return self._done

    def reset(self, seed: int | None = None, **kwargs) -> EconomicObservation:
        # SDK passes POST /reset body fields as kwargs (e.g. task_id)
        new_task = kwargs.get("task_id", None)
        if new_task and new_task in self.TASKS and new_task != self.task_id:
            self.task_id = new_task
            self.task_cfg = self.TASKS[new_task]
        if seed is not None:
            self._seed = seed
        self._rng        = np.random.default_rng(self._seed)
        self._step_count = 0
        self._done       = False
        self._history    = []
        self.current_shock = None
        self.shock_remaining = 0
        self.shock_total_duration = 0
        self.last_action = None
        self.speeches_used = 0
        self.consumer_confidence = 0.5
        self.money_supply = 0.0

        if self.task_id == "task1_stability":
            self.tax_rate, self.ubi_amount = 0.28, 1.0
            self.public_good_level = 0.20
            self.gov_budget   = 600.0
            self.inflation    = 0.02
            self.unemployment = 0.04
            self.interest_rate = 0.04
            self.import_tariff = 0.02
            self.minimum_wage  = 5.0
            wealth_overrides   = {"poor": 35, "middle": 65, "wealthy": 150, "ultra_rich": 450}

        elif self.task_id == "task2_recession":
            self.tax_rate, self.ubi_amount = 0.15, 2.0
            self.public_good_level = 0.05
            self.gov_budget   = -2500.0
            self.inflation    = -0.01
            self.unemployment = 0.18
            self.interest_rate = 0.08
            self.import_tariff = 0.05
            self.minimum_wage  = 4.0
            wealth_overrides   = {"poor": 5, "middle": 25, "wealthy": 100, "ultra_rich": 400}

        elif self.task_id == "task3_crisis":
            self.tax_rate, self.ubi_amount = 0.25, 5.0
            self.public_good_level = 0.2
            self.gov_budget   = 200.0
            self.inflation    = 0.06
            self.unemployment = 0.12
            self.interest_rate = 0.05
            self.import_tariff = 0.0
            self.minimum_wage  = 3.0
            wealth_overrides   = {"poor": 5, "middle": 50, "wealthy": 200, "ultra_rich": 2400}

        elif self.task_id == "task4_stagflation":
            self.tax_rate, self.ubi_amount = 0.35, 8.0
            self.public_good_level = 0.15
            self.gov_budget   = -1200.0
            self.inflation    = 0.12
            self.unemployment = 0.22
            self.interest_rate = 0.06
            self.import_tariff = 0.08
            self.minimum_wage  = 6.0
            wealth_overrides   = {"poor": 6, "middle": 30, "wealthy": 120, "ultra_rich": 500}

        elif self.task_id == "task5_pandemic":
            self.tax_rate, self.ubi_amount = 0.22, 12.0
            self.public_good_level = 0.10
            self.gov_budget   = -3000.0
            self.inflation    = 0.01
            self.unemployment = 0.25
            self.interest_rate = 0.02
            self.import_tariff = 0.03
            self.minimum_wage  = 5.0
            wealth_overrides   = {"poor": 4, "middle": 25, "wealthy": 180, "ultra_rich": 1200}

        self.citizen_groups = [
            CitizenGroup(name, self._rng, wealth_overrides.get(name))
            for name in ["poor", "middle", "wealthy", "ultra_rich"]
        ]

        # Apply difficulty modifiers (curriculum learning)
        if self.difficulty.initial_budget_penalty > 0:
            self.gov_budget -= self.difficulty.initial_budget_penalty
        if self.difficulty.initial_inflation_add > 0:
            self.inflation += self.difficulty.initial_inflation_add
        if self.difficulty.initial_unemployment_add > 0:
            self.unemployment += self.difficulty.initial_unemployment_add

        # Initialize market agent consortium
        if self.enable_market_agent:
            self.market_consortium = MarketConsortium(self._rng)
            self.market_consortium.reset(self._rng)
            self._last_market_response = None

        self.prev_gdp = sum(g.wealth.sum() * 0.05 for g in self.citizen_groups)
        return self._build_obs()

    def step(self, action: PolicyAction, timeout_s: Optional[float] = None, **kwargs) -> EconomicObservation:  # type: ignore[override]
        """Take one policy step. Returns EconomicObservation with .done, .reward, .metadata."""
        # Guard: auto-reset if step() called on a fresh/un-reset instance
        if not self.citizen_groups:
            self.reset(seed=kwargs.get("seed"), task_id=kwargs.get("task_id"))
        if self._done:
            raise RuntimeError("Episode finished — call reset() first")

        # ── Volatility penalty (all 8 levers) ────────────────────────────
        volatility_penalty = 0.0
        if self.last_action is not None:
            shifts = [
                abs(action.tax_delta - self.last_action.tax_delta),
                abs(action.ubi_delta - self.last_action.ubi_delta) / 10.0,
                abs(action.public_good_delta - self.last_action.public_good_delta),
                abs(action.interest_rate_delta - self.last_action.interest_rate_delta) / 0.03 * 0.3,
                abs(action.import_tariff_delta - self.last_action.import_tariff_delta) / 0.05 * 0.2,
                abs(action.money_supply_delta - self.last_action.money_supply_delta) / 500.0 * 0.2,
                abs(action.minimum_wage_delta - self.last_action.minimum_wage_delta) / 2.0 * 0.15,
            ]
            volatility_penalty = float(np.clip(sum(shifts) * 0.4, 0.0, 0.2))
        self.last_action = action

        # ── Apply policy levers ──────────────────────────────────────────
        self.tax_rate          = float(np.clip(self.tax_rate + action.tax_delta, 0.0, 0.8))
        self.ubi_amount        = float(np.clip(self.ubi_amount + action.ubi_delta, 0.0, MAX_UBI))
        self.public_good_level = float(np.clip(self.public_good_level + action.public_good_delta, 0.0, 1.0))
        self.interest_rate     = float(np.clip(self.interest_rate + action.interest_rate_delta, 0.0, 0.20))
        self.import_tariff     = float(np.clip(self.import_tariff + action.import_tariff_delta, 0.0, 0.30))
        self.minimum_wage      = float(np.clip(self.minimum_wage + action.minimum_wage_delta, 0.0, 20.0))
        self.money_supply     += float(np.clip(action.money_supply_delta, -500.0, 500.0))
        stimulus               = float(np.clip(action.stimulus_package, 0.0, 500.0))

        # ── Phillips Curve inflation (academic-grade) ─────────────────────
        phillips_inflation = phillips_curve_inflation(
            self.unemployment, self.inflation,
            money_supply_growth=action.money_supply_delta / 1000.0
        )

        # Taylor Rule transmission: interest rate effect on inflation
        rate_inflation_effect = (0.05 - self.interest_rate) * 0.12

        # Additional inflation pressures
        fiscal_pressure = (self.ubi_amount / MAX_UBI) * 0.04
        if self.gov_budget < 0:
            fiscal_pressure += abs(self.gov_budget) / MAX_BUDGET * 0.04
        if stimulus > 0:
            fiscal_pressure += stimulus / 1000.0 * 0.03
        fiscal_pressure += self.import_tariff * 0.02

        self.inflation = float(np.clip(
            phillips_inflation * 0.4 + self.inflation * 0.3
            + fiscal_pressure + rate_inflation_effect
            + self._rng.normal(0, 0.008),
            -0.1, 0.5
        ))

        # ── Persistent shock mechanics ────────────────────────────────────
        shock_gdp_mult = 1.0
        shock_unrest_add = 0.0

        if self.shock_remaining > 0:
            effect = compute_shock_effect(self.current_shock, self.shock_remaining, self.shock_total_duration)
            shock_gdp_mult = effect["gdp_mult"]
            shock_unrest_add = effect["unrest_add"]
            self.inflation = float(np.clip(self.inflation + effect["inflation_add"], -0.1, 0.5))
            self.shock_remaining -= 1
            if self.shock_remaining == 0:
                self.current_shock = None
        else:
            shock = sample_shock(self._rng)
            if shock:
                self.current_shock = shock
                self.shock_remaining = shock["duration"]
                self.shock_total_duration = shock["duration"]
                shock_gdp_mult = shock["gdp_mult"]
                shock_unrest_add = shock["unrest_add"]
                self.inflation = float(np.clip(self.inflation + shock["inflation_add"], -0.1, 0.5))

        # ── Inter-group investment dynamics (Stackelberg) ─────────────────
        # Step 1: Wealthy/ultra-rich decide investment
        # Step 2: Poor/middle respond to employment signal
        wealthy_grp = next((g for g in self.citizen_groups if g.name == "wealthy"), None)
        ultra_rich_grp = next((g for g in self.citizen_groups if g.name == "ultra_rich"), None)
        rich_investment = 0.0
        if wealthy_grp:
            rich_investment += wealthy_grp.investment_output
        if ultra_rich_grp:
            rich_investment += ultra_rich_grp.investment_output

        # ── Citizen simulation ────────────────────────────────────────────
        total_gdp = total_taxes = total_unrest = total_satisf = total_unemp = 0.0
        total_citizens = sum(g.n for g in self.citizen_groups)
        total_taxes_evaded = 0.0
        total_trust = 0.0
        total_investment = 0.0
        total_capital_flight = 0.0
        any_strike = False

        for grp in self.citizen_groups:
            r = grp.step(
                self.tax_rate, self.ubi_amount, self.public_good_level,
                self.inflation, self.interest_rate, self.import_tariff, stimulus,
                minimum_wage=self.minimum_wage,
                money_supply_delta=action.money_supply_delta,
                rich_investment=rich_investment if grp.name in ("poor", "middle") else 0.0
            )
            total_gdp    += r["gross_income"] * shock_gdp_mult
            total_taxes  += r["taxes_paid"]
            total_taxes_evaded += r.get("taxes_evaded", 0.0)
            total_unrest += r["unrest"] * (grp.n / total_citizens)
            total_satisf += r["satisfaction"] * (grp.n / total_citizens)
            total_unemp  += r["unemployment_rate"] * (grp.n / total_citizens)
            total_trust  += r.get("trust", 1.0) * (grp.n / total_citizens)
            total_investment += r.get("investment_output", 0.0)
            total_capital_flight += r.get("capital_flight", 0.0) * (grp.n / total_citizens)
            if r.get("on_strike", False):
                any_strike = True

        # Apply shock unrest (with difficulty scaling)
        if shock_unrest_add != 0:
            scaled_unrest = shock_unrest_add * self.difficulty.shock_severity_mult
            total_unrest = float(np.clip(total_unrest + scaled_unrest, 0.0, 1.0))

        # ── Multi-Agent: Market speculator reaction (#8) ──────────────
        if self.enable_market_agent and self.market_consortium:
            market_response = self.market_consortium.observe_and_act(
                {
                    "gdp": total_gdp, "gdp_delta": total_gdp - self.prev_gdp,
                    "inflation": self.inflation, "unemployment": total_unemp,
                    "gov_budget": self.gov_budget, "unrest": total_unrest,
                    "interest_rate": self.interest_rate,
                    "shock_event": self.current_shock["name"] if self.current_shock else "none",
                },
                action.model_dump(),
            )
            self._last_market_response = market_response

            # Apply market effects to economy
            total_gdp += market_response.get("investment_boost", 0) * total_gdp
            total_gdp *= (1.0 - market_response.get("capital_outflow", 0))
            self.inflation += market_response.get("inflation_pressure", 0)
            self.consumer_confidence += market_response.get("confidence_effect", 0)
            self.consumer_confidence = float(np.clip(self.consumer_confidence, 0.0, 1.0))

            # Herd amplification of shocks
            if shock_gdp_mult < 1.0:
                herd = market_response.get("herd_amplification", 1.0)
                shock_gdp_mult = 1.0 - (1.0 - shock_gdp_mult) * herd

            # Speculative attack effects
            if market_response.get("attack_triggered", False):
                severity = market_response.get("attack_severity", 0.1)
                total_gdp *= (1.0 - severity * 0.5)
                total_unrest = float(np.clip(total_unrest + severity * 0.3, 0.0, 1.0))
                total_capital_flight = float(np.clip(
                    total_capital_flight + severity, 0.0, 1.0
                ))

        # Okun's Law: unemployment above NAIRU reduces GDP
        okun_mult = okuns_law_gdp_effect(total_unemp)
        total_gdp *= max(okun_mult, 0.5)

        # Investment → GDP boost (Solow capital accumulation)
        total_gdp += total_investment * 0.3

        # Interest rate → investment: lower rates stimulate
        inv_boost = max(0, (0.05 - self.interest_rate)) * total_gdp * 0.08
        total_gdp += inv_boost

        # Policy speech
        if getattr(action, "policy_speech", None) and self.speeches_used < 3:
            self.speeches_used += 1
            total_unrest = float(np.clip(total_unrest - 0.05, 0.0, 1.0))
            total_satisf = float(np.clip(total_satisf + 0.05, 0.0, 1.0))

        # ── Consumer confidence ───────────────────────────────────────────
        gdp_growth_signal = (total_gdp - self.prev_gdp) / max(self.prev_gdp, 1.0)
        strike_penalty = -0.15 if any_strike else 0.0
        confidence_delta = (
            gdp_growth_signal * 0.3
            + (0.5 - total_unrest) * 0.1
            - abs(self.inflation - 0.02) * 0.5
            + (total_trust - 1.0) * 0.1
            - total_capital_flight * 0.3
            + strike_penalty
        )
        self.consumer_confidence = float(np.clip(
            self.consumer_confidence * 0.8 + 0.2 * (0.5 + confidence_delta),
            0.0, 1.0
        ))

        self.unemployment = total_unemp

        # Budget calculation
        self.gov_budget += total_taxes - self.ubi_amount * 100 - self.public_good_level * 200 - stimulus
        self.gov_budget = float(np.clip(self.gov_budget, -MAX_BUDGET, MAX_BUDGET))

        gini = self._gini()
        gdp_growth = float(np.clip((total_gdp - self.prev_gdp) / max(self.prev_gdp, 1.0), -1, 1))
        self.prev_gdp = total_gdp
        self._step_count += 1

        # ── Termination ───────────────────────────────────────────────────
        crisis_termination = False
        if self.task_id == "task3_crisis" and gini > 0.88:
            crisis_termination = True
        if self._step_count > 0 and self._step_count % 10 == 0:
            if total_satisf < 0.40:
                crisis_termination = True

        reward = self._compute_reward(gdp_growth, gini, total_satisf, total_unrest, volatility_penalty)
        if crisis_termination:
            reward = StepReward(
                total=0.0, gdp_component=0.0, equality_component=0.0,
                satisfaction_component=0.0, unrest_penalty=0.0,
                deficit_penalty=0.0, volatility_penalty=0.0, task_progress=0.0,
            )

        obs = self._build_obs(satisfaction=total_satisf, unrest=total_unrest,
                               investment=total_investment, cap_flight=total_capital_flight,
                               strike=any_strike)
        done = self._step_count >= self.task_cfg["max_steps"] or crisis_termination
        self._done = done

        info = {
            "step": self._step_count, "gdp": total_gdp, "gini": gini,
            "satisfaction": total_satisf, "unrest": total_unrest,
            "unemployment": self.unemployment, "inflation": self.inflation,
            "gov_budget": self.gov_budget, "interest_rate": self.interest_rate,
            "import_tariff": self.import_tariff, "consumer_confidence": self.consumer_confidence,
            "trust": total_trust, "taxes_evaded": total_taxes_evaded,
            "investment": total_investment, "capital_flight": total_capital_flight,
            "strike_active": any_strike, "money_supply": self.money_supply,
            "minimum_wage": self.minimum_wage,
            "reasoning": action.reasoning or "",
            "shock": (self.current_shock["name"] if self.current_shock else "none"),
            "shock_remaining": self.shock_remaining,
            "crisis_terminated": crisis_termination,
            "action": action.model_dump(),
            # Multi-agent state (#8)
            "market_response": self._last_market_response if self.enable_market_agent else None,
            # Curriculum state (#9)
            "difficulty_level": self.difficulty.level if self.difficulty else 0,
            "difficulty_name": self.difficulty.name if self.difficulty else "Normal",
        }
        self._history.append({**info, "reward": reward.total})

        # ── Set SDK fields on the observation ────────────────────────────────
        obs.reward = reward.total
        obs.done = done
        obs.metadata = {
            **info,
            "reward_breakdown": reward.model_dump(),
        }
        return obs

    @property
    def state(self) -> Any:  # type: ignore[override]
        """SDK-compliant state property. Returns State with episode and step info."""
        try:
            return State(
                episode_id=self.task_id,
                step_count=self._step_count,
            )
        except Exception:
            return {"episode_id": self.task_id, "step_count": self._step_count}

    def _full_state(self) -> dict[str, Any]:
        """Rich internal state dict — used by graders, Gradio demo, and /state HTTP endpoint."""
        return {
            "task_id": self.task_id, "step": self._step_count, "done": self._done,
            "tax_rate": self.tax_rate, "ubi_amount": self.ubi_amount,
            "public_good_level": self.public_good_level, "gov_budget": self.gov_budget,
            "inflation": self.inflation, "unemployment": self.unemployment,
            "interest_rate": self.interest_rate, "import_tariff": self.import_tariff,
            "consumer_confidence": self.consumer_confidence,
            "money_supply": self.money_supply, "minimum_wage": self.minimum_wage,
            "current_shock": (self.current_shock["name"] if self.current_shock else None),
            "shock_remaining": self.shock_remaining,
            "gini": self._gini(), "history_length": len(self._history),
            "visual_dashboard": self._generate_dashboard(),
            "class_wealth": {g.name: float(g.wealth.mean()) for g in self.citizen_groups},
            "class_trust": {g.name: float(g.trust) for g in self.citizen_groups},
            "class_evasion": {g.name: float(g.tax_evasion_rate) for g in self.citizen_groups},
            "class_flight": {g.name: float(g.capital_flight) for g in self.citizen_groups},
            "class_investment": {g.name: float(g.investment_output) for g in self.citizen_groups},
        }

    def _build_obs(self, satisfaction=0.5, unrest=0.0,
                   investment=0.0, cap_flight=0.0, strike=False) -> EconomicObservation:
        gdp_now = sum(g.wealth.sum() * 0.05 for g in self.citizen_groups)
        gdp_delta = gdp_now - self.prev_gdp
        wealth_map = {g.name: float(g.wealth.mean()) for g in self.citizen_groups}
        return EconomicObservation(
            step=self._step_count, gdp=float(gdp_now), gdp_delta=float(gdp_delta),
            gini=self._gini(),
            satisfaction=float(np.clip(satisfaction, 0.0, 1.0)),
            unrest=float(np.clip(unrest, 0.0, 1.0)),
            gov_budget=self.gov_budget, tax_rate=self.tax_rate,
            ubi_amount=self.ubi_amount, public_good_level=self.public_good_level,
            inflation=self.inflation, unemployment=self.unemployment,
            shock_event=(self.current_shock["name"] if self.current_shock else "none"),
            shock_duration_remaining=self.shock_remaining,
            interest_rate=self.interest_rate, import_tariff=self.import_tariff,
            consumer_confidence=self.consumer_confidence,
            money_supply=self.money_supply, minimum_wage=self.minimum_wage,
            capital_flight_rate=float(np.clip(cap_flight, 0, 1)),
            strike_active=strike,
            private_investment=float(investment),
            poor_wealth=wealth_map.get("poor", 0),
            middle_wealth=wealth_map.get("middle", 0),
            wealthy_wealth=wealth_map.get("wealthy", 0),
            ultra_rich_wealth=wealth_map.get("ultra_rich", 0),
            task_id=self.task_id, task_description=self.task_cfg["description"],
        )

    def _gini(self) -> float:
        if not self.citizen_groups:
            return 0.0
        all_w = np.concatenate([g.wealth for g in self.citizen_groups])
        all_w = np.sort(all_w)
        n = len(all_w)
        idx = np.arange(1, n + 1)
        return float(np.clip(
            (2 * idx - n - 1).dot(all_w) / (n * all_w.sum() + 1e-9), 0, 1
        ))

    def _compute_reward(self, gdp_growth, gini, satisfaction, unrest, vol_pen) -> StepReward:
        gdp_norm = float(np.clip((gdp_growth + 0.5) / 1.0, 0.0, 1.0))
        deficit_pen = abs(min(self.gov_budget, 0)) / MAX_BUDGET
        tax_penalty = float(np.clip(max(0, self.tax_rate - 0.5) * 0.4, 0.0, 1.0))

        gdp_c     = 0.35 * gdp_norm
        eq_c      = 0.25 * (1.0 - gini)
        sat_c     = 0.15 * float(np.clip(satisfaction, 0.0, 1.0))
        unrest_p  = -0.12 * float(np.clip(unrest, 0.0, 1.0))
        deficit_p = -0.08 * float(np.clip(deficit_pen, 0.0, 1.0))
        tax_p     = -0.05 * tax_penalty
        v_pen     = -0.08 * vol_pen

        raw = gdp_c + eq_c + sat_c + unrest_p + deficit_p + tax_p + v_pen
        total = float(np.clip((raw + 0.20) / 0.95, 0.0, 1.0))
        progress = self._task_progress(gdp_growth, gini, unrest)

        return StepReward(
            total=total, gdp_component=gdp_c, equality_component=eq_c,
            satisfaction_component=sat_c, unrest_penalty=unrest_p,
            deficit_penalty=deficit_p, volatility_penalty=v_pen,
            task_progress=progress,
        )

    def _task_progress(self, gdp_growth, gini, unrest) -> float:
        if self.task_id == "task1_stability":
            return (
                (1.0 if gdp_growth >= 0 else 0.0)
                + float(np.clip(1.0 - max(0, gini - 0.45) / 0.3, 0, 1))
                + float(np.clip(1.0 - unrest / 0.15, 0, 1))
                + (1.0 if self.gov_budget >= 0 else 0.0)
            ) / 4.0
        elif self.task_id == "task2_recession":
            return (
                float(np.clip(gdp_growth / 0.02, 0, 1))
                + float(np.clip((self.gov_budget + 2500) / 2500, 0, 1))
                + float(np.clip(1.0 - unrest / 0.15, 0, 1))
                + float(np.clip(1.0 - self.unemployment / 0.20, 0, 1))
            ) / 4.0
        elif self.task_id == "task3_crisis":
            return (
                float(np.clip(1.0 - max(0, gini - 0.40) / 0.20, 0, 1))
                + float(np.clip(1.0 - unrest / 0.10, 0, 1))
                + float(np.clip(gdp_growth / 0.01, 0, 1))
                + float(np.clip(1.0 - max(0, self.inflation - 0.05) / 0.10, 0, 1))
            ) / 4.0
        elif self.task_id == "task4_stagflation":
            return (
                float(np.clip(gdp_growth / 0.02, 0, 1))
                + float(np.clip(1.0 - max(0, self.inflation - 0.03) / 0.12, 0, 1))
                + float(np.clip(1.0 - max(0, self.unemployment - 0.06) / 0.18, 0, 1))
                + float(np.clip((self.gov_budget + 1200) / 1200, 0, 1))
            ) / 4.0
        elif self.task_id == "task5_pandemic":
            return (
                float(np.clip(1.0 - max(0, self.unemployment - 0.06) / 0.22, 0, 1))
                + float(np.clip(gdp_growth / 0.02, 0, 1))
                + float(np.clip(1.0 - max(0, self.inflation - 0.04) / 0.10, 0, 1))
                + float(np.clip((self.gov_budget + 3000) / 3000, 0, 1))
            ) / 4.0
        return 0.5

    def _generate_dashboard(self) -> str:
        if not self._history:
            return ""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            fig.suptitle(f"SocialContract-v0 — {self.task_id} Dashboard",
                         fontsize=14, fontweight="bold")
            steps = [h["step"] for h in self._history]

            # Panel 1: GDP + Investment
            ax = axes[0, 0]
            gdp = [h["gdp"] for h in self._history]
            inv = [h.get("investment", 0) for h in self._history]
            ax.plot(steps, gdp, "b-", linewidth=1.5, label="GDP")
            ax2 = ax.twinx()
            ax2.plot(steps, inv, "g--", linewidth=1, alpha=0.6, label="Investment")
            ax2.tick_params(axis='y', labelcolor='green', labelsize=7)
            ax.fill_between(steps, gdp, alpha=0.1, color="blue")
            ax.set_title("GDP & Investment", fontsize=10)
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(alpha=0.3)

            # Panel 2: Gini
            ax = axes[0, 1]
            gini = [h["gini"] for h in self._history]
            ax.plot(steps, gini, "r-", linewidth=1.5)
            ax.axhline(y=0.48, color="orange", linestyle="--", alpha=0.7, label="Target")
            ax.set_title("Gini (Inequality)", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            # Panel 3: Budget
            ax = axes[0, 2]
            budgets = [h["gov_budget"] for h in self._history]
            colors = ["green" if b >= 0 else "red" for b in budgets]
            ax.bar(steps, budgets, color=colors, alpha=0.7)
            ax.axhline(y=0, color="black", linewidth=0.5)
            ax.set_title("Gov Budget", fontsize=10)
            ax.grid(alpha=0.3)

            # Panel 4: Inflation & Unemployment (Phillips Curve)
            ax = axes[1, 0]
            inf = [h["inflation"] for h in self._history]
            unemp = [h["unemployment"] for h in self._history]
            ax.plot(steps, inf, "m-", linewidth=1.5, label="Inflation")
            ax.plot(steps, unemp, "c-", linewidth=1.5, label="Unemployment")
            ax.axhline(y=NAIRU, color="gray", linestyle=":", alpha=0.5, label=f"NAIRU={NAIRU}")
            ax.set_title("Phillips Curve Indicators", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            # Panel 5: Trust & Capital Flight
            ax = axes[1, 1]
            trust = [h.get("trust", 1.0) for h in self._history]
            flight = [h.get("capital_flight", 0.0) for h in self._history]
            ax.plot(steps, trust, "g-", linewidth=1.5, label="Trust")
            ax3 = ax.twinx()
            ax3.plot(steps, flight, "r--", linewidth=1, alpha=0.7, label="Cap. Flight")
            ax3.tick_params(axis='y', labelcolor='red', labelsize=7)
            ax.set_title("Trust & Capital Flight", fontsize=10)
            ax.legend(fontsize=7, loc="upper left")
            ax.grid(alpha=0.3)

            # Panel 6: Reward & Confidence
            ax = axes[1, 2]
            rewards = [h["reward"] for h in self._history]
            confidence = [h.get("consumer_confidence", 0.5) for h in self._history]
            ax.plot(steps, rewards, "gold", linewidth=1.5, label="Reward")
            ax.plot(steps, confidence, "purple", linewidth=1.5, label="Confidence", alpha=0.7)
            ax.set_title("Reward & Confidence", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return ""
