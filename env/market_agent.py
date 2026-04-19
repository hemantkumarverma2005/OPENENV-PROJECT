"""
market_agent.py — Adversarial Market Speculator Agent
═════════════════════════════════════════════════════
Adds a second agent to the environment: a "market speculator" that
reacts to the policy advisor's decisions. This creates genuine
multi-agent dynamics and strengthens Theme #1 alignment.

The market agent:
  - Observes the same economy as the policy advisor
  - Decides investment allocation (domestic/foreign/cash)
  - Can trigger speculative attacks on the currency
  - Reacts to policy credibility (consistent policies → invests more)
  - Can amplify or dampen economic shocks through herd behavior

Academic foundations:
  - Soros (1994) — Reflexivity: market expectations affect outcomes
  - Kyle (1985) — Strategic informed trading
  - Diamond & Dybvig (1983) — Bank run / coordination failure dynamics
  - Keynes (1936) — Animal spirits, beauty contest reasoning
"""

import numpy as np
from typing import Optional


class MarketSpeculator:
    """
    An adversarial market agent that reacts to policy decisions.
    
    The speculator maintains beliefs about the economy and adjusts
    its positioning (invest/divest/attack) based on observed policy
    and outcomes. This creates a feedback loop where the policy
    advisor must consider market reactions.
    """

    def __init__(self, rng: np.random.Generator, aggressiveness: float = 0.5):
        """
        Args:
            rng: Random number generator for reproducibility
            aggressiveness: How aggressively the speculator reacts [0, 1]
                           0 = passive index fund, 1 = aggressive hedge fund
        """
        self.rng = rng
        self.aggressiveness = np.clip(aggressiveness, 0.0, 1.0)

        # Internal state
        self.confidence = 0.5          # Belief in government competence [0, 1]
        self.position = 0.0            # Net market position [-1 short, +1 long]
        self.speculation_pressure = 0.0  # Pressure on the economy [-0.3, 0.3]
        self.attack_cooldown = 0       # Steps until next speculative attack allowed
        self.policy_memory = []        # Remember past policies for trend analysis
        self.attack_count = 0          # Total attacks this episode

        # Portfolio allocation
        self.domestic_allocation = 0.6   # Fraction invested domestically
        self.foreign_allocation = 0.2    # Fraction invested abroad
        self.cash_allocation = 0.2       # Fraction held as cash

    def reset(self, rng: Optional[np.random.Generator] = None):
        """Reset speculator state for a new episode."""
        if rng is not None:
            self.rng = rng
        self.confidence = 0.5
        self.position = 0.0
        self.speculation_pressure = 0.0
        self.attack_cooldown = 0
        self.policy_memory = []
        self.attack_count = 0
        self.domestic_allocation = 0.6
        self.foreign_allocation = 0.2
        self.cash_allocation = 0.2

    def observe_and_act(self, obs_dict: dict, policy_action: dict) -> dict:
        """
        Observe the economy and policy action, then decide market response.

        Args:
            obs_dict: Current economic observation (dict form)
            policy_action: The policy advisor's action this step (dict form)

        Returns:
            market_response: dict with market effects on the economy
        """
        # ── Update confidence based on policy consistency ────────────────
        self._update_confidence(obs_dict, policy_action)

        # ── Store policy memory ─────────────────────────────────────────
        self.policy_memory.append({
            "tax_delta": policy_action.get("tax_delta", 0.0),
            "ubi_delta": policy_action.get("ubi_delta", 0.0),
            "interest_rate_delta": policy_action.get("interest_rate_delta", 0.0),
            "stimulus": policy_action.get("stimulus_package", 0.0),
            "money_supply_delta": policy_action.get("money_supply_delta", 0.0),
            "gdp": obs_dict.get("gdp", 0),
            "inflation": obs_dict.get("inflation", 0),
            "gov_budget": obs_dict.get("gov_budget", 0),
        })
        if len(self.policy_memory) > 15:
            self.policy_memory = self.policy_memory[-15:]

        # ── Portfolio allocation decision ────────────────────────────────
        self._rebalance_portfolio(obs_dict)

        # ── Speculative attack decision ──────────────────────────────────
        attack_effect = self._consider_attack(obs_dict)

        # ── Calculate market effects on economy ──────────────────────────
        # Domestic investment boosts GDP
        investment_boost = (
            self.domestic_allocation * self.confidence * 0.05
            * (1.0 + self.position * 0.3)
        )

        # Foreign allocation = capital leaving the country
        capital_outflow = self.foreign_allocation * (1.0 - self.confidence) * 0.03

        # Speculation pressure affects consumer confidence and inflation
        inflation_pressure = self.speculation_pressure * 0.02
        confidence_effect = self.speculation_pressure * -0.05

        # ── Herd behavior amplification ──────────────────────────────────
        # When speculator is very bearish, it can amplify negative shocks
        shock_event = obs_dict.get("shock_event", "none")
        herd_amplification = 1.0
        if shock_event != "none" and self.position < -0.3:
            herd_amplification = 1.0 + abs(self.position) * self.aggressiveness * 0.3

        # ── Update cooldown ─────────────────────────────────────────────
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        return {
            "investment_boost": float(investment_boost),
            "capital_outflow": float(np.clip(capital_outflow, 0, 0.10)),
            "inflation_pressure": float(np.clip(inflation_pressure, -0.02, 0.03)),
            "confidence_effect": float(np.clip(confidence_effect, -0.10, 0.10)),
            "speculation_pressure": float(self.speculation_pressure),
            "herd_amplification": float(herd_amplification),
            "attack_triggered": attack_effect["triggered"],
            "attack_severity": attack_effect.get("severity", 0.0),
            "market_position": float(self.position),
            "market_confidence": float(self.confidence),
            "portfolio": {
                "domestic": round(self.domestic_allocation, 3),
                "foreign": round(self.foreign_allocation, 3),
                "cash": round(self.cash_allocation, 3),
            },
        }

    def _update_confidence(self, obs_dict: dict, policy_action: dict):
        """
        Update speculator's confidence in the government.
        Based on Barro-Gordon credibility + observed outcomes.
        """
        # ── Policy consistency check ────────────────────────────────────
        volatility_penalty = 0.0
        if len(self.policy_memory) >= 2:
            prev = self.policy_memory[-1]
            # Reversals in policy direction are very bad for confidence
            tax_reversal = (
                policy_action.get("tax_delta", 0) * prev["tax_delta"] < -1e-6
            )
            rate_reversal = (
                policy_action.get("interest_rate_delta", 0)
                * prev["interest_rate_delta"] < -1e-6
            )
            if tax_reversal:
                volatility_penalty += 0.05 * self.aggressiveness
            if rate_reversal:
                volatility_penalty += 0.03 * self.aggressiveness

        # ── Outcome-based confidence ────────────────────────────────────
        inflation = obs_dict.get("inflation", 0.02)
        gov_budget = obs_dict.get("gov_budget", 0)
        gdp_delta = obs_dict.get("gdp_delta", 0)

        outcome_signal = 0.0
        # Low inflation is good
        if inflation < 0.04:
            outcome_signal += 0.02
        elif inflation > 0.08:
            outcome_signal -= 0.04 * self.aggressiveness

        # Positive GDP growth is good
        if gdp_delta > 0:
            outcome_signal += 0.01
        else:
            outcome_signal -= 0.02

        # Sustainable budget
        if gov_budget > 0:
            outcome_signal += 0.01
        elif gov_budget < -2000:
            outcome_signal -= 0.03 * self.aggressiveness

        # Large stimulus packages erode confidence (inflationary)
        if policy_action.get("stimulus_package", 0) > 200:
            outcome_signal -= 0.02 * self.aggressiveness

        # Excessive QE erodes confidence
        if abs(policy_action.get("money_supply_delta", 0)) > 200:
            outcome_signal -= 0.01 * self.aggressiveness

        # ── Apply ────────────────────────────────────────────────────────
        self.confidence = float(np.clip(
            self.confidence * 0.85 + 0.15 * (0.5 + outcome_signal)
            - volatility_penalty,
            0.1, 0.95
        ))

    def _rebalance_portfolio(self, obs_dict: dict):
        """Adjust portfolio allocation based on confidence and conditions."""
        inflation = obs_dict.get("inflation", 0.02)
        interest_rate = obs_dict.get("interest_rate", 0.05)

        # High confidence → more domestic investment
        target_domestic = 0.3 + self.confidence * 0.5
        # Low confidence → more foreign/cash
        target_foreign = max(0.05, 0.4 - self.confidence * 0.3)
        target_cash = 1.0 - target_domestic - target_foreign

        # High inflation → more foreign assets (inflation hedge)
        if inflation > 0.06:
            target_foreign += 0.1
            target_domestic -= 0.1

        # High interest rates → more cash (opportunity cost)
        if interest_rate > 0.10:
            target_cash += 0.1
            target_domestic -= 0.1

        # Smooth transition (don't rebalance instantly)
        alpha = 0.3
        self.domestic_allocation = float(np.clip(
            self.domestic_allocation * (1 - alpha) + target_domestic * alpha,
            0.05, 0.85
        ))
        self.foreign_allocation = float(np.clip(
            self.foreign_allocation * (1 - alpha) + target_foreign * alpha,
            0.05, 0.60
        ))
        self.cash_allocation = float(np.clip(
            1.0 - self.domestic_allocation - self.foreign_allocation,
            0.05, 0.50
        ))

        # Normalize
        total = self.domestic_allocation + self.foreign_allocation + self.cash_allocation
        self.domestic_allocation /= total
        self.foreign_allocation /= total
        self.cash_allocation /= total

        # Update position based on domestic allocation
        self.position = float(np.clip(
            (self.domestic_allocation - 0.5) * 2.0,
            -1.0, 1.0
        ))

        # Update speculation pressure
        self.speculation_pressure = float(np.clip(
            (0.5 - self.confidence) * self.aggressiveness * 0.5
            + (0.5 - self.domestic_allocation) * 0.3,
            -0.3, 0.3
        ))

    def _consider_attack(self, obs_dict: dict) -> dict:
        """
        Decide whether to trigger a speculative attack.
        Based on Diamond-Dybvig coordination failure.

        Attacks happen when:
          - Confidence is very low (<0.3)
          - Government deficit is large
          - Inflation is high
          - Attack cooldown has expired
        """
        if self.attack_cooldown > 0:
            return {"triggered": False}

        inflation = obs_dict.get("inflation", 0.02)
        gov_budget = obs_dict.get("gov_budget", 0)
        unrest = obs_dict.get("unrest", 0)

        attack_pressure = (
            max(0, 0.3 - self.confidence) * 2.0          # Low confidence
            + max(0, inflation - 0.06) * 3.0              # High inflation
            + max(0, -gov_budget / 3000) * 1.5             # Large deficit
            + max(0, unrest - 0.15) * 1.0                  # Social instability
        ) * self.aggressiveness

        # Stochastic threshold (not every bad situation triggers attack)
        threshold = self.rng.uniform(0.8, 1.5)

        if attack_pressure > threshold:
            self.attack_cooldown = 5  # Can't attack again for 5 steps
            self.attack_count += 1
            severity = float(np.clip(attack_pressure * 0.3, 0.05, 0.25))

            # Attack effects
            self.confidence *= 0.8  # Attack further erodes confidence
            self.domestic_allocation *= 0.7  # Flight from domestic assets
            self.speculation_pressure = float(np.clip(
                self.speculation_pressure + severity, -0.3, 0.3
            ))

            return {
                "triggered": True,
                "severity": severity,
                "type": "speculative_attack",
                "description": (
                    f"Market speculator triggered a speculative attack! "
                    f"Severity: {severity:.2f}. Capital fleeing domestic assets."
                ),
            }

        return {"triggered": False}

    def get_state(self) -> dict:
        """Return full speculator state for observation/logging."""
        return {
            "market_confidence": round(self.confidence, 4),
            "market_position": round(self.position, 4),
            "speculation_pressure": round(self.speculation_pressure, 4),
            "attack_cooldown": self.attack_cooldown,
            "total_attacks": self.attack_count,
            "portfolio": {
                "domestic": round(self.domestic_allocation, 3),
                "foreign": round(self.foreign_allocation, 3),
                "cash": round(self.cash_allocation, 3),
            },
        }


class MarketConsortium:
    """
    A group of market agents with different strategies, creating
    emergent market dynamics through agent heterogeneity.

    Includes:
      - A conservative institutional investor (low aggressiveness)
      - A hedge fund (high aggressiveness)
      - A momentum trader (follows trends)
    """

    def __init__(self, rng: np.random.Generator):
        self.agents = [
            MarketSpeculator(rng, aggressiveness=0.2),   # Institutional
            MarketSpeculator(rng, aggressiveness=0.7),   # Hedge fund
            MarketSpeculator(rng, aggressiveness=0.5),   # Momentum
        ]
        self.weights = [0.5, 0.3, 0.2]  # Institutional has most capital

    def reset(self, rng: Optional[np.random.Generator] = None):
        for agent in self.agents:
            agent.reset(rng)

    def observe_and_act(self, obs_dict: dict, policy_action: dict) -> dict:
        """Aggregate responses from all market agents."""
        responses = [
            agent.observe_and_act(obs_dict, policy_action)
            for agent in self.agents
        ]

        # Weighted aggregation
        agg = {
            "investment_boost": sum(
                r["investment_boost"] * w for r, w in zip(responses, self.weights)
            ),
            "capital_outflow": sum(
                r["capital_outflow"] * w for r, w in zip(responses, self.weights)
            ),
            "inflation_pressure": sum(
                r["inflation_pressure"] * w for r, w in zip(responses, self.weights)
            ),
            "confidence_effect": sum(
                r["confidence_effect"] * w for r, w in zip(responses, self.weights)
            ),
            "speculation_pressure": sum(
                r["speculation_pressure"] * w for r, w in zip(responses, self.weights)
            ),
            "herd_amplification": max(r["herd_amplification"] for r in responses),
            "attack_triggered": any(r["attack_triggered"] for r in responses),
            "attack_severity": max(r.get("attack_severity", 0) for r in responses),
            "agents": [
                {
                    "type": ["institutional", "hedge_fund", "momentum"][i],
                    "confidence": r["market_confidence"],
                    "position": r["market_position"],
                    "portfolio": r["portfolio"],
                }
                for i, r in enumerate(responses)
            ],
        }

        # Average market confidence
        agg["market_confidence"] = sum(
            r["market_confidence"] * w for r, w in zip(responses, self.weights)
        )

        return agg

    def get_state(self) -> dict:
        return {
            "agents": [
                {"type": ["institutional", "hedge_fund", "momentum"][i],
                 **agent.get_state()}
                for i, agent in enumerate(self.agents)
            ]
        }
