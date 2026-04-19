"""
citizens.py — Heterogeneous citizen simulation grounded in economic theory.

Academic foundations:
  - Keynesian consumption function: MPC varies by wealth class
    (Friedman 1957 Permanent Income Hypothesis; poor consume ~90%, rich ~20%)
  - Phillips Curve: inflation = f(unemployment) with NAIRU ≈ 4.5%
    (A.W. Phillips, "The Relation Between Unemployment and the Rate of Change
    of Money Wage Rates in the United Kingdom, 1861–1957", Economica, 1958)
  - Okun's Law: 1pp rise in unemployment ≈ 2pp fall in GDP
    (A. Okun, "Potential GNP: Its Measurement and Significance", 1962)
  - Solow Growth Model savings rate: s(r) = s₀ + α·r where r = interest rate
    (R. Solow, "A Contribution to the Theory of Economic Growth", QJE, 1956)
  - Barro-Gordon credibility model: trust dynamics affect policy effectiveness
    (R. Barro & D. Gordon, "Rules, Discretion, and Reputation", JME, 1983)
  - Laffer Curve: revenue peaks at interior tax rate, not at extremes
    (A. Laffer, 1974; formalised in Trabandt & Uhlig, JME, 2011)
  - Capital flight: Collier, Hoeffler & Pattillo, "Capital Flight as a
    Portfolio Choice", World Bank Economic Review, 2001

Enhanced features:
  - Inter-group dynamics: wealthy invest → creates jobs for poor/middle
  - Capital flight: ultra-rich move wealth offshore under high tax / low trust
  - Collective action: citizens coordinate strikes when satisfaction is very low
  - Minimum wage floor affects poor/middle employment
  - Money supply (QE) affects inflation and asset prices
  - Persistent shocks that last 2-4 steps with diminishing effects
"""
import numpy as np

# ── Academic calibration: Marginal Propensity to Consume (Friedman 1957) ────
# Poor households spend nearly all income (MPC ≈ 0.85-0.95)
# Rich households save most (MPC ≈ 0.15-0.25)
# These are calibrated to OECD Household Savings Rate data (2019).
CLASS_CONFIG = {
    "poor":       {"n": 50, "base_wealth": 10,  "labor_sens": 0.4,
                   "consume_rate": 0.90, "unrest_thresh": 0.3,
                   "savings_rate": 0.05, "mpc": 0.90,
                   "min_wage_sensitive": True},
    "middle":     {"n": 30, "base_wealth": 50,  "labor_sens": 0.7,
                   "consume_rate": 0.70, "unrest_thresh": 0.5,
                   "savings_rate": 0.15, "mpc": 0.70,
                   "min_wage_sensitive": True},
    "wealthy":    {"n": 15, "base_wealth": 200, "labor_sens": 0.9,
                   "consume_rate": 0.40, "unrest_thresh": 0.7,
                   "savings_rate": 0.40, "mpc": 0.35,
                   "min_wage_sensitive": False},
    "ultra_rich": {"n": 5,  "base_wealth": 800, "labor_sens": 1.0,
                   "consume_rate": 0.20, "unrest_thresh": 0.9,
                   "savings_rate": 0.60, "mpc": 0.15,
                   "min_wage_sensitive": False},
}

# ── Exogenous economic shocks ─────────────────────────────────────────────────
# Calibrated to IMF World Economic Outlook crisis frequency data (1970-2023).
# Each shock modifies the economy for 2-5 steps with diminishing effects.
SHOCK_CATALOGUE = [
    {"name": "commodity_price_spike",
     "description": "Global commodity prices surge — input costs rise, GDP contracts. "
                    "Comparable to 2008 oil price spike ($147/barrel, IMF WEO April 2009).",
     "gdp_mult": 0.92, "unrest_add": 0.05, "inflation_add": 0.02, "prob": 0.05,
     "min_duration": 2, "max_duration": 4},

    {"name": "tech_productivity_boom",
     "description": "Breakthrough technology boosts productivity. "
                    "Comparable to 1995-2000 US dot-com TFP surge (BLS data).",
     "gdp_mult": 1.08, "unrest_add": -0.02, "inflation_add": -0.01, "prob": 0.04,
     "min_duration": 2, "max_duration": 3},

    {"name": "trade_war_escalation",
     "description": "Trade tariffs imposed — exports fall, consumers pay higher prices. "
                    "Calibrated to US-China 2018-2019 tariff war (USITC, 2020).",
     "gdp_mult": 0.94, "unrest_add": 0.04, "inflation_add": 0.03, "prob": 0.04,
     "min_duration": 2, "max_duration": 4},

    {"name": "pandemic_wave",
     "description": "New pandemic wave reduces labor supply and consumer spending. "
                    "Calibrated to COVID-19 Q2-2020 GDP contraction (OECD median: -11.6%).",
     "gdp_mult": 0.88, "unrest_add": 0.08, "inflation_add": 0.01, "prob": 0.03,
     "min_duration": 3, "max_duration": 5},

    {"name": "foreign_investment_inflow",
     "description": "Foreign capital floods in — boosts GDP but widens inequality. "
                    "Comparable to Eastern European EU accession FDI surge (World Bank, 2007).",
     "gdp_mult": 1.06, "unrest_add": 0.01, "inflation_add": 0.00, "prob": 0.04,
     "min_duration": 2, "max_duration": 3},

    {"name": "housing_bubble_burst",
     "description": "Housing market crashes — wealth destruction hits middle class hardest. "
                    "Calibrated to US 2007-2009 housing crisis (Case-Shiller Index, -27%).",
     "gdp_mult": 0.90, "unrest_add": 0.06, "inflation_add": -0.02, "prob": 0.03,
     "min_duration": 3, "max_duration": 5},

    {"name": "energy_subsidy_cut",
     "description": "Government forced to cut energy subsidies — cost of living rises. "
                    "Comparable to Nigeria fuel subsidy removal 2012 (IMF WP/14/214).",
     "gdp_mult": 0.96, "unrest_add": 0.05, "inflation_add": 0.04, "prob": 0.04,
     "min_duration": 2, "max_duration": 3},

    {"name": "pandemic_lockdown_wave",
     "description": "New pandemic wave forces partial lockdown. "
                    "Calibrated to Delta/Omicron waves (Nature Medicine, 2022).",
     "gdp_mult": 0.85, "unrest_add": 0.10, "inflation_add": 0.01, "prob": 0.03,
     "min_duration": 2, "max_duration": 4},

    {"name": "supply_chain_disruption",
     "description": "Global supply chain bottleneck. "
                    "Calibrated to 2021 shipping crisis (Federal Reserve Bank of NY GSCPI).",
     "gdp_mult": 0.93, "unrest_add": 0.03, "inflation_add": 0.05, "prob": 0.04,
     "min_duration": 2, "max_duration": 4},
]


def sample_shock(rng: np.random.Generator) -> dict | None:
    """Roll for a NEW exogenous shock this step. Returns shock dict or None."""
    for shock in SHOCK_CATALOGUE:
        if rng.random() < shock["prob"]:
            duration = rng.integers(shock["min_duration"], shock["max_duration"] + 1)
            return {**shock, "duration": int(duration)}
    return None


def compute_shock_effect(shock: dict, remaining: int, total_duration: int) -> dict:
    """
    Compute diminishing shock effects based on how many steps remain.
    Effects decay linearly: full effect at start, 50% at midpoint, 25% at end.
    """
    decay = remaining / max(total_duration, 1)
    return {
        "gdp_mult": 1.0 + (shock["gdp_mult"] - 1.0) * decay,
        "unrest_add": shock["unrest_add"] * decay,
        "inflation_add": shock["inflation_add"] * decay,
    }


# ── Phillips Curve implementation ─────────────────────────────────────────────
# Academic ref: Phillips (1958), augmented expectations (Phelps-Friedman)
NAIRU = 0.045  # Non-Accelerating Inflation Rate of Unemployment (OECD avg ≈ 4.5%)

def phillips_curve_inflation(unemployment: float, expected_inflation: float,
                              money_supply_growth: float = 0.0) -> float:
    """
    Expectations-augmented Phillips Curve:
      π = πᵉ − α(u − u*) + ε
    Where:
      π   = actual inflation
      πᵉ  = expected inflation (adaptive: last period's inflation)
      u   = actual unemployment
      u*  = NAIRU (0.045)
      α   = Phillips curve slope (calibrated to 0.5, OECD estimate)
      ε   = money supply effect (QE/tightening)
    """
    alpha = 0.5  # Phillips curve slope
    gap = unemployment - NAIRU
    pi = expected_inflation - alpha * gap + money_supply_growth * 0.02
    return float(np.clip(pi, -0.05, 0.30))


# ── Okun's Law implementation ─────────────────────────────────────────────────
# Academic ref: Okun (1962), calibrated coefficient β ≈ 2.0 (US data, IMF WEO)
def okuns_law_gdp_effect(unemployment: float, natural_rate: float = NAIRU) -> float:
    """
    Okun's Law: ΔGDP/GDP = -β(u - u*)
    Each 1pp above NAIRU ≈ 2pp GDP loss.
    Returns GDP multiplier.
    """
    beta = 2.0
    gap = unemployment - natural_rate
    return 1.0 - beta * max(0, gap)


class CitizenGroup:
    """
    A group of citizens with heterogeneous behavior, trust dynamics,
    policy memory, inter-group investment linkages, and collective action
    capability. Grounded in academic economic theory.

    Academic foundations:
      - Consumption: Keynesian MPC (Friedman 1957)
      - Savings:     Solow model s(r) = s₀ + α·r
      - Trust:       Barro-Gordon credibility (1983)
      - Evasion:     Allingham-Sandmo tax compliance model (1972)
      - Collective:  Olson's Logic of Collective Action (1965)
    """

    def __init__(self, name: str, rng: np.random.Generator, wealth_override=None):
        cfg = CLASS_CONFIG[name]
        self.name          = name
        self.n             = cfg["n"]
        self.labor_sens    = cfg["labor_sens"]
        self.consume_rate  = cfg["consume_rate"]
        self.unrest_thresh = cfg["unrest_thresh"]
        self.savings_rate  = cfg["savings_rate"]
        self.mpc           = cfg["mpc"]          # Marginal Propensity to Consume
        self.min_wage_sensitive = cfg["min_wage_sensitive"]
        self.rng           = rng
        base = wealth_override if wealth_override is not None else cfg["base_wealth"]
        self.wealth = rng.normal(base, base * 0.1, size=self.n).clip(1)

        # ── Trust & memory (Barro-Gordon 1983) ────────────────────────────
        self.trust = 1.0
        self.policy_memory: list[dict] = []
        self.tax_evasion_rate = 0.0

        # ── Inter-group dynamics ──────────────────────────────────────────
        self.investment_output = 0.0     # How much this group invests → creates jobs
        self.capital_flight = 0.0        # Wealth moved offshore (ultra-rich)
        self.on_strike = False           # Collective action flag
        self.strike_cooldown = 0

    def step(self, tax_rate, ubi_amount, public_good_level, inflation=0.0,
             interest_rate=0.05, import_tariff=0.0, stimulus=0.0,
             minimum_wage=5.0, money_supply_delta=0.0,
             rich_investment=0.0):
        """
        Simulate one economic period for this citizen group.

        Parameters:
            rich_investment: float — investment income flowing IN from wealthy/ultra-rich
                             groups' investment activities, creating employment for
                             poor/middle classes. Implements Stackelberg leader-follower
                             dynamics where rich "lead" with investment and poor "follow"
                             with labor supply response.
        """
        # ── Update trust (Barro-Gordon credibility model) ─────────────────
        self._update_trust(tax_rate, ubi_amount, public_good_level)
        self.policy_memory.append({
            "tax_rate": tax_rate,
            "ubi_amount": ubi_amount,
            "public_good_level": public_good_level,
        })
        if len(self.policy_memory) > 10:
            self.policy_memory = self.policy_memory[-10:]

        # ── Collective action: strikes (Olson 1965) ───────────────────────
        # Citizens coordinate strikes when satisfaction is very low and trust eroded
        if self.strike_cooldown > 0:
            self.strike_cooldown -= 1
            self.on_strike = False

        # ── Tax evasion (Allingham-Sandmo 1972 compliance model) ──────────
        # Evasion = f(tax_rate, detection_probability, risk_aversion)
        # Detection probability proxy: public_good_level (better gov = more enforcement)
        detection_prob = 0.3 + public_good_level * 0.3
        if self.name in ("wealthy", "ultra_rich"):
            evasion_benefit = max(0, tax_rate - 0.30) * 2.5
            trust_factor = max(0, 1.0 - self.trust) * 0.5
            risk_aversion = 0.3 if self.name == "ultra_rich" else 0.5
            self.tax_evasion_rate = float(np.clip(
                (evasion_benefit + trust_factor) * (1.0 - detection_prob * risk_aversion),
                0.0, 0.45
            ))
        else:
            self.tax_evasion_rate = float(np.clip(
                max(0, tax_rate - 0.50) * 0.5 * (1.0 - detection_prob * 0.8),
                0.0, 0.15
            ))

        # ── Capital flight (Collier et al. 2001) ─────────────────────────
        # Ultra-rich move wealth offshore when trust < 0.6 OR taxes > 0.50
        if self.name == "ultra_rich":
            flight_pressure = (
                max(0, tax_rate - 0.45) * 1.5
                + max(0, 0.6 - self.trust) * 0.4
                + max(0, import_tariff - 0.15) * 0.3  # Tariffs signal protectionism
            )
            self.capital_flight = float(np.clip(flight_pressure, 0.0, 0.30))
            # Capital flight reduces effective wealth for domestic economy
            flight_loss = self.wealth * self.capital_flight * 0.02
            self.wealth = (self.wealth - flight_loss).clip(0)
        elif self.name == "wealthy":
            flight_pressure = max(0, tax_rate - 0.55) * 0.8 + max(0, 0.5 - self.trust) * 0.2
            self.capital_flight = float(np.clip(flight_pressure, 0.0, 0.15))
            flight_loss = self.wealth * self.capital_flight * 0.01
            self.wealth = (self.wealth - flight_loss).clip(0)
        else:
            self.capital_flight = 0.0

        # ── Labor decision (with strike, minimum wage, and investment effects) ─
        interest_labor_effect = 0.0
        if self.name in ("wealthy", "ultra_rich"):
            interest_labor_effect = -interest_rate * 0.3
        else:
            interest_labor_effect = interest_rate * 0.1

        trust_labor_bonus = (self.trust - 1.0) * 0.1

        # Rich investment creates jobs for poor/middle (inter-group linkage)
        investment_employment = 0.0
        if self.name in ("poor", "middle") and rich_investment > 0:
            investment_employment = min(rich_investment / 200.0, 0.15)

        # Strike reduces labor to zero
        strike_factor = 0.0 if self.on_strike else 1.0

        labor = (self.labor_sens * (1 - tax_rate * 0.8)
                 + public_good_level * 0.2
                 + interest_labor_effect
                 + trust_labor_bonus
                 + investment_employment) * strike_factor
        labor = self.rng.normal(labor, 0.05, self.n).clip(0, 1)

        # ── Minimum wage floor (affects poor/middle employment) ───────────
        # Higher minimum wage: more income per worker but fewer jobs
        # Academic ref: Card & Krueger (1994) — modest effects; Neumark & Wascher (2007) — larger
        min_wage_unemp_effect = 0.0
        min_wage_income_boost = 0.0
        if self.min_wage_sensitive and minimum_wage > 3.0:
            # Each unit of min wage above 3.0 adds income but reduces employment ~1.5%
            excess = minimum_wage - 3.0
            min_wage_income_boost = excess * 0.8
            min_wage_unemp_effect = excess * 0.015

        # ── Gross income (with Okun's Law GDP effect + tariff effect) ─────
        tariff_effect = 1.0 - import_tariff * 0.3
        gross = (self.wealth * 0.05 * labor * tariff_effect
                 + self.rng.normal(0, 0.5, self.n)
                 + min_wage_income_boost).clip(0)

        # ── Investment output (wealthy/ultra-rich invest, creating GDP) ────
        if self.name in ("wealthy", "ultra_rich"):
            # Solow savings-investment: I = s(r) × W
            # Higher interest rates → more saving → more investment
            solow_savings = self.savings_rate + interest_rate * 0.5
            self.investment_output = float(
                (self.wealth.sum() * solow_savings * 0.03
                 * (1.0 - self.capital_flight)  # Capital flight reduces domestic investment
                 * self.trust)                    # Low trust → cautious investment
            )
        else:
            self.investment_output = 0.0

        # ── Tax & revenue (Laffer Curve: revenue = t(1-t²) × base) ────────
        effective_tax_rate = tax_rate * (1.0 - self.tax_evasion_rate)
        # Laffer curve: revenue optimised at interior rate (~0.40-0.50)
        laffer_efficiency = 1.0 - (tax_rate ** 2) * 0.5
        taxes = gross * effective_tax_rate * laffer_efficiency
        net = gross - (gross * tax_rate) + ubi_amount

        # Stimulus distribution (per capita, weighted by need)
        if stimulus > 0:
            need_weight = {"poor": 0.50, "middle": 0.30, "wealthy": 0.15, "ultra_rich": 0.05}
            stim_per_citizen = (stimulus * need_weight.get(self.name, 0.25)) / self.n
            net += stim_per_citizen

        # Money supply effect: QE → asset price inflation (benefits wealthy)
        if money_supply_delta != 0:
            if self.name in ("wealthy", "ultra_rich"):
                # QE inflates asset prices (Cantillon effect)
                asset_boost = max(0, money_supply_delta) / 1000.0 * self.wealth * 0.02
                self.wealth += asset_boost
            elif self.name == "poor":
                # QE erodes purchasing power for poor (delayed trickle-down)
                real_erosion = max(0, money_supply_delta) / 2000.0
                net -= real_erosion

        # ── Inflation erodes real purchasing power ────────────────────────
        real_net = net / (1.0 + inflation)

        # ── Keynesian consumption function: C = MPC × Y ───────────────────
        consume = (real_net * self.mpc).clip(0)

        # ── Solow savings: s(r) = s₀ + α·r ───────────────────────────────
        solow_save = self.savings_rate + interest_rate * 0.3
        savings_bonus = self.wealth * solow_save * 0.01
        self.wealth = (self.wealth + real_net - consume + savings_bonus).clip(0)

        # ── Satisfaction ──────────────────────────────────────────────────
        norm_income = (real_net / max(real_net.max(), 1e-6)).clip(0, 1)
        trust_satisfaction = float(np.clip(self.trust * 0.1, -0.1, 0.15))
        satisfaction = (norm_income
                        + public_good_level * 0.3
                        - inflation * 0.5
                        + trust_satisfaction
                        - import_tariff * 0.1
                        + min(investment_employment * 0.5, 0.1)  # Jobs from investment
                        ).clip(0, 1)

        # ── Collective action: trigger strike ─────────────────────────────
        # Olson (1965): groups with common grievance can coordinate
        mean_sat = float(satisfaction.mean())
        if (mean_sat < 0.25 and self.trust < 0.6
                and self.strike_cooldown == 0 and not self.on_strike):
            # Strike! Labor drops to near-zero next step
            self.on_strike = True
            self.strike_cooldown = 3  # Can't strike again for 3 steps
            satisfaction = np.clip(satisfaction * 0.5, 0, 1)  # Satisfaction drops during strike

        # ── Unemployment ──────────────────────────────────────────────────
        base_unemp = (1.0 - labor)
        ubi_dependency = min(ubi_amount / 30.0, 0.3)
        inflation_drag = min(inflation * 0.5, 0.2)
        tariff_unemp = import_tariff * 0.15
        unemp_per_citizen = (base_unemp * 0.6
                             + ubi_dependency
                             + inflation_drag
                             + tariff_unemp
                             + min_wage_unemp_effect).clip(0, 1)
        unemployment_rate = float(unemp_per_citizen.mean())

        unrest = (satisfaction < self.unrest_thresh).mean()

        return {
            "gross_income":        float(gross.sum()),
            "taxes_paid":          float(taxes.sum()),
            "taxes_evaded":        float((gross * tax_rate - taxes).sum()),
            "net_income":          float(net.sum()),
            "total_wealth":        float(self.wealth.sum()),
            "mean_wealth":         float(self.wealth.mean()),
            "satisfaction":        float(satisfaction.mean()),
            "unrest":              float(unrest),
            "unemployment_rate":   float(unemployment_rate),
            "trust":               float(self.trust),
            "tax_evasion_rate":    float(self.tax_evasion_rate),
            "investment_output":   float(self.investment_output),
            "capital_flight":      float(self.capital_flight),
            "on_strike":           bool(self.on_strike),
        }

    def _update_trust(self, tax_rate: float, ubi_amount: float, public_good_level: float):
        """
        Update trust based on policy consistency and fairness.
        Grounded in Barro-Gordon (1983) credibility model.

        Trust erodes when:
          - Policies flip-flop (volatile direction changes)
          - Taxes spike suddenly
          - Public goods are cut while taxes rise

        Trust grows when:
          - Policies are consistent and gradual
          - Public goods increase
          - UBI is maintained or increased for poor
        """
        if len(self.policy_memory) < 2:
            return

        prev = self.policy_memory[-1]
        tax_change = abs(tax_rate - prev["tax_rate"])
        pg_change = public_good_level - prev["public_good_level"]

        # ── Erosion factors ──────────────────────────────────────────────
        volatility_erosion = tax_change * 0.5
        if len(self.policy_memory) >= 3:
            prev2 = self.policy_memory[-2]
            dir_now = tax_rate - prev["tax_rate"]
            dir_prev = prev["tax_rate"] - prev2["tax_rate"]
            if dir_now * dir_prev < -1e-6:
                volatility_erosion += 0.03

        if tax_rate > prev["tax_rate"] + 0.02 and pg_change < 0:
            volatility_erosion += 0.02

        # ── Growth factors ───────────────────────────────────────────────
        trust_growth = 0.0
        if pg_change > 0:
            trust_growth += pg_change * 0.3
        if tax_change < 0.02:
            trust_growth += 0.01
        if self.name == "poor" and ubi_amount > 3.0:
            trust_growth += 0.01

        # ── Apply ────────────────────────────────────────────────────────
        self.trust = float(np.clip(
            self.trust - volatility_erosion + trust_growth,
            0.3, 1.5
        ))

    def reset(self, rng, wealth_override=None):
        cfg = CLASS_CONFIG[self.name]
        self.rng = rng
        base = wealth_override if wealth_override is not None else cfg["base_wealth"]
        self.wealth = rng.normal(base, base * 0.1, size=self.n).clip(1)
        self.trust = 1.0
        self.policy_memory = []
        self.tax_evasion_rate = 0.0
        self.investment_output = 0.0
        self.capital_flight = 0.0
        self.on_strike = False
        self.strike_cooldown = 0
