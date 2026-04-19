"""
pydantic_models.py
──────────────────
Typed Observation, Action, and Reward models per OpenEnv spec.

Action space: 8 policy levers reflecting the full real-world toolkit:
  1. Fiscal:    tax_delta, ubi_delta, public_good_delta, stimulus_package
  2. Monetary:  interest_rate_delta, money_supply_delta
  3. Trade:     import_tariff_delta
  4. Labor:     minimum_wage_delta
"""

from pydantic import BaseModel, Field
from typing import Optional


class EconomicObservation(BaseModel):
    """Full snapshot of the economy the LLM agent sees each step."""

    step: int = Field(..., description="Current step number (0-indexed)")
    gdp: float = Field(..., description="Total GDP this step")
    gdp_delta: float = Field(..., description="GDP change vs last step")
    gini: float = Field(..., ge=0.0, le=1.0, description="Gini coefficient (0=equal, 1=max inequality)")
    satisfaction: float = Field(..., ge=0.0, le=1.0, description="Mean citizen satisfaction")
    unrest: float = Field(..., ge=0.0, le=1.0, description="Fraction of citizens in unrest")
    gov_budget: float = Field(..., description="Government budget (negative = deficit)")
    tax_rate: float = Field(..., ge=0.0, le=0.8, description="Current tax rate")
    ubi_amount: float = Field(..., ge=0.0, le=50.0, description="Current UBI per citizen per step")
    public_good_level: float = Field(..., ge=0.0, le=1.0, description="Public goods investment level")
    inflation: float = Field(..., ge=-0.5, le=1.0, description="Current inflation rate")
    unemployment: float = Field(..., ge=0.0, le=1.0, description="Fraction of citizens effectively unemployed")
    shock_event: str = Field("none", description="Current exogenous economic shock (or 'none')")
    shock_duration_remaining: int = Field(0, description="Steps remaining for current shock (0=none)")
    interest_rate: float = Field(0.05, ge=0.0, le=0.20, description="Central bank interest rate")
    import_tariff: float = Field(0.0, ge=0.0, le=0.30, description="Import tariff rate")
    consumer_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Aggregate consumer confidence index")
    # ── New: inter-group dynamics ─────────────────────────────────────────
    money_supply: float = Field(0.0, description="Cumulative money supply change (QE/tightening)")
    minimum_wage: float = Field(5.0, ge=0.0, le=20.0, description="Current minimum wage level")
    capital_flight_rate: float = Field(0.0, ge=0.0, le=1.0, description="Ultra-rich capital flight fraction")
    strike_active: bool = Field(False, description="Whether a citizen strike is currently underway")
    private_investment: float = Field(0.0, description="Domestic private investment from wealthy classes")
    # ── Wealth by class ──────────────────────────────────────────────────
    poor_wealth: float = Field(..., description="Mean wealth of the poor class")
    middle_wealth: float = Field(..., description="Mean wealth of the middle class")
    wealthy_wealth: float = Field(..., description="Mean wealth of the wealthy class")
    ultra_rich_wealth: float = Field(..., description="Mean wealth of the ultra-rich class")
    task_id: str = Field(..., description="Current task identifier")
    task_description: str = Field(..., description="Natural language description of the task goal")

    def to_prompt(self) -> str:
        """Convert observation to a natural language economic report for the LLM."""
        gdp_arrow = "↑" if self.gdp_delta > 0 else "↓"
        budget_status = "SURPLUS" if self.gov_budget >= 0 else "DEFICIT"

        # Task-specific warning thresholds matching actual grader gates
        gini_gate = {"task1_stability": 0.43, "task3_crisis": 0.48}.get(self.task_id, 0.55)
        unrest_gate = {"task1_stability": 0.15, "task3_crisis": 0.10, "task2_recession": 0.40}.get(self.task_id, 0.20)
        inflation_gate = {"task4_stagflation": 0.04, "task5_pandemic": 0.06}.get(self.task_id, 0.05)
        unemp_gate = {"task4_stagflation": 0.08, "task5_pandemic": 0.08}.get(self.task_id, 0.15)

        gini_warn = f" ⚠️ BREACH (gate={gini_gate})" if self.gini > gini_gate else ""
        unrest_warn = f" ⚠️ BREACH (gate={unrest_gate})" if self.unrest > unrest_gate else ""
        inflation_warn = f" ⚠️ BREACH (gate={inflation_gate})" if self.inflation > inflation_gate else ""
        unemp_warn = f" ⚠️ BREACH (gate={unemp_gate})" if self.unemployment > unemp_gate else ""
        confidence_warn = " ⚠️ LOW" if self.consumer_confidence < 0.35 else ""
        flight_warn = " ⚠️ FLIGHT" if self.capital_flight_rate > 0.10 else ""

        shock_line = ""
        if self.shock_event != "none":
            shock_line = f"\n⚡ ECONOMIC SHOCK: {self.shock_event} (persisting for {self.shock_duration_remaining} more step(s))\n"

        strike_line = ""
        if self.strike_active:
            strike_line = "\n🚫 LABOR STRIKE UNDERWAY — labor output severely reduced!\n"

        return f"""
ECONOMIC POLICY REPORT — Step {self.step}
{'='*55}
TASK OBJECTIVE: {self.task_description}
{'='*55}
{shock_line}{strike_line}
MACROECONOMIC INDICATORS:
  GDP:              {self.gdp:,.1f}  ({gdp_arrow} {abs(self.gdp_delta):,.1f} from last step)
  Gini Coefficient: {self.gini:.3f}{gini_warn}  (0=equal, 1=max inequality)
  Gov Budget:       {self.gov_budget:,.1f}  [{budget_status}]
  Inflation:        {self.inflation:.3f}{inflation_warn}  (0=stable, higher=prices rising)
  Unemployment:     {self.unemployment:.2f}{unemp_warn}  (fraction of citizens unemployed)
  Interest Rate:    {self.interest_rate:.3f}  (central bank rate)
  Import Tariff:    {self.import_tariff:.3f}  (trade protection level)
  Consumer Conf.:   {self.consumer_confidence:.2f}{confidence_warn}  (0=pessimistic, 1=optimistic)
  Money Supply Δ:   {self.money_supply:+.1f}  (cumulative QE/tightening)
  Minimum Wage:     {self.minimum_wage:.1f}  (wage floor)

SOCIAL INDICATORS:
  Citizen Satisfaction: {self.satisfaction:.2f} / 1.00
  Social Unrest:        {self.unrest:.2f} / 1.00{unrest_warn}

INTER-GROUP DYNAMICS:
  Private Investment:   {self.private_investment:,.1f}  (wealthy → job creation)
  Capital Flight:       {self.capital_flight_rate:.1%}{flight_warn}  (ultra-rich offshore)

CURRENT POLICIES:
  Tax Rate:         {self.tax_rate:.2f}  (0.0 to 0.80)
  UBI Amount:       {self.ubi_amount:.1f}  (0.0 to 50.0 per citizen)
  Public Goods:     {self.public_good_level:.2f}  (0.0 to 1.0)
  Interest Rate:    {self.interest_rate:.3f}  (0.0 to 0.20)
  Import Tariff:    {self.import_tariff:.3f}  (0.0 to 0.30)
  Min Wage:         {self.minimum_wage:.1f}  (0.0 to 20.0)

WEALTH BY CLASS:
  Poor (50 citizens):       {self.poor_wealth:.1f} avg wealth
  Middle (30 citizens):     {self.middle_wealth:.1f} avg wealth
  Wealthy (15 citizens):    {self.wealthy_wealth:.1f} avg wealth
  Ultra-Rich (5 citizens):  {self.ultra_rich_wealth:.1f} avg wealth

INSTRUCTIONS:
You are an economic policy advisor. Recommend policy adjustments.
You have 8 policy levers. Use them strategically based on the current situation.
Respond ONLY with valid JSON in this exact format:
{{
  "tax_delta": <float between -0.10 and 0.10>,
  "ubi_delta": <float between -10.0 and 10.0>,
  "public_good_delta": <float between -0.10 and 0.10>,
  "interest_rate_delta": <float between -0.03 and 0.03>,
  "stimulus_package": <float between 0.0 and 500.0, or 0 for none>,
  "import_tariff_delta": <float between -0.05 and 0.05>,
  "money_supply_delta": <float between -500 and 500, 0=neutral, positive=QE>,
  "minimum_wage_delta": <float between -2.0 and 2.0>,
  "reasoning": "<your reasoning in under 80 words>",
  "policy_speech": "<optional public speech (1 sentence) to calm markets>"
}}
""".strip()


class PolicyAction(BaseModel):
    """Policy adjustment action from the LLM agent — 8 policy levers."""
    # ── Fiscal policy ─────────────────────────────────────────────────────
    tax_delta: float = Field(..., ge=-0.10, le=0.10,
                             description="Change to tax rate (±0.10 max per step)")
    ubi_delta: float = Field(..., ge=-10.0, le=10.0,
                             description="Change to UBI amount (±10.0 max per step)")
    public_good_delta: float = Field(..., ge=-0.10, le=0.10,
                                     description="Change to public goods level (±0.10 max per step)")
    stimulus_package: float = Field(0.0, ge=0.0, le=500.0,
                                     description="One-time fiscal stimulus spending (0=none, max 500)")
    # ── Monetary policy ───────────────────────────────────────────────────
    interest_rate_delta: float = Field(0.0, ge=-0.03, le=0.03,
                                       description="Change to central bank interest rate (±0.03 max)")
    money_supply_delta: float = Field(0.0, ge=-500.0, le=500.0,
                                       description="Quantitative easing (+) or tightening (-). "
                                                    "Positive injects money, inflates assets. "
                                                    "Negative withdraws money, fights inflation.")
    # ── Trade policy ──────────────────────────────────────────────────────
    import_tariff_delta: float = Field(0.0, ge=-0.05, le=0.05,
                                        description="Change to import tariff rate (±0.05 max)")
    # ── Labor policy ──────────────────────────────────────────────────────
    minimum_wage_delta: float = Field(0.0, ge=-2.0, le=2.0,
                                       description="Change to minimum wage (±2.0 per step). "
                                                    "Higher wages boost poor income but may "
                                                    "reduce employment (Card-Krueger debate).")
    # ── Communication ─────────────────────────────────────────────────────
    reasoning: Optional[str] = Field(None, description="Agent's reasoning for the action")
    policy_speech: Optional[str] = Field(None, description="Optional public speech to calm markets")


class StepReward(BaseModel):
    """Structured reward signal after each step."""
    total: float = Field(..., ge=0.0, le=1.0, description="Overall reward this step, normalised to [0.0, 1.0]")
    gdp_component: float = Field(..., description="GDP growth contribution (positive)")
    equality_component: float = Field(..., description="Equality contribution (positive)")
    satisfaction_component: float = Field(..., description="Satisfaction contribution (positive)")
    unrest_penalty: float = Field(..., description="Unrest penalty (negative or zero)")
    deficit_penalty: float = Field(..., description="Budget deficit penalty (negative or zero)")
    volatility_penalty: float = Field(0.0, description="Policy volatility penalty (negative or zero)")
    task_progress: float = Field(..., ge=0.0, le=1.0,
                                 description="Task-specific progress score (0.0-1.0)")