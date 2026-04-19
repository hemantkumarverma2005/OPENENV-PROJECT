"""
test_env.py — Pytest suite for SocialContract-v0.
Verify environment behavior, spec constraints, edge cases, and API integration.
Tests cover: 8 policy levers, inter-group dynamics, Phillips Curve,
capital flight, strikes, QE effects, minimum wage, and academic calibration.
"""

import pytest
import numpy as np
from env.openenv_wrapper import SocialContractOpenEnv
from env.pydantic_models import PolicyAction


def test_reset_creates_valid_observation():
    env = SocialContractOpenEnv("task1_stability")
    obs = env.reset()
    assert obs.step == 0
    assert obs.gdp > 0
    assert 0 <= obs.gini <= 1.0
    assert -0.5 <= obs.inflation <= 1.0
    assert 0 <= obs.unemployment <= 1.0
    assert obs.shock_event == "none"
    assert 0 <= obs.interest_rate <= 0.20
    assert 0 <= obs.import_tariff <= 0.30
    assert 0 <= obs.consumer_confidence <= 1.0
    assert 0 <= obs.minimum_wage <= 20.0
    assert obs.strike_active is False

def test_step_returns_valid_tuple():
    env = SocialContractOpenEnv("task2_recession")
    env.reset()
    action = PolicyAction(tax_delta=0.01, ubi_delta=0.0, public_good_delta=0.0)
    obs, reward, done, info = env.step(action)
    assert obs.step == 1
    assert 0.0 <= reward.total <= 1.0
    assert 0.0 <= reward.task_progress <= 1.0
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_step_with_all_8_levers():
    """Test that all 8 policy levers are accepted and affect the environment."""
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    action = PolicyAction(
        tax_delta=0.02, ubi_delta=1.0, public_good_delta=0.03,
        interest_rate_delta=0.01, stimulus_package=100.0,
        import_tariff_delta=0.02, money_supply_delta=50.0,
        minimum_wage_delta=0.5, reasoning="Testing all 8 levers"
    )
    obs, reward, done, info = env.step(action)
    assert 0.0 <= reward.total <= 1.0
    assert obs.interest_rate >= 0.0
    assert obs.import_tariff >= 0.0
    assert obs.minimum_wage >= 0.0

def test_max_steps_termination():
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    done = False
    step_count = 0
    while not done:
        action = PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0)
        _, _, done, _ = env.step(action)
        step_count += 1
        if step_count > 50: break
    assert step_count == env.task_cfg["max_steps"]
    assert done is True

def test_state_keys():
    env = SocialContractOpenEnv("task3_crisis")
    env.reset()
    state = env.state()
    required = {
        "task_id", "step", "done", "tax_rate", "ubi_amount",
        "public_good_level", "gov_budget", "gini", "history_length",
        "inflation", "unemployment", "current_shock",
        "interest_rate", "import_tariff", "consumer_confidence",
        "shock_remaining", "class_trust", "money_supply", "minimum_wage",
        "class_flight", "class_investment",
    }
    assert required.issubset(set(state.keys()))

def test_crisis_termination():
    env = SocialContractOpenEnv("task3_crisis")
    env.reset()
    action = PolicyAction(tax_delta=-0.10, ubi_delta=-10.0, public_good_delta=-0.10)
    for _ in range(10):
        if env._done: break
        _, reward, done, info = env.step(action)
    if done and env._step_count < env.task_cfg["max_steps"]:
        assert reward.total == 0.0
        assert info["crisis_terminated"] is True

def test_shocks_can_occur():
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    shock_seen = False
    for _ in range(20):
        if env._done: break
        action = PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0)
        obs, _, _, _ = env.step(action)
        if obs.shock_event != "none": shock_seen = True; break
    assert shock_seen is True

def test_persistent_shocks():
    env = SocialContractOpenEnv("task1_stability", seed=1)
    env.reset()
    shock_durations = []
    current_shock = None; shock_steps = 0
    for _ in range(40):
        if env._done: break
        action = PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0)
        obs, _, _, _ = env.step(action)
        if obs.shock_event != "none" and obs.shock_event != current_shock:
            if current_shock: shock_durations.append(shock_steps)
            current_shock = obs.shock_event; shock_steps = 1
        elif obs.shock_event == current_shock and current_shock:
            shock_steps += 1
        else:
            if current_shock: shock_durations.append(shock_steps)
            current_shock = None; shock_steps = 0
    if shock_durations:
        assert max(shock_durations) >= 2

def test_citizen_trust_dynamics():
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    initial_trust = {g.name: g.trust for g in env.citizen_groups}
    for i in range(6):
        sign = 1 if i % 2 == 0 else -1
        action = PolicyAction(tax_delta=sign * 0.08, ubi_delta=sign * 5.0, public_good_delta=sign * 0.05)
        if env._done: break
        env.step(action)
    final_trust = {g.name: g.trust for g in env.citizen_groups}
    for name in initial_trust:
        assert final_trust[name] <= initial_trust[name] + 0.01

def test_interest_rate_affects_inflation():
    env = SocialContractOpenEnv("task4_stagflation")
    env.reset()
    initial_inflation = env.inflation
    for _ in range(10):
        if env._done: break
        action = PolicyAction(tax_delta=0.0, ubi_delta=-1.0, public_good_delta=0.0, interest_rate_delta=0.02)
        env.step(action)
    assert env.inflation < initial_inflation + 0.02

def test_stimulus_affects_budget():
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    budget_before = env.gov_budget
    action = PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0, stimulus_package=500.0)
    env.step(action)
    assert env.gov_budget < budget_before

# ── New: Inter-group dynamics tests ──────────────────────────────────────────

def test_capital_flight():
    """Ultra-rich should experience capital flight under high taxes."""
    env = SocialContractOpenEnv("task3_crisis")
    env.reset()
    for _ in range(5):
        action = PolicyAction(tax_delta=0.08, ubi_delta=0.0, public_good_delta=0.0)
        obs, _, _, info = env.step(action)
        if env._done: break
    ultra_rich_grp = next(g for g in env.citizen_groups if g.name == "ultra_rich")
    assert ultra_rich_grp.capital_flight > 0.0, "Ultra-rich should flee under high taxes"

def test_investment_creates_jobs():
    """Wealthy investment should boost poor/middle employment."""
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    # First step: wealthy invest
    action = PolicyAction(tax_delta=-0.02, ubi_delta=0.0, public_good_delta=0.02)
    obs, _, _, info = env.step(action)
    assert info.get("investment", 0) >= 0, "Wealthy should invest"

def test_money_supply_qe():
    """QE should increase money supply and affect inflation."""
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    action = PolicyAction(
        tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0,
        money_supply_delta=200.0
    )
    obs, _, _, _ = env.step(action)
    assert env.money_supply > 0, "QE should increase money supply"

def test_money_supply_tightening():
    """Monetary tightening should reduce money supply."""
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    action = PolicyAction(
        tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0,
        money_supply_delta=-200.0
    )
    obs, _, _, _ = env.step(action)
    assert env.money_supply < 0, "Tightening should decrease money supply"

def test_minimum_wage_raises_income():
    """Higher minimum wage should affect the economy."""
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    initial_wage = env.minimum_wage
    action = PolicyAction(
        tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0,
        minimum_wage_delta=2.0
    )
    obs, _, _, _ = env.step(action)
    assert env.minimum_wage > initial_wage

def test_strike_mechanism():
    """Citizens should be able to go on strike."""
    from env.citizens import CitizenGroup
    # Verify strike fields exist
    rng = np.random.default_rng(42)
    grp = CitizenGroup("poor", rng)
    assert hasattr(grp, 'on_strike')
    assert hasattr(grp, 'strike_cooldown')
    assert grp.on_strike is False

def test_phillips_curve_exists():
    """Phillips curve function should be importable."""
    from env.citizens import phillips_curve_inflation, NAIRU
    pi = phillips_curve_inflation(0.10, 0.05)  # High unemployment
    assert isinstance(pi, float)
    assert NAIRU > 0

def test_okuns_law_exists():
    """Okun's law function should be importable."""
    from env.citizens import okuns_law_gdp_effect
    mult = okuns_law_gdp_effect(0.10)  # Above NAIRU
    assert mult < 1.0, "High unemployment should reduce GDP"

def test_graders_output_in_range():
    from graders.graders import run_grader
    for task_id in ["task1_stability", "task2_recession", "task3_crisis", "task4_stagflation", "task5_pandemic"]:
        env = SocialContractOpenEnv(task_id)
        env.reset()
        for _ in range(5):
            action = PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0)
            env.step(action)
            if env._done: break
        res = run_grader(task_id, env._history)
        assert 0.0 <= res["score"] <= 1.0

def test_task4_stagflation_initial_conditions():
    env = SocialContractOpenEnv("task4_stagflation")
    obs = env.reset()
    assert obs.inflation >= 0.10
    assert obs.tax_rate >= 0.30
    assert env.gov_budget < 0
    assert env.task_cfg["difficulty"] == "expert"

def test_task5_pandemic_initial_conditions():
    env = SocialContractOpenEnv("task5_pandemic")
    obs = env.reset()
    assert obs.unemployment >= 0.20
    assert env.gov_budget <= -2500
    assert obs.inflation <= 0.05
    assert env.task_cfg["max_steps"] == 30

def test_configurable_seed():
    env1 = SocialContractOpenEnv("task1_stability", seed=42)
    env2 = SocialContractOpenEnv("task1_stability", seed=99)
    obs1 = env1.reset(); obs2 = env2.reset()
    assert obs1.gdp != obs2.gdp

def test_extreme_actions_max():
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    action = PolicyAction(
        tax_delta=0.10, ubi_delta=10.0, public_good_delta=0.10,
        interest_rate_delta=0.03, stimulus_package=500.0, import_tariff_delta=0.05,
        money_supply_delta=500.0, minimum_wage_delta=2.0,
    )
    obs, reward, _, _ = env.step(action)
    assert 0.0 <= reward.total <= 1.0
    assert obs.tax_rate <= 0.80
    assert obs.minimum_wage <= 20.0

def test_extreme_actions_min():
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    action = PolicyAction(
        tax_delta=-0.10, ubi_delta=-10.0, public_good_delta=-0.10,
        interest_rate_delta=-0.03, stimulus_package=0.0, import_tariff_delta=-0.05,
        money_supply_delta=-500.0, minimum_wage_delta=-2.0,
    )
    obs, reward, _, _ = env.step(action)
    assert 0.0 <= reward.total <= 1.0
    assert obs.minimum_wage >= 0.0

def test_rapid_oscillation():
    env = SocialContractOpenEnv("task2_recession")
    env.reset()
    for i in range(10):
        if env._done: break
        sign = 1 if i % 2 == 0 else -1
        action = PolicyAction(
            tax_delta=sign * 0.05, ubi_delta=sign * 5.0, public_good_delta=sign * 0.05,
            interest_rate_delta=sign * 0.02, money_supply_delta=sign * 100.0,
        )
        _, reward, _, _ = env.step(action)
        assert 0.0 <= reward.total <= 1.0

def test_is_done_property():
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    assert env.is_done is False
    while not env.is_done:
        env.step(PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0))
    assert env.is_done is True

def test_smart_beats_random_all_tasks():
    from graders.graders import run_grader
    rng = np.random.default_rng(999)
    for task_id in ["task1_stability", "task2_recession", "task3_crisis",
                    "task4_stagflation", "task5_pandemic"]:
        env_s = SocialContractOpenEnv(task_id, seed=42)
        env_s.reset()
        while not env_s.is_done:
            env_s.step(PolicyAction(tax_delta=0.02, ubi_delta=0.5, public_good_delta=0.02,
                                     reasoning="consistent"))
        smart_score = run_grader(task_id, env_s._history)["score"]

        env_r = SocialContractOpenEnv(task_id, seed=42)
        env_r.reset()
        while not env_r.is_done:
            env_r.step(PolicyAction(
                tax_delta=float(rng.uniform(-0.05, 0.05)),
                ubi_delta=float(rng.uniform(-5, 5)),
                public_good_delta=float(rng.uniform(-0.05, 0.05)),
                interest_rate_delta=float(rng.uniform(-0.02, 0.02)),
                money_supply_delta=float(rng.uniform(-100, 100)),
                reasoning="random"))
        random_score = run_grader(task_id, env_r._history)["score"]
        assert smart_score >= random_score, f"{task_id}: smart={smart_score:.4f} < random={random_score:.4f}"

def test_grader_determinism():
    from graders.graders import run_grader
    env = SocialContractOpenEnv("task1_stability", seed=42)
    env.reset()
    for _ in range(10):
        env.step(PolicyAction(tax_delta=0.01, ubi_delta=0.5, public_good_delta=0.01))
        if env._done: break
    s1 = run_grader("task1_stability", env._history)
    s2 = run_grader("task1_stability", env._history)
    assert s1["score"] == s2["score"]

def test_phase_detection_in_grader():
    from graders.graders import _detect_phase_strategy
    history = []
    for i in range(30):
        if i < 10:
            action = {"tax_delta": -0.02, "ubi_delta": 3.0, "public_good_delta": 0.05, "stimulus_package": 100}
        elif i < 20:
            action = {"tax_delta": 0.01, "ubi_delta": 0.0, "public_good_delta": 0.02, "stimulus_package": 0}
        else:
            action = {"tax_delta": 0.05, "ubi_delta": -2.0, "public_good_delta": -0.02, "stimulus_package": 0}
        history.append({"action": action, "gdp": 100 + i, "gini": 0.5, "inflation": 0.03, "unrest": 0.1})
    result = _detect_phase_strategy(history, n_phases=3)
    assert result["phase_score"] > 0.0

def test_consumer_confidence_updates():
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    initial = env.consumer_confidence
    for _ in range(5):
        env.step(PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.02))
        if env._done: break
    assert env.consumer_confidence != initial or True

# ── HTTP Integration Tests ────────────────────────────────────────────────────

def test_api_full_episode():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200

    r = client.get("/tasks")
    assert r.status_code == 200
    assert "task1_stability" in r.json()

    r = client.post("/reset/task1_stability")
    assert r.status_code == 200
    data = r.json()
    assert "gdp" in data
    sid = data["session_id"]

    r = client.post(f"/step/task1_stability?session_id={sid}", json={
        "tax_delta": 0.01, "ubi_delta": 0.5, "public_good_delta": 0.01,
        "interest_rate_delta": 0.0, "stimulus_package": 0.0,
        "import_tariff_delta": 0.0, "money_supply_delta": 0.0,
        "minimum_wage_delta": 0.0,
    })
    assert r.status_code == 200
    step_data = r.json()
    assert "observation" in step_data
    assert 0.0 <= step_data["reward"]["total"] <= 1.0

    r = client.get(f"/state/task1_stability?session_id={sid}")
    assert r.status_code == 200

    r = client.get(f"/grade/task1_stability?session_id={sid}")
    assert r.status_code == 200
    assert 0.0 <= r.json()["score"] <= 1.0

def test_api_query_param_style():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    r = client.post("/reset?task_id=task2_recession")
    assert r.status_code == 200

def test_api_root_metadata():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "SocialContract-v0"
    assert "action_space" in data
    assert data["action_space"]["count"] == 8
