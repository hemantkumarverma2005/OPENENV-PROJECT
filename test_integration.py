"""Quick integration test for all new features."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from env.openenv_wrapper import SocialContractOpenEnv
from env.pydantic_models import PolicyAction
from env.market_agent import MarketConsortium, MarketSpeculator
from env.curriculum import AdaptiveCurriculum, DIFFICULTY_LEVELS

def test_basic():
    env = SocialContractOpenEnv("task1_stability")
    obs = env.reset()
    print(f"[PASS] Basic env: GDP={obs.gdp:.1f}, Gini={obs.gini:.3f}")

def test_market_agent():
    env = SocialContractOpenEnv("task4_stagflation", enable_market_agent=True)
    obs = env.reset()
    a = PolicyAction(
        tax_delta=0.01, ubi_delta=1.0, public_good_delta=0.02,
        interest_rate_delta=0.01, stimulus_package=50,
        import_tariff_delta=0.0, money_supply_delta=0,
        minimum_wage_delta=0, reasoning="test",
    )
    obs = env.step(a)
    mr = obs.metadata.get("market_response")
    assert mr is not None
    print(f"[PASS] Market agent: confidence={mr['market_confidence']:.3f}, "
          f"attack={mr.get('attack_triggered', False)}")

def test_curriculum():
    cur = AdaptiveCurriculum()
    for i in range(10):
        result = cur.update(0.75)
    assert cur.current_level > 0
    print(f"[PASS] Curriculum: level={cur.current_level} ({cur.get_difficulty().name})")

def test_difficulty_levels():
    env = SocialContractOpenEnv("task1_stability", difficulty_level=3)
    obs = env.reset()
    print(f"[PASS] Difficulty L3: budget={obs.gov_budget:.0f}, "
          f"inflation={obs.inflation:.3f}")

def test_market_consortium():
    import numpy as np
    rng = np.random.default_rng(42)
    mc = MarketConsortium(rng)
    resp = mc.observe_and_act(
        {"gdp": 300, "gdp_delta": 10, "inflation": 0.08,
         "unemployment": 0.15, "gov_budget": -2000, "unrest": 0.1,
         "interest_rate": 0.05, "shock_event": "none"},
        {"tax_delta": 0.02, "interest_rate_delta": 0.01},
    )
    assert "agents" in resp
    print(f"[PASS] Consortium: {len(resp['agents'])} agents, "
          f"agg_confidence={resp['market_confidence']:.3f}")

def test_full_episode_with_features():
    from demo import smart_policy
    from graders.graders import run_grader
    env = SocialContractOpenEnv("task4_stagflation", seed=42,
                                 enable_market_agent=True, difficulty_level=1)
    obs = env.reset()
    while not env.is_done:
        action = smart_policy(obs)
        obs = env.step(action)
    grade = run_grader("task4_stagflation", env._history)
    print(f"[PASS] Full episode with market agent: score={grade['score']:.4f}")

if __name__ == "__main__":
    tests = [test_basic, test_market_agent, test_curriculum,
             test_difficulty_levels, test_market_consortium,
             test_full_episode_with_features]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
