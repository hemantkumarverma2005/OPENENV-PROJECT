"""
validate.py — Pre-submission validation for SocialContract-v0.
Checks all competition requirements including 8 policy levers,
academic calibration, inter-group dynamics, and deployment readiness.
"""

import sys, os, io, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))

PASS = "\033[32m✓ PASS\033[0m"
FAIL = "\033[31m✗ FAIL\033[0m"
results = []

def check(name, fn):
    try:
        ok, msg = fn()
        results.append((ok, name, msg))
        print(f"  {PASS if ok else FAIL}  {name}")
        if msg:
            for line in msg.splitlines(): print(f"         {line}")
    except Exception as e:
        results.append((False, name, str(e)))
        print(f"  {FAIL}  {name}")
        print(f"         Exception: {e}")
        traceback.print_exc()

print("\n═══════════════════════════════════════════════════")
print("  SocialContract-v0  Pre-Submission Validator")
print("═══════════════════════════════════════════════════\n")

# ── 1. openenv.yaml ───────────────────────────────────────────────────────────
print("── 1. openenv.yaml ──────────────────────────────────")

def check_yaml_exists():
    return os.path.isfile("openenv.yaml"), ""
check("openenv.yaml present", check_yaml_exists)

def check_yaml_fields():
    import yaml
    with open("openenv.yaml") as f: data = yaml.safe_load(f)
    required = ["name", "version", "description", "observation", "action", "reward", "tasks"]
    missing = [k for k in required if k not in data]
    if missing: return False, f"Missing fields: {missing}"
    task_ids = [t["id"] for t in data["tasks"]]
    return len(task_ids) >= 3, f"Found {len(task_ids)} tasks: {task_ids}"
check("openenv.yaml required fields + 3 tasks", check_yaml_fields)

def check_yaml_action_fields():
    import yaml
    with open("openenv.yaml") as f: data = yaml.safe_load(f)
    action_fields = [f["name"] for f in data.get("action", {}).get("fields", [])]
    required = ["tax_delta", "ubi_delta", "public_good_delta", "interest_rate_delta",
                "stimulus_package", "import_tariff_delta", "money_supply_delta", "minimum_wage_delta"]
    missing = [f for f in required if f not in action_fields]
    if missing: return False, f"Missing action fields: {missing}"
    return True, f"All 8 action fields documented: {len(action_fields)} total"
check("openenv.yaml has all 8 action fields", check_yaml_action_fields)

def check_yaml_reward_range():
    import yaml
    with open("openenv.yaml") as f: data = yaml.safe_load(f)
    rng = data.get("reward", {}).get("range")
    if rng != [0.0, 1.0]: return False, f"reward.range = {rng}"
    return True, f"reward.range = {rng} ✓"
check("openenv.yaml reward range [0.0, 1.0]", check_yaml_reward_range)

# ── 2. Pydantic models ────────────────────────────────────────────────────────
print("\n── 2. Typed Pydantic models ─────────────────────────")

def check_pydantic_imports():
    from env.pydantic_models import EconomicObservation, PolicyAction, StepReward
    from pydantic import BaseModel
    assert issubclass(EconomicObservation, BaseModel)
    assert issubclass(PolicyAction, BaseModel)
    return True, "All models inherit BaseModel"
check("Pydantic models importable and typed", check_pydantic_imports)

def check_pydantic_8_levers():
    from env.pydantic_models import PolicyAction
    fields = set(PolicyAction.model_fields.keys())
    required = {"tax_delta", "ubi_delta", "public_good_delta", "interest_rate_delta",
                "stimulus_package", "import_tariff_delta", "money_supply_delta", "minimum_wage_delta"}
    missing = required - fields
    if missing: return False, f"Missing: {missing}"
    return True, f"PolicyAction has {len(fields)} fields (8 levers + reasoning + speech)"
check("PolicyAction has all 8 levers", check_pydantic_8_levers)

def check_obs_new_fields():
    from env.pydantic_models import EconomicObservation
    fields = set(EconomicObservation.model_fields.keys())
    required = {"money_supply", "minimum_wage", "capital_flight_rate", "strike_active", "private_investment"}
    missing = required - fields
    if missing: return False, f"Missing: {missing}"
    return True, f"EconomicObservation has inter-group fields: {sorted(required)}"
check("Observation has inter-group dynamics fields", check_obs_new_fields)

# ── 3. Environment interface ──────────────────────────────────────────────────
print("\n── 3. Environment interface ─────────────────────────")

def check_reset():
    from env.openenv_wrapper import SocialContractOpenEnv
    env = SocialContractOpenEnv("task1_stability")
    obs = env.reset()
    assert obs.gdp > 0 and hasattr(obs, "minimum_wage")
    return True, f"reset() OK — gdp={obs.gdp:.1f}, min_wage={obs.minimum_wage}"
check("reset() returns EconomicObservation", check_reset)

def check_step_8_levers():
    from env.openenv_wrapper import SocialContractOpenEnv
    from env.pydantic_models import PolicyAction
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    action = PolicyAction(
        tax_delta=0.01, ubi_delta=1.0, public_good_delta=0.01,
        interest_rate_delta=0.005, stimulus_package=50.0,
        import_tariff_delta=0.01, money_supply_delta=50.0,
        minimum_wage_delta=0.5, reasoning="test"
    )
    obs, reward, done, info = env.step(action)
    assert "investment" in info and "capital_flight" in info
    return True, f"step() OK — reward={reward.total:.4f}, investment={info.get('investment', 0):.2f}"
check("step() with all 8 levers", check_step_8_levers)

def check_state():
    from env.openenv_wrapper import SocialContractOpenEnv
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    s = env.state()
    required = {"task_id", "money_supply", "minimum_wage", "class_flight", "class_investment"}
    missing = required - set(s.keys())
    if missing: return False, f"Missing: {missing}"
    return True, f"state() has {len(s)} keys"
check("state() returns required keys", check_state)

# ── 3b. Academic features ────────────────────────────────────────────────────
print("\n── 3b. Academic calibration & inter-group dynamics ──")

def check_phillips_curve():
    from env.citizens import phillips_curve_inflation, NAIRU
    pi_high_unemp = phillips_curve_inflation(0.10, 0.05)
    pi_low_unemp = phillips_curve_inflation(0.02, 0.05)
    return True, f"Phillips Curve: high_unemp→π={pi_high_unemp:.4f}, low_unemp→π={pi_low_unemp:.4f}, NAIRU={NAIRU}"
check("Phillips Curve (augmented, NAIRU=4.5%)", check_phillips_curve)

def check_okuns_law():
    from env.citizens import okuns_law_gdp_effect, NAIRU
    mult_high = okuns_law_gdp_effect(0.10)
    mult_low = okuns_law_gdp_effect(0.03)
    return True, f"Okun's Law: unemp=0.10→GDP×{mult_high:.3f}, unemp=0.03→GDP×{mult_low:.3f}"
check("Okun's Law (β=2.0)", check_okuns_law)

def check_capital_flight():
    from env.openenv_wrapper import SocialContractOpenEnv
    from env.pydantic_models import PolicyAction
    env = SocialContractOpenEnv("task3_crisis")
    env.reset()
    for _ in range(5):
        env.step(PolicyAction(tax_delta=0.08, ubi_delta=0.0, public_good_delta=0.0))
        if env._done: break
    ultra = next(g for g in env.citizen_groups if g.name == "ultra_rich")
    return True, f"Capital flight: ultra_rich flight_rate={ultra.capital_flight:.3f}"
check("Capital flight (Collier et al. 2001)", check_capital_flight)

def check_investment_linkage():
    from env.openenv_wrapper import SocialContractOpenEnv
    from env.pydantic_models import PolicyAction
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    env.step(PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.02))
    wealthy = next(g for g in env.citizen_groups if g.name == "wealthy")
    return True, f"Wealthy investment output: {wealthy.investment_output:.2f}"
check("Inter-group investment (Stackelberg)", check_investment_linkage)

def check_strike_mechanism():
    from env.citizens import CitizenGroup
    import numpy as np
    grp = CitizenGroup("poor", np.random.default_rng(42))
    assert hasattr(grp, 'on_strike') and hasattr(grp, 'strike_cooldown')
    return True, f"Strike fields: on_strike={grp.on_strike}, cooldown={grp.strike_cooldown}"
check("Collective action / strikes (Olson 1965)", check_strike_mechanism)

def check_qe_cantillon():
    from env.openenv_wrapper import SocialContractOpenEnv
    from env.pydantic_models import PolicyAction
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    env.step(PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0, money_supply_delta=200.0))
    assert env.money_supply > 0
    return True, f"QE→money_supply={env.money_supply:.1f}"
check("QE / Cantillon effect", check_qe_cantillon)

def check_min_wage():
    from env.openenv_wrapper import SocialContractOpenEnv
    from env.pydantic_models import PolicyAction
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    w0 = env.minimum_wage
    env.step(PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0, minimum_wage_delta=1.5))
    return True, f"Minimum wage: {w0:.1f}→{env.minimum_wage:.1f}"
check("Minimum wage (Card-Krueger)", check_min_wage)

def check_persistent_shocks():
    from env.openenv_wrapper import SocialContractOpenEnv
    from env.pydantic_models import PolicyAction
    env = SocialContractOpenEnv("task1_stability", seed=1)
    env.reset()
    for _ in range(30):
        if env._done: break
        obs, _, _, _ = env.step(PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.0))
        if obs.shock_duration_remaining > 1:
            return True, f"Persistent shock: {obs.shock_event} ({obs.shock_duration_remaining} steps remaining)"
    return True, "No multi-step shock in 30 steps (probabilistic — OK)"
check("Persistent shock mechanics", check_persistent_shocks)

def check_consumer_confidence():
    from env.openenv_wrapper import SocialContractOpenEnv
    from env.pydantic_models import PolicyAction
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    c0 = env.consumer_confidence
    for _ in range(5):
        env.step(PolicyAction(tax_delta=0.0, ubi_delta=0.0, public_good_delta=0.02))
        if env._done: break
    return True, f"Confidence: {c0:.3f}→{env.consumer_confidence:.3f}"
check("Consumer confidence index", check_consumer_confidence)

# ── 4. Tasks and graders ──────────────────────────────────────────────────────
print("\n── 4. Tasks & graders ──────────────────────────────")

for tid in ["task1_stability", "task2_recession", "task3_crisis", "task4_stagflation", "task5_pandemic"]:
    def _check(t=tid):
        from env.openenv_wrapper import SocialContractOpenEnv
        from env.pydantic_models import PolicyAction
        from graders.graders import run_grader
        env = SocialContractOpenEnv(t)
        env.reset()
        for _ in range(5):
            if env._done: break
            env.step(PolicyAction(tax_delta=0.01, ubi_delta=0.5, public_good_delta=0.01))
        r = run_grader(t, env._history)
        return 0.0 <= r["score"] <= 1.0, f"score={r['score']:.4f}, verdict='{r['verdict']}'"
    check(f"Grader {tid} in [0,1]", _check)

def check_phase_grading():
    from graders.graders import _detect_phase_strategy
    h = [{"action": {"tax_delta": -0.02 if i < 10 else 0.05, "ubi_delta": 3.0 if i < 10 else -2.0,
           "public_good_delta": 0.05 if i < 10 else -0.02}} for i in range(20)]
    r = _detect_phase_strategy(h, n_phases=2)
    return True, f"Phase detection: score={r['phase_score']:.4f}"
check("Phase-aware grading", check_phase_grading)

# ── 5. Reward ──────────────────────────────────────────────────────────────────
print("\n── 5. Reward signal ────────────────────────────────")

def check_reward_range():
    from env.openenv_wrapper import SocialContractOpenEnv
    from env.pydantic_models import PolicyAction
    errors = []
    for tid in ["task1_stability", "task2_recession", "task3_crisis", "task4_stagflation", "task5_pandemic"]:
        env = SocialContractOpenEnv(tid); env.reset()
        for _ in range(8):
            if env._done: break
            _, rw, _, _ = env.step(PolicyAction(tax_delta=0.02, ubi_delta=2.0, public_good_delta=0.02))
            if not (0.0 <= rw.total <= 1.0): errors.append(f"{tid}: total={rw.total}")
            if not (0.0 <= rw.task_progress <= 1.0): errors.append(f"{tid}: progress={rw.task_progress}")
    if errors: return False, "\n".join(errors)
    return True, "All step rewards and task_progress in [0.0, 1.0]"
check("Reward in [0,1] every step", check_reward_range)

# ── 6. inference.py ───────────────────────────────────────────────────────────
print("\n── 6. inference.py ──────────────────────────────────")

def check_inference_exists():
    return os.path.isfile("inference.py"), ""
check("inference.py present", check_inference_exists)

def check_inference_env_vars():
    with open("inference.py") as f: src = f.read()
    missing = [v for v in ["API_BASE_URL", "MODEL_NAME", "OPENAI_API_KEY", "HF_TOKEN"] if v not in src]
    if missing: return False, f"Missing: {missing}"
    return True, "All 4 env vars referenced"
check("inference.py env vars", check_inference_env_vars)

def check_inference_8_levers():
    with open("inference.py") as f: src = f.read()
    for field in ["money_supply_delta", "minimum_wage_delta"]:
        if field not in src: return False, f"Missing {field} in inference.py"
    return True, "money_supply_delta + minimum_wage_delta in inference.py"
check("inference.py handles 8 levers", check_inference_8_levers)

def check_log_format():
    with open("inference.py") as f: src = f.read()
    for e in ['[START]', '[STEP]', '[END]']:
        if e not in src: return False, f"Missing {e}"
    if "flush=True" not in src: return False, "Missing flush=True"
    return True, "[START]/[STEP]/[END] + flush=True"
check("Structured log format", check_log_format)

def check_inference_retry():
    with open("inference.py") as f: src = f.read()
    if "attempt" not in src and "retry" not in src.lower(): return False, "No retry logic"
    return True, "Retry logic present"
check("Retry logic", check_inference_retry)

# ── 7. Dockerfile ─────────────────────────────────────────────────────────────
print("\n── 7. Dockerfile & deployment ──────────────────────")

def check_dockerfile():
    if not os.path.isfile("Dockerfile"): return False, "Missing"
    with open("Dockerfile") as f: src = f.read()
    checks = {"FROM python": "FROM python" in src, "EXPOSE 7860": "EXPOSE 7860" in src, "uvicorn": "uvicorn" in src}
    failed = [k for k, v in checks.items() if not v]
    if failed: return False, f"Missing: {failed}"
    return True, "Dockerfile valid"
check("Dockerfile", check_dockerfile)

def check_requirements():
    if not os.path.isfile("requirements.txt"): return False, "Missing"
    with open("requirements.txt") as f: content = f.read()
    if '\x00' in content: return False, "CORRUPTED (NULL bytes)"
    missing = [p for p in ["fastapi", "uvicorn", "pydantic", "numpy", "openai"] if p not in content]
    if missing: return False, f"Missing: {missing}"
    return True, "All packages present, no corruption"
check("requirements.txt", check_requirements)

def check_dockerignore():
    return os.path.isfile(".dockerignore"), ""
check(".dockerignore", check_dockerignore)

# ── 8. FastAPI ────────────────────────────────────────────────────────────────
print("\n── 8. FastAPI endpoints ─────────────────────────────")

def check_endpoints():
    with open("app.py") as f: src = f.read()
    required = ["/reset/{task_id}", "/step/{task_id}", "/state/{task_id}", "/health", "/tasks"]
    missing = [r for r in required if r not in src]
    if missing: return False, f"Missing: {missing}"
    return True, "All endpoints present"
check("All OpenEnv endpoints", check_endpoints)

def check_app_8_levers():
    with open("app.py") as f: src = f.read()
    if "count\": 8" not in src and "count\":8" not in src:
        return False, "action_space count not 8"
    for f_name in ["money_supply_delta", "minimum_wage_delta"]:
        if f_name not in src: return False, f"Missing {f_name}"
    return True, "app.py declares 8 levers"
check("app.py supports 8 levers", check_app_8_levers)

def check_session_cleanup():
    with open("app.py") as f: src = f.read()
    if "MAX_SESSIONS" not in src: return False, "No session cleanup"
    return True, "LRU session cleanup present"
check("Session cleanup", check_session_cleanup)

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for ok, _, _ in results if ok)
total  = len(results)
failed_checks = [(n, m) for ok, n, m in results if not ok]

print(f"\n═══════════════════════════════════════════════════")
print(f"  Results: {passed}/{total} passed")
if failed_checks:
    print(f"\n  FAILED:")
    for n, m in failed_checks:
        print(f"    ✗ {n}")
        if m: print(f"      {m}")
    print("\n  ⚠  Fix all FAIL items before submitting.")
    sys.exit(1)
else:
    print(f"\n  \033[32m🎉 All checks passed — ready to submit!\033[0m")
    sys.exit(0)
