"""
app.py — SocialContract-v0 OpenEnv Server
──────────────────────────────────────────
Built on the openenv-core SDK using create_app().

SDK provides automatically:
  POST /reset          — reset episode (stateless HTTP)
  POST /step           — step episode (stateless HTTP)
  GET  /state          — current env state
  WS   /ws             — WebSocket persistent sessions (concurrent)
  GET  /schema         — JSON schemas for action / observation
  GET  /metadata       — environment metadata
  GET  /health         — health check
  GET  /docs           — Swagger UI
  GET  /web            — interactive Gradio web UI (set ENABLE_WEB_INTERFACE=true)

Custom extensions added below:
  GET  /tasks          — list all 5 tasks
  GET  /full_state     — rich state dict (for Gradio demo & graders)
  GET  /summary        — random-baseline benchmark summary
  POST /grade/{task_id} — grade a history dict
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

try:
    from openenv.core.env_server.http_server import create_app as _sdk_create_app
    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SDK_AVAILABLE = False

from env.openenv_wrapper import SocialContractOpenEnv
from env.pydantic_models import PolicyAction, EconomicObservation
from graders.graders import run_grader


# ── Build base app via SDK (gives us /reset /step /ws /schema /health etc.) ──
if _SDK_AVAILABLE:
    app = _sdk_create_app(
        SocialContractOpenEnv,          # factory: called with no args → task1_stability
        PolicyAction,
        EconomicObservation,
        max_concurrent_envs=100,        # up to 100 concurrent WebSocket sessions
    )
else:  # pragma: no cover — fallback without SDK
    app = FastAPI(title="SocialContract-v0", version="1.0.0")

# Preserve the /grade endpoint session-compatibility that some tests rely on
# (keeps MAX_SESSIONS reference for validate.py check)
MAX_SESSIONS = 100

@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Extension: /tasks
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/tasks", tags=["Environment Info"])
def list_tasks():
    """List all 5 available tasks with difficulty and step budget."""
    return {
        task_id: {
            "description": cfg["description"][:200] + "…",
            "difficulty": cfg["difficulty"],
            "max_steps": cfg["max_steps"],
        }
        for task_id, cfg in SocialContractOpenEnv.TASKS.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Custom Extension: /full_state  (rich dict for Gradio / graders)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/full_state", tags=["State Management"])
def full_state():
    """
    Returns rich environment state dict (for debugging / Gradio demo).
    Starts a fresh task1_stability env — for session state use WebSocket /ws.
    """
    env = SocialContractOpenEnv("task1_stability")
    env.reset()
    return JSONResponse(content=env._full_state())


# ─────────────────────────────────────────────────────────────────────────────
# Custom Extension: /grade/{task_id}  (grade a submitted history)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/grade/{task_id}", tags=["Environment Info"])
def grade_history(task_id: str, history: list):
    """
    Grade a submitted episode history against the task-specific rubric.
    POST the history list (same format as env._history).
    """
    if task_id not in SocialContractOpenEnv.TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}")
    result = run_grader(task_id, history)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Custom Extension: /summary  (random baseline benchmark)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/summary", tags=["Environment Info"])
def random_baseline_summary():
    """
    Runs 1 episode of random policy per task and returns graded scores.
    Useful as a sanity check — smart agents should outperform this.
    """
    rng = np.random.default_rng(0)
    results = {}
    for task_id in SocialContractOpenEnv.TASKS:
        env = SocialContractOpenEnv(task_id)
        env.reset()
        total_reward = 0.0
        while not env.is_done:
            action = PolicyAction(
                tax_delta=float(rng.uniform(-0.05, 0.05)),
                ubi_delta=float(rng.uniform(-5.0, 5.0)),
                public_good_delta=float(rng.uniform(-0.05, 0.05)),
                interest_rate_delta=float(rng.uniform(-0.02, 0.02)),
                stimulus_package=float(rng.uniform(0.0, 100.0)),
                import_tariff_delta=float(rng.uniform(-0.03, 0.03)),
                money_supply_delta=float(rng.uniform(-100.0, 100.0)),
                minimum_wage_delta=float(rng.uniform(-1.0, 1.0)),
                reasoning="random baseline",
            )
            obs = env.step(action)
            total_reward += obs.reward or 0.0

        grade = run_grader(task_id, env._history)
        results[task_id] = {
            "difficulty": SocialContractOpenEnv.TASKS[task_id]["difficulty"],
            "steps_run": env._step_count,
            "random_score": round(grade["score"], 4),
            "verdict": grade["verdict"],
            "total_reward": round(total_reward, 4),
        }

    mean_score = round(
        sum(r["random_score"] for r in results.values()) / len(results), 4
    )
    return {
        "description": "Random-action baseline across all 5 tasks.",
        "note": "action_space: 8 levers — tax, UBI, public_good, interest_rate, "
                "stimulus, import_tariff, money_supply, minimum_wage",
        "action_space": {"count": 8, "fields": [
            "tax_delta", "ubi_delta", "public_good_delta", "interest_rate_delta",
            "stimulus_package", "import_tariff_delta", "money_supply_delta",
            "minimum_wage_delta",
        ]},
        "random_baseline": results,
        "random_mean_score": mean_score,
        "environment": "SocialContract-v0",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)