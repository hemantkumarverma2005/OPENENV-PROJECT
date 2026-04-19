"""
app.py
──────
FastAPI server exposing the OpenEnv interface as HTTP endpoints.
Deployed as a HuggingFace Space (Docker SDK).

Session isolation: each client gets a unique session_id returned on /reset.
Pass session_id on subsequent /step, /state, /grade calls.

Features:
  - Session cleanup (max 100 sessions, LRU eviction)
  - Multiple parameter styles for compatibility (path, query, body)
  - All 6 policy levers supported

Supports all parameter styles for maximum compatibility with OpenEnv checkers:
  - Path param:  POST /reset/{task_id}
  - Query param: POST /reset?task_id=task1_stability
  - Body JSON:   POST /reset  {"task_id": "task1_stability"}
"""

import sys, os, uuid, time
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn

from env.openenv_wrapper import SocialContractOpenEnv
from env.pydantic_models import PolicyAction
from graders.graders import run_grader

app = FastAPI(
    title       = "SocialContract-v0 OpenEnv",
    description = "Economic Policy Advisory Environment — OpenEnv compliant. "
                  "6 policy levers: tax, UBI, public goods, interest rate, stimulus, tariffs.",
    version     = "1.0.0",
)

# session_key -> (SocialContractOpenEnv, last_access_time)
MAX_SESSIONS = 100
_sessions: dict[str, tuple[SocialContractOpenEnv, float]] = {}


# ── Request body models ──────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str
    session_id: Optional[str] = None
    seed: Optional[int] = None

class StepRequest(BaseModel):
    task_id: str
    action: Optional[PolicyAction] = None
    # Also allow flat action fields for flexibility
    tax_delta: Optional[float] = None
    ubi_delta: Optional[float] = None
    public_good_delta: Optional[float] = None
    interest_rate_delta: Optional[float] = None
    stimulus_package: Optional[float] = None
    import_tariff_delta: Optional[float] = None
    money_supply_delta: Optional[float] = None
    minimum_wage_delta: Optional[float] = None
    reasoning: Optional[str] = None
    session_id: Optional[str] = None

class StateRequest(BaseModel):
    task_id: str
    session_id: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _session_key(task_id: str, session_id: Optional[str]) -> str:
    return session_id if session_id else f"__default__{task_id}"


def _cleanup_sessions():
    """Evict oldest sessions if we exceed MAX_SESSIONS (LRU)."""
    if len(_sessions) > MAX_SESSIONS:
        sorted_keys = sorted(_sessions, key=lambda k: _sessions[k][1])
        to_remove = sorted_keys[:len(_sessions) - MAX_SESSIONS + 10]
        for k in to_remove:
            del _sessions[k]


def get_env(task_id: str, session_id: Optional[str]) -> SocialContractOpenEnv:
    if task_id not in SocialContractOpenEnv.TASKS:
        raise HTTPException(400, f"Unknown task_id: {task_id}")
    key = _session_key(task_id, session_id)
    if key not in _sessions:
        raise HTTPException(400, "Session not found. Call /reset first.")
    env, _ = _sessions[key]
    _sessions[key] = (env, time.time())  # Update access time
    return env


# ── Core logic ───────────────────────────────────────────────────────────────

def _do_reset(task_id: str, session_id: Optional[str] = None, seed: Optional[int] = None):
    if task_id not in SocialContractOpenEnv.TASKS:
        raise HTTPException(400, f"Unknown task_id: {task_id}. Available: {list(SocialContractOpenEnv.TASKS.keys())}")

    _cleanup_sessions()

    sid = session_id or str(uuid.uuid4())
    key = _session_key(task_id, sid)

    env = SocialContractOpenEnv(task_id)
    _sessions[key] = (env, time.time())
    obs = env.reset(seed=seed)

    result = obs.model_dump()
    result["session_id"] = sid
    return result


def _do_step(task_id: str, action: PolicyAction, session_id: Optional[str] = None):
    env = get_env(task_id, session_id)
    if env.is_done:
        raise HTTPException(400, "Episode finished. Call /reset first.")

    try:
        obs, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(400, f"Invalid action: {str(e)}")

    return {
        "observation": obs.model_dump(),
        "reward":      reward.model_dump(),
        "done":        done,
        "info":        info,
    }


def _do_state(task_id: str, session_id: Optional[str] = None):
    env = get_env(task_id, session_id)
    return env.state()


def _do_grade(task_id: str, session_id: Optional[str] = None):
    env = get_env(task_id, session_id)
    result = run_grader(task_id, env._history)
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name":    "SocialContract-v0",
        "version": "1.0.0",
        "status":  "ok",
        "tasks":   list(SocialContractOpenEnv.TASKS.keys()),
        "action_space": {
            "levers": ["tax_delta", "ubi_delta", "public_good_delta",
                        "interest_rate_delta", "stimulus_package", "import_tariff_delta",
                        "money_supply_delta", "minimum_wage_delta"],
            "count": 8,
        },
    }


# ── RESET (3 styles) ─────────────────────────────────────────────────────────

@app.post("/reset/{task_id}")
def reset_with_path(
    task_id: str,
    session_id: Optional[str] = Query(default=None),
    seed: Optional[int] = Query(default=None),
):
    """Reset environment (task_id in path)."""
    return _do_reset(task_id, session_id, seed)


@app.post("/reset")
async def reset_generic(request: Request):
    """Reset environment — accepts task_id from query param OR JSON body.
    If no task_id is provided, defaults to the first available task."""
    # Try query params first
    task_id = request.query_params.get("task_id")
    session_id = request.query_params.get("session_id")
    seed_str = request.query_params.get("seed")
    seed = int(seed_str) if seed_str else None

    # If no task_id in query, try JSON body
    if not task_id:
        try:
            body = await request.json()
            task_id = body.get("task_id")
            session_id = session_id or body.get("session_id")
            seed = seed or body.get("seed")
        except Exception:
            body = {}

    # If still no task_id, try form data
    if not task_id:
        try:
            form = await request.form()
            task_id = form.get("task_id")
            session_id = session_id or form.get("session_id")
            seed_val = form.get("seed")
            if seed_val:
                seed = int(seed_val)
        except Exception:
            pass

    # Default to the first available task if none specified
    if not task_id:
        task_id = list(SocialContractOpenEnv.TASKS.keys())[0]

    return _do_reset(task_id, session_id, seed)


# ── STEP (3 styles) ──────────────────────────────────────────────────────────

@app.post("/step/{task_id}")
def step_with_path(
    task_id: str,
    action: PolicyAction,
    session_id: Optional[str] = Query(default=None),
):
    """Apply action (task_id in path)."""
    return _do_step(task_id, action, session_id)


@app.post("/step")
async def step_generic(request: Request):
    """Apply action — accepts task_id from query param OR JSON body."""
    task_id = request.query_params.get("task_id")
    session_id = request.query_params.get("session_id")

    try:
        body = await request.json()
    except Exception:
        body = {}

    if not task_id:
        task_id = body.get("task_id")
    if not session_id:
        session_id = body.get("session_id")

    if not task_id:
        raise HTTPException(400, "task_id is required")

    # Extract action from body — could be nested under "action" key or flat
    action_data = body.get("action", {})
    if not action_data:
        # Try flat fields
        action_data = {
            "tax_delta": body.get("tax_delta", 0.0),
            "ubi_delta": body.get("ubi_delta", 0.0),
            "public_good_delta": body.get("public_good_delta", 0.0),
            "interest_rate_delta": body.get("interest_rate_delta", 0.0),
            "stimulus_package": body.get("stimulus_package", 0.0),
            "import_tariff_delta": body.get("import_tariff_delta", 0.0),
            "money_supply_delta": body.get("money_supply_delta", 0.0),
            "minimum_wage_delta": body.get("minimum_wage_delta", 0.0),
            "reasoning": body.get("reasoning", ""),
        }

    action = PolicyAction(**action_data)
    return _do_step(task_id, action, session_id)


# ── STATE (3 styles) ─────────────────────────────────────────────────────────

@app.get("/state/{task_id}")
def state_with_path(
    task_id: str,
    session_id: Optional[str] = Query(default=None),
):
    """Return full environment state (task_id in path)."""
    return _do_state(task_id, session_id)


@app.get("/state")
def state_generic(
    task_id: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
):
    """Return full environment state (task_id as query param)."""
    if not task_id:
        raise HTTPException(400, "task_id is required")
    return _do_state(task_id, session_id)


# ── GRADE (3 styles) ─────────────────────────────────────────────────────────

@app.get("/grade/{task_id}")
def grade_with_path(
    task_id: str,
    session_id: Optional[str] = Query(default=None),
):
    """Grade the current episode (task_id in path)."""
    return _do_grade(task_id, session_id)


@app.get("/grade")
def grade_generic(
    task_id: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
):
    """Grade the current episode (task_id as query param)."""
    if not task_id:
        raise HTTPException(400, "task_id is required")
    return _do_grade(task_id, session_id)


# ── Other endpoints ──────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata."""
    return {
        task_id: {
            "description": cfg["description"][:120] + "...",
            "difficulty":  cfg["difficulty"],
            "max_steps":   cfg["max_steps"],
        }
        for task_id, cfg in SocialContractOpenEnv.TASKS.items()
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/summary")
def summary():
    """
    Run a random-action baseline against all 5 tasks and return scores.
    Deterministic: seeded RNG, same result every call.
    """
    import numpy as np
    from env.pydantic_models import PolicyAction

    rng = np.random.default_rng(0)
    results = {}

    for task_id in SocialContractOpenEnv.TASKS:
        env = SocialContractOpenEnv(task_id)
        env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = PolicyAction(
                tax_delta         = float(rng.uniform(-0.05, 0.05)),
                ubi_delta         = float(rng.uniform(-5.0,  5.0)),
                public_good_delta = float(rng.uniform(-0.05, 0.05)),
                interest_rate_delta = float(rng.uniform(-0.02, 0.02)),
                stimulus_package  = float(rng.uniform(0.0, 100.0)),
                import_tariff_delta = float(rng.uniform(-0.03, 0.03)),
                money_supply_delta = float(rng.uniform(-100.0, 100.0)),
                minimum_wage_delta = float(rng.uniform(-1.0, 1.0)),
                reasoning         = "random baseline",
            )
            _, reward, done, _ = env.step(action)
            total_reward += reward.total

        grade = run_grader(task_id, env._history)
        results[task_id] = {
            "difficulty":    SocialContractOpenEnv.TASKS[task_id]["difficulty"],
            "steps_run":     env._step_count,
            "random_score":  grade["score"],
            "verdict":       grade["verdict"],
            "total_reward":  round(total_reward, 4),
        }

    mean_score = round(sum(r["random_score"] for r in results.values()) / len(results), 4)

    return {
        "description": (
            "Random-action baseline. A capable LLM agent should "
            "significantly outperform these on all five tasks."
        ),
        "random_baseline": results,
        "random_mean_score": mean_score,
        "environment": "SocialContract-v0",
        "note": "Scores are deterministic — seeded RNG, same result every call.",
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)