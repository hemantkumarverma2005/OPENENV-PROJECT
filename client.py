"""
client.py - SocialContract-v0 SDK Client

A Pythonic client for the SocialContract-v0 OpenEnv environment.

`reset()` / `step()` use a persistent WebSocket session so they behave like a
normal multi-step environment loop. Raw HTTP endpoints are also available via
`reset_http()` / `step_http()` for one-off stateless calls.

Usage (recommended episode loop):
    from client import SocialContractClient
    with SocialContractClient(base_url="http://localhost:7860") as env:
        obs = env.reset(task_id="task2_recession")
        while not obs["done"]:
            action = {"tax_delta": 0.02, "ubi_delta": 1.0, ...}
            obs = env.step(action)
        print(f"Final reward: {obs['reward']}")

Usage (raw stateless HTTP snapshot):
    with SocialContractClient(base_url="http://localhost:7860") as env:
        reset_resp = env.reset_http(task_id="task3_crisis")
        step_resp = env.step_http(
            {"tax_delta": 0.01, "ubi_delta": 0.0, "public_good_delta": 0.0},
            task_id="task3_crisis",
        )
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

try:
    import httpx
    import websockets

    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


class SocialContractClient:
    """
    Client for the SocialContract-v0 OpenEnv environment.

    `reset()` / `step()` provide a stateful client loop backed by a persistent
    WebSocket. Raw HTTP helpers remain available for schema, health, grading,
    and single-request inspection.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        if not _DEPS_OK:
            raise ImportError(
                "Install client dependencies: pip install httpx websockets"
            )
        self.base_url = base_url.rstrip("/")
        self._http: Optional[httpx.Client] = None
        self._ws = None
        self._task_id = "task1_stability"
        self._seed = 42
        self._ws_uri = self.base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        ) + "/ws"

    def __enter__(self) -> "SocialContractClient":
        self._http = httpx.Client(base_url=self.base_url, timeout=60.0)
        return self

    def __exit__(self, *args) -> None:
        if self._ws is not None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self._close_ws_async())
            else:
                loop.create_task(self._close_ws_async())
        if self._http:
            self._http.close()

    def _run_sync(self, coro_factory):
        """Run an async helper from sync code without leaking coroutines."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro_factory())
        raise RuntimeError(
            "This event loop is already running. Use the async client methods "
            "`areset()` / `astep()` / `aws_reset()` / `aws_step()` instead."
        )

    def _normalize_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten SDK envelopes into a single observation dict."""
        if "observation" in payload:
            obs = dict(payload["observation"])
            if "done" in payload:
                obs["done"] = payload["done"]
            if "reward" in payload:
                obs["reward"] = payload["reward"]
            return obs
        return payload

    # Stateful episode API -------------------------------------------------
    def reset(
        self,
        task_id: str = "task1_stability",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Reset the persistent episode session."""
        return self._run_sync(lambda: self.areset(task_id=task_id, seed=seed))

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Step the persistent episode session."""
        return self._run_sync(lambda: self.astep(action))

    async def areset(
        self,
        task_id: str = "task1_stability",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Async version of reset()."""
        self._task_id = task_id
        self._seed = seed
        return await self._ws_reset_async(task_id=task_id, seed=seed)

    async def astep(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of step()."""
        return await self._ws_step_async(action)

    # Raw HTTP API ---------------------------------------------------------
    def reset_http(
        self,
        task_id: str = "task1_stability",
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Call the raw HTTP /reset endpoint.
        This is stateless and returns the full SDK response envelope.
        """
        assert self._http, "Use as context manager: `with SocialContractClient(...) as env:`"
        resp = self._http.post("/reset", json={"seed": seed, "task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    def step_http(
        self,
        action: Dict[str, Any],
        *,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Call the raw HTTP /step endpoint.
        This is stateless and best for one-off step previews, not full episodes.
        """
        assert self._http, "Use as context manager: `with SocialContractClient(...) as env:`"
        payload: Dict[str, Any] = {"action": action}
        if task_id is not None:
            payload["task_id"] = task_id
        if seed is not None:
            payload["seed"] = seed
        resp = self._http.post("/step", json=payload)
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> Dict[str, Any]:
        """List all available tasks."""
        assert self._http
        return self._http.get("/tasks").json()

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        assert self._http
        return self._http.get("/health").json()

    def schema(self) -> Dict[str, Any]:
        """Get JSON schema for action and observation."""
        assert self._http
        return self._http.get("/schema").json()

    def grade(self, task_id: str, history: list) -> Dict[str, Any]:
        """Grade a completed episode history."""
        assert self._http
        resp = self._http.post(f"/grade/{task_id}", json=history)
        resp.raise_for_status()
        return resp.json()

    # WebSocket API --------------------------------------------------------
    def ws_reset(
        self,
        task_id: str = "task1_stability",
        seed: int = 42,
        ws=None,
    ) -> Dict[str, Any]:
        """Sync wrapper for a persistent WebSocket reset."""
        return self._run_sync(lambda: self.aws_reset(task_id=task_id, seed=seed, ws=ws))

    def ws_step(self, action: Dict[str, Any], ws=None) -> Dict[str, Any]:
        """Sync wrapper for a persistent WebSocket step."""
        return self._run_sync(lambda: self.aws_step(action, ws=ws))

    async def aws_reset(
        self,
        task_id: str = "task1_stability",
        seed: int = 42,
        ws=None,
    ) -> Dict[str, Any]:
        """Async version of ws_reset()."""
        self._task_id = task_id
        self._seed = seed
        return await self._ws_reset_async(task_id, seed, ws)

    async def aws_step(self, action: Dict[str, Any], ws=None) -> Dict[str, Any]:
        """Async version of ws_step()."""
        return await self._ws_step_async(action, ws)

    async def _ensure_ws_async(self):
        if self._ws is None or getattr(self._ws, "closed", False):
            import websockets as _ws

            self._ws = await _ws.connect(self._ws_uri)
        return self._ws

    async def _close_ws_async(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def _ws_reset_async(self, task_id, seed, ws=None):
        conn = ws or await self._ensure_ws_async()
        await conn.send(
            json.dumps(
                {
                    "type": "reset",
                    "data": {"seed": seed, "task_id": task_id},
                }
            )
        )
        msg = json.loads(await conn.recv())
        return self._normalize_response(msg.get("data", msg))

    async def _ws_step_async(self, action, ws=None):
        conn = ws or await self._ensure_ws_async()
        await conn.send(json.dumps({"type": "step", "data": action}))
        msg = json.loads(await conn.recv())
        return self._normalize_response(msg.get("data", msg))


if __name__ == "__main__":
    print("SocialContract-v0 Client - Quick Test")
    print("Starting test against http://localhost:7860 ...")
    try:
        with SocialContractClient() as env:
            print(f"Health: {env.health()}")
            print(f"Tasks: {list(env.tasks().keys())}")
            obs = env.reset(task_id="task1_stability")
            print(
                f"Reset OK - step={obs['step']}, gdp={obs['gdp']:.1f}, "
                f"gini={obs['gini']:.3f}"
            )
            action = {
                "tax_delta": 0.01,
                "ubi_delta": 0.5,
                "public_good_delta": 0.01,
                "interest_rate_delta": 0.0,
                "stimulus_package": 0.0,
                "import_tariff_delta": 0.0,
                "money_supply_delta": 0.0,
                "minimum_wage_delta": 0.0,
                "reasoning": "client test step",
            }
            obs = env.step(action)
            print(
                f"Step OK - reward={obs['reward']:.4f}, done={obs['done']}, "
                f"gdp={obs['gdp']:.1f}"
            )
        print("All good!")
    except Exception as e:
        print(f"Error (is the server running?): {e}")
