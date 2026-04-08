"""
app.py — FastAPI Server for Email Triage OpenEnv HF Space
==========================================================
Exposes the MyEnvV4Env as an HTTP API so the pre-submission validation
script (and automated HF Space ping) can verify the environment is live.

Endpoints:
  POST /reset            — reset the environment, returns initial observation
  POST /step             — submit an action, returns step result
  GET  /state            — returns current environment state
  GET  /health           — liveness probe
  GET  /tasks            — list available tasks with descriptions

The validation script pings:
  POST $PING_URL/reset   with body {}  and expects HTTP 200
"""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from my_env_v4 import MyEnvV4Action, MyEnvV4Env

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "Real-world email triage environment for agent benchmarking. "
        "Three tasks: URGENT classification (easy), department routing (medium), "
        "and full triage (hard)."
    ),
    version="1.0.0",
)

# ── Global env instance (single-session for demo / validation purposes) ────────
# For production multi-user deployments, replace with a session map.
_env: Optional[MyEnvV4Env] = None


# ─── Helpers ──────────────────────────────────────────────────────────────────


async def _parse_body(request: Request) -> Dict[str, Any]:
    """Safely parse JSON body; return empty dict on failure."""
    try:
        body = await request.json()
        return body if isinstance(body, dict) else {}
    except Exception:
        return {}


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Liveness probe — always returns 200 when server is up."""
    return {"status": "ok", "env": "email_triage_env", "version": "1.0.0"}


@app.get("/tasks")
async def list_tasks():
    """List available tasks with descriptions and difficulty."""
    return {
        "tasks": [
            {
                "name": "email_urgent",
                "difficulty": "easy",
                "description": "Classify each email as URGENT or NOT_URGENT.",
                "num_emails": 5,
                "max_reward": 5.0,
                "action_format": "URGENT or NOT_URGENT",
            },
            {
                "name": "email_route",
                "difficulty": "medium",
                "description": "Route each email to the correct department.",
                "num_emails": 5,
                "max_reward": 5.0,
                "action_format": "BILLING | TECH_SUPPORT | SALES | GENERAL",
            },
            {
                "name": "email_triage",
                "difficulty": "hard",
                "description": "Full triage: priority + department + response.",
                "num_emails": 3,
                "max_reward": 3.0,
                "action_format": "Priority P1/P2/P3, Department, and response message",
            },
        ]
    }


@app.post("/reset")
async def reset(request: Request):
    """
    Reset the environment and return the initial observation.

    Body (optional JSON):
      { "task": "email_urgent" | "email_route" | "email_triage" }

    Returns:
      StepResult as JSON with observation.echoed_message, reward (None), done (false)
    """
    global _env
    body = await _parse_body(request)
    task_name = body.get("task", os.getenv("MY_ENV_V4_TASK", "email_urgent"))

    # Validate task name
    valid_tasks = {"email_urgent", "email_route", "email_triage"}
    if task_name not in valid_tasks:
        task_name = "email_urgent"

    os.environ["MY_ENV_V4_TASK"] = task_name
    _env = MyEnvV4Env(task_name=task_name)
    result = await _env.reset()

    return JSONResponse(content=result.model_dump(), status_code=200)


@app.post("/step")
async def step(request: Request):
    """
    Submit one agent action and receive the next observation + reward.

    Body (required JSON):
      { "message": "<agent response string>" }

    Returns:
      StepResult as JSON with observation.echoed_message, reward (float), done (bool)
    """
    global _env
    body = await _parse_body(request)
    message = body.get("message", "")

    if _env is None:
        # Auto-create env if /reset was never called; default to email_urgent
        task_name = os.getenv("MY_ENV_V4_TASK", "email_urgent")
        _env = MyEnvV4Env(task_name=task_name)
        await _env.reset()

    action = MyEnvV4Action(message=message)
    result = await _env.step(action)

    return JSONResponse(content=result.model_dump(), status_code=200)


@app.get("/state")
async def state():
    """Return current environment state snapshot."""
    global _env
    if _env is None:
        return JSONResponse(
            content={"error": "No active environment. Call /reset first."},
            status_code=200,
        )
    snapshot = await _env.state()
    return JSONResponse(content=snapshot, status_code=200)

if __name__ == "__main__":
    import uvicorn
    # Using port 8000 to avoid conflicts with other apps (like Gradio) which often use 7860
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
