"""
inference.py — Email Triage Environment Baseline Inference Script
=================================================================
MANDATORY environment variables:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (strictly followed):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Rules:
  - One [START] line at episode begin.
  - One [STEP] line per step, immediately after env.step() returns.
  - One [END] line after env.close(), always emitted (even on exception).
  - reward and rewards formatted to 2 decimal places.
  - done and success are lowercase booleans: true or false.
  - error is the raw error string, or null if none.
  - score is formatted to 3 decimal places.
  - All fields on a single line — no embedded newlines.

Example output:
  [START] task=email_urgent env=email_triage_env model=Qwen/Qwen2.5-72B-Instruct
  [STEP] step=1 action=URGENT reward=1.00 done=false error=null
  [STEP] step=2 action=NOT_URGENT reward=1.00 done=false error=null
  [END] success=true steps=2 score=0.400 rewards=1.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from my_env_v4 import MyEnvV4Action, MyEnvV4Env

# ─── Configuration ────────────────────────────────────────────────────────────

IMAGE_NAME: Optional[str] = os.getenv("IMAGE_NAME")            # Docker image (optional)
API_KEY: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK: str = os.getenv("MY_ENV_V4_BENCHMARK", "email_triage_env")
# Note: active task is set per-run via MY_ENV_V4_TASK env var (see TASKS list below)

MAX_STEPS: int = 8
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 150
SUCCESS_SCORE_THRESHOLD: float = 0.1  # Normalised score in [0, 1]

# Task registry: name → max achievable reward
TASKS = [
    {"name": "email_urgent", "max_reward": 5.0},   # 5 emails × 1.0 = 5.0 max
    {"name": "email_route",  "max_reward": 5.0},   # 5 emails × 1.0 = 5.0 max
    {"name": "email_triage", "max_reward": 3.0},   # 3 emails × 1.0 = 3.0 max
]

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert email triage assistant for a software company.

    You will receive one email at a time. Read it carefully and respond as instructed.

    Task types you may encounter:
      • URGENT CLASSIFICATION: Respond with exactly one word — URGENT or NOT_URGENT.
      • DEPARTMENT ROUTING: Respond with exactly one of — BILLING, TECH_SUPPORT, SALES, GENERAL.
      • FULL TRIAGE: Include priority (P1/P2/P3), department, and a concise response to the sender.

    Always follow the specific instructions given with each email.
    Be decisive and concise. Your accuracy directly determines your reward.
    """
).strip()


# ─── Logging Helpers (strict format — do not modify) ─────────────────────────


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Collapse newlines in action so the entire log line stays on one line
    action_safe = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── Prompt Builder ───────────────────────────────────────────────────────────


def build_user_prompt(
    step: int, last_echoed: str, last_reward: float, history: List[str]
) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current email / instruction:
        {last_echoed}

        Last reward received: {last_reward:.2f}

        Recent history (last 4 steps):
        {history_block}

        Provide your response for the email above.
        """
    ).strip()


# ─── Model Call ───────────────────────────────────────────────────────────────


def get_model_message(
    client: OpenAI,
    step: int,
    last_echoed: str,
    last_reward: float,
    history: List[str],
) -> str:
    """Call the LLM and return its response. Falls back to a safe default."""
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "URGENT"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "URGENT"


# ─── Single Task Runner ───────────────────────────────────────────────────────


async def run_task(client: OpenAI, task_name: str, max_reward: float) -> dict:
    """
    Run a full episode for the given task.
    Emits [START] … [STEP] × N … [END] to stdout.
    Returns a summary dict with task, score, success, and per-step rewards.
    """
    # Propagate task name to env via environment variable
    os.environ["MY_ENV_V4_TASK"] = task_name

    env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken: int = 0
    score: float = 0.0
    success: bool = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Reset ──────────────────────────────────────────────────────────
        result = await env.reset()
        last_echoed: str = result.observation.echoed_message
        last_reward: float = 0.0

        # ── Episode loop ───────────────────────────────────────────────────
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(
                client, step, last_echoed, last_reward, history
            )

            result = await env.step(MyEnvV4Action(message=message))
            obs     = result.observation
            reward  = result.reward if result.reward is not None else 0.0
            done    = result.done
            error: Optional[str] = None

            rewards.append(reward)
            steps_taken = step
            last_echoed = obs.echoed_message
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {message!r} -> reward {reward:+.2f}")

            if done:
                break

        # ── Normalise score to [0, 1] ──────────────────────────────────────
        score = sum(rewards) / max_reward if max_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_results = []

    for task_cfg in TASKS:
        result = await run_task(
            client,
            task_name=task_cfg["name"],
            max_reward=task_cfg["max_reward"],
        )
        all_results.append(result)

    # Final summary (informational — not part of the strict log format)
    print("\n[SUMMARY] Baseline scores across all tasks:", flush=True)
    for r in all_results:
        status = "SUCCESS" if r["success"] else "FAIL"
        rewards_str = ",".join(f"{x:.2f}" for x in r["rewards"])
        print(
            f"  [{status}] task={r['task']} score={r['score']:.3f} "
            f"steps={r['steps']} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
