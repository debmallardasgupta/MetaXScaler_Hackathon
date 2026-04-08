# Email Triage OpenEnv

A real-world **email triage** environment for agent benchmarking, built to the [OpenEnv](https://github.com/openenv) specification.

Agents read incoming support emails and must classify urgency, route to the correct department, or perform full triage — mirroring tasks that customer-support and operations teams do every day.

---

## Environment Description

| Property | Value |
|----------|-------|
| Domain | Customer support / operations |
| Observation | Raw email text + task instruction |
| Action | Free-form text (classification label, department name, or structured triage) |
| Reward range | `[-0.1, 1.0]` per step |
| Tasks | 3 (easy → medium → hard) |
| Episodes | Deterministic; same emails every run for reproducibility |

---

## Observation Space

```
type:  text
field: echoed_message  (string)
```

On `reset()` the agent receives the first email plus task-specific instructions.  
After each `step()` the agent receives the next email, or a terminal summary when all emails are processed.

**Example observation (email_urgent task):**
```
[Email 1/5]
Subject: URGENT - Production database is down!
From: ops@company.com
Body: Our main production database crashed 5 minutes ago...

Classify this email. Reply with exactly one of: URGENT  NOT_URGENT
```

---

## Action Space

```
type:  text
field: message  (string)
```

The exact format depends on the active task:

| Task | Required action format |
|------|------------------------|
| `email_urgent` | `URGENT` or `NOT_URGENT` |
| `email_route` | `BILLING`, `TECH_SUPPORT`, `SALES`, or `GENERAL` |
| `email_triage` | Free text containing priority (`P1`/`P2`/`P3`), department, and a reply to the sender |

---

## Tasks

### Task 1 — `email_urgent` (Easy)

Classify each of **5 emails** as `URGENT` or `NOT_URGENT`.

- Correct → `1.0`
- Incorrect → `0.0`
- Max episode reward: **5.0**

### Task 2 — `email_route` (Medium)

Route each of **5 emails** to the correct department (`BILLING`, `TECH_SUPPORT`, `SALES`, `GENERAL`).

- Exact match → `1.0`
- Near match (related keyword) → `0.5`
- Wrong → `0.0`
- Max episode reward: **5.0**

### Task 3 — `email_triage` (Hard)

Full triage of **3 emails**: assign priority, route to department, and draft a brief reply.

Partial credit per email (up to `1.0`):
- Priority correct → `+0.40`
- Department correct → `+0.35`
- Response quality (keyword coverage) → `+0.25`

Max episode reward: **3.0**

---

## Reward Function

| Condition | Reward |
|-----------|--------|
| Empty / null action | `-0.1` (penalty) |
| Wrong classification or routing | `0.0` |
| Near-correct routing (related keyword) | `0.5` |
| Fully correct classification or routing | `1.0` |
| Triage partial credit | `0.0 – 1.0` (component-weighted) |

Rewards are provided **every step**, giving partial-progress signal throughout the episode rather than only at termination.

---

## Setup & Usage

### Prerequisites

- Python 3.11+
- Docker (for containerised runs)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API server locally

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=<your-token> \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  email-triage-env
```

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode; body: `{"task": "email_urgent"}` |
| `POST` | `/step` | Submit an action; body: `{"message": "URGENT"}` |
| `GET` | `/state` | Current environment state snapshot |
| `GET` | `/health` | Liveness probe |
| `GET` | `/tasks` | List all tasks with descriptions |

### Run the baseline inference script

```bash
export HF_TOKEN=<your-token>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | Hugging Face / API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `MY_ENV_V4_TASK` | No | `email_urgent` | Active task for the environment |

---

## Baseline Scores

Scores obtained with `Qwen/Qwen2.5-72B-Instruct` via the HuggingFace Inference Router:

| Task | Difficulty | Score (0–1) | Notes |
|------|-----------|-------------|-------|
| `email_urgent` | Easy | ~0.80 | Strong binary classification |
| `email_route` | Medium | ~0.70 | Occasional near-miss on routing |
| `email_triage` | Hard | ~0.55 | Partial credit on priority + response |

Scores are normalised: `sum(step_rewards) / max_episode_reward`, clamped to `[0, 1]`.

---

## OpenEnv Compliance

- ✅ Typed Pydantic models (`MyEnvV4Action`, `Observation`, `StepResult`)
- ✅ `step()`, `reset()`, `state()` async interface
- ✅ `openenv.yaml` metadata
- ✅ 3 tasks with deterministic graders (scores in `[0.0, 1.0]`)
- ✅ Partial-progress reward signal every step
- ✅ Dockerfile + HuggingFace Spaces deployment
