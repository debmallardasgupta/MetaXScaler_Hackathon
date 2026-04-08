"""
my_env_v4.py — Email Triage OpenEnv Environment
================================================
Real-world email triage: classification, routing, and full triage.
"""

import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Observation(BaseModel):
    echoed_message: str


class MyEnvV4Action(BaseModel):
    message: str


class StepResult(BaseModel):
    observation: Observation
    reward: Optional[float]
    done: bool
    info: Dict[str, Any] = {}


TASK_EMAILS: Dict[str, List[Dict[str, Any]]] = {
    "email_urgent": [
        {
            "email": (
                "Subject: URGENT - Production database is down!\n"
                "From: ops@company.com\n"
                "Body: Our main production database crashed 5 minutes ago. "
                "All services are unavailable. We need immediate DBA help!"
            ),
            "label": "URGENT",
        },
        {
            "email": (
                "Subject: Monthly newsletter — April edition\n"
                "From: newsletter@marketing.com\n"
                "Body: Hello! Our April newsletter is here. "
                "Check out the latest product updates and team news."
            ),
            "label": "NOT_URGENT",
        },
        {
            "email": (
                "Subject: EMERGENCY: Payment system failure — all transactions rejected\n"
                "From: payments@fintech.com\n"
                "Body: CRITICAL: Payment gateway is down. Every transaction is failing "
                "with error code 503. Revenue impact is growing by the minute. "
                "Escalate immediately!"
            ),
            "label": "URGENT",
        },
        {
            "email": (
                "Subject: Team lunch reminder — Friday at 1pm\n"
                "From: hr@company.com\n"
                "Body: Just a friendly reminder that our team lunch is this Friday at 1pm "
                "in the main conference room. Please RSVP by Thursday."
            ),
            "label": "NOT_URGENT",
        },
        {
            "email": (
                "Subject: Security breach detected — customer data at risk\n"
                "From: security@company.com\n"
                "Body: We have detected unauthorised access to our customer database. "
                "PII may be exposed. This requires ASAP escalation to CISO and legal."
            ),
            "label": "URGENT",
        },
    ],
    "email_route": [
        {
            "email": (
                "Subject: Invoice #4521 — incorrect amount charged\n"
                "From: client@acme.com\n"
                "Body: I received invoice #4521 and the total is $500 higher than agreed. "
                "Please review and issue a corrected invoice."
            ),
            "department": "BILLING",
            "partial_keywords": ["FINANCE", "PAYMENT", "INVOICE", "ACCOUNT"],
        },
        {
            "email": (
                "Subject: Cannot log in — error 500 on login page\n"
                "From: user@startup.com\n"
                "Body: For the past hour I have been unable to log into my account. "
                "The page shows error 500. I have tried Chrome, Firefox, and Safari."
            ),
            "department": "TECH_SUPPORT",
            "partial_keywords": ["TECHNICAL", "SUPPORT", "ENGINEERING", "IT", "HELP"],
        },
        {
            "email": (
                "Subject: Interested in enterprise plan — 500-seat organisation\n"
                "From: cto@bigcorp.com\n"
                "Body: We are evaluating your platform for our 500-person engineering team. "
                "Can we schedule a demo and discuss enterprise pricing?"
            ),
            "department": "SALES",
            "partial_keywords": ["BUSINESS", "ACCOUNT", "REVENUE"],
        },
        {
            "email": (
                "Subject: How do I bulk-export my project data?\n"
                "From: power_user@startup.com\n"
                "Body: I need to export all 3 years of project data to migrate to a new tool. "
                "Is there a bulk export option, or does your API support this?"
            ),
            "department": "TECH_SUPPORT",
            "partial_keywords": ["TECHNICAL", "SUPPORT", "ENGINEERING", "IT", "HELP"],
        },
        {
            "email": (
                "Subject: Agency partnership proposal\n"
                "From: bd@agency.com\n"
                "Body: Our digital agency works with 50+ SMBs. We would love to become a "
                "reseller partner and offer your product to our client base."
            ),
            "department": "SALES",
            "partial_keywords": ["BUSINESS", "ACCOUNT", "REVENUE"],
        },
    ],
    "email_triage": [
        {
            "email": (
                "Subject: CRITICAL: Production database corruption detected\n"
                "From: dba@company.com\n"
                "Body: Multiple production tables are returning corrupt data. "
                "SELECT queries are failing with integrity errors. "
                "All live users are affected. Immediate DBA response required."
            ),
            "priority": "P1",
            "department": "TECH_SUPPORT",
            "response_keywords": ["acknowledge", "urgent", "immediate", "escalat", "critical", "p1"],
        },
        {
            "email": (
                "Subject: Question about line items on this month's billing statement\n"
                "From: finance@client.com\n"
                "Body: Could you help clarify the 'Platform Usage' line item on our "
                "May statement? The amount seems higher than usual and we need "
                "to reconcile it for our books."
            ),
            "priority": "P3",
            "department": "BILLING",
            "response_keywords": ["clarif", "billing", "statement", "invoice", "explain", "review"],
        },
        {
            "email": (
                "Subject: Feature request: dark mode for the dashboard\n"
                "From: power_user@startup.com\n"
                "Body: Love the product! One improvement that would make a huge difference: "
                "a dark mode option for the main dashboard. "
                "Many of us work late nights and eye strain is real."
            ),
            "priority": "P3",
            "department": "GENERAL",
            "response_keywords": ["thank", "noted", "roadmap", "feature", "request", "consider"],
        },
    ],
}


def _grade_urgent(response: str, label: str) -> float:
    r = response.upper().strip()
    if label == "URGENT":
        if "NOT_URGENT" in r or "NOT URGENT" in r:
            return 0.0
        if "URGENT" in r:
            return 1.0
        return 0.0
    if label == "NOT_URGENT":
        if "NOT_URGENT" in r or "NOT URGENT" in r:
            return 1.0
        return 0.0
    return 0.0


def _grade_route(response: str, department: str, partial_keywords: List[str]) -> float:
    r = response.upper().strip()
    if department in r:
        return 1.0
    for kw in partial_keywords:
        if kw in r:
            return 0.5
    return 0.0


def _grade_triage(response: str, priority: str, department: str, response_keywords: List[str]) -> float:
    score = 0.0
    r_upper = response.upper()
    r_lower = response.lower()
    if priority in r_upper:
        score += 0.40
    if department in r_upper:
        score += 0.35
    threshold = max(len(response_keywords) * 0.5, 1)
    hits = sum(1 for kw in response_keywords if kw in r_lower)
    score += 0.25 * min(hits / threshold, 1.0)
    return round(min(score, 1.0), 4)


class _EnvState:
    def __init__(self, task_name: str) -> None:
        self.task_name: str = task_name
        self.step_count: int = 0
        self.current_email_idx: int = 0
        self.history: List[str] = []
        self.last_message: str = ""
        self.cumulative_reward: float = 0.0
        self.done: bool = False
        self.emails: List[Dict[str, Any]] = TASK_EMAILS.get(task_name, TASK_EMAILS["email_urgent"])

    def get_current_prompt(self) -> str:
        idx = self.current_email_idx
        if idx >= len(self.emails):
            return (
                f"Task '{self.task_name}' complete. "
                f"All {len(self.emails)} emails processed. "
                f"Cumulative reward: {self.cumulative_reward:.2f}"
            )
        email_content = self.emails[idx]["email"]
        n_total = len(self.emails)
        if self.task_name == "email_urgent":
            return (
                f"[Email {idx + 1}/{n_total}]\n{email_content}\n\n"
                f"Classify this email. Reply with exactly one of: URGENT  NOT_URGENT"
            )
        elif self.task_name == "email_route":
            return (
                f"[Email {idx + 1}/{n_total}]\n{email_content}\n\n"
                f"Route this email. Reply with exactly one of: BILLING  TECH_SUPPORT  SALES  GENERAL"
            )
        elif self.task_name == "email_triage":
            return (
                f"[Email {idx + 1}/{n_total}]\n{email_content}\n\n"
                f"Perform full triage. Include ALL of the following:\n"
                f"  Priority : P1 (critical) / P2 (moderate) / P3 (low)\n"
                f"  Department: BILLING / TECH_SUPPORT / SALES / GENERAL\n"
                f"  Response : a brief reply to the sender"
            )
        return email_content


class MyEnvV4Env:
    """Email Triage Environment (OpenEnv-compatible, async)."""

    MAX_STEPS: int = 8

    def __init__(self, task_name: str = "email_urgent") -> None:
        self._task_name = task_name
        self._state = _EnvState(task_name)
        self._closed: bool = False

    @classmethod
    async def from_docker_image(cls, image_name: Optional[str] = None) -> "MyEnvV4Env":
        task_name = os.getenv("MY_ENV_V4_TASK", "email_urgent")
        if task_name not in TASK_EMAILS:
            task_name = "email_urgent"
        return cls(task_name=task_name)

    async def reset(self) -> StepResult:
        task_name = os.getenv("MY_ENV_V4_TASK", self._task_name)
        if task_name not in TASK_EMAILS:
            task_name = "email_urgent"
        self._task_name = task_name
        self._state = _EnvState(task_name)
        self._closed = False
        prompt = self._state.get_current_prompt()
        return StepResult(
            observation=Observation(echoed_message=prompt),
            reward=None,
            done=False,
            info={"step": 0, "task": task_name, "total_emails": len(self._state.emails)},
        )

    async def step(self, action: MyEnvV4Action) -> StepResult:
        if self._closed:
            return StepResult(
                observation=Observation(echoed_message="Environment is already closed."),
                reward=0.0, done=True, info={"error": "env_closed"},
            )
        if self._state.done:
            return StepResult(
                observation=Observation(echoed_message="Episode already finished. Call reset() to start again."),
                reward=0.0, done=True, info={"error": "episode_done"},
            )
        raw_message = action.message if action.message is not None else ""
        message = raw_message.strip()
        self._state.step_count += 1
        self._state.last_message = message
        self._state.history.append(message)
        reward = self._compute_reward(message)
        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 4)
        self._state.current_email_idx += 1
        all_done = self._state.current_email_idx >= len(self._state.emails)
        max_hit = self._state.step_count >= self.MAX_STEPS
        empty_msg = not message
        done = all_done or max_hit or empty_msg
        self._state.done = done
        if done:
            obs_text = (
                f"Episode complete after {self._state.step_count} step(s). "
                f"Cumulative reward: {self._state.cumulative_reward:.4f}. "
                f"Task: {self._state.task_name}."
            )
        else:
            obs_text = self._state.get_current_prompt()
        return StepResult(
            observation=Observation(echoed_message=obs_text),
            reward=reward,
            done=done,
            info={
                "step": self._state.step_count,
                "email_idx": self._state.current_email_idx,
                "cumulative_reward": self._state.cumulative_reward,
                "task": self._state.task_name,
                "done": done,
            },
        )

    async def state(self) -> Dict[str, Any]:
        return {
            "task": self._state.task_name,
            "step_count": self._state.step_count,
            "current_email_idx": self._state.current_email_idx,
            "total_emails": len(self._state.emails),
            "cumulative_reward": self._state.cumulative_reward,
            "done": self._state.done,
            "closed": self._closed,
            "history": list(self._state.history),
            "last_message": self._state.last_message,
        }

    async def close(self) -> None:
        self._closed = True

    def _compute_reward(self, message: str) -> float:
        if not message:
            return -0.1
        idx = self._state.current_email_idx
        emails = self._state.emails
        if idx >= len(emails):
            return 0.0
        email_data = emails[idx]
        task = self._state.task_name
        try:
            if task == "email_urgent":
                return _grade_urgent(message, email_data["label"])
            elif task == "email_route":
                return _grade_route(message, email_data["department"], email_data.get("partial_keywords", []))
            elif task == "email_triage":
                return _grade_triage(
                    message, email_data["priority"], email_data["department"],
                    email_data.get("response_keywords", [])
                )
        except Exception:
            return 0.0
        return 0.0
