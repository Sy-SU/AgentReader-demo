"""Structured, UI-neutral events emitted by the Agent Runtime."""

from collections.abc import Callable
from typing import Literal, TypedDict


EventKind = Literal[
    "llm_started",
    "llm_finished",
    "tool_started",
    "tool_finished",
    "turn_finished",
    "run_failed",
    "plan_created",
    "plan_step_changed",
    "plan_replanned",
    "task_blocked",
    "task_cancelled",
    "checkpoint_saved",
]


class AgentEvent(TypedDict, total=False):
    """One observable Runtime lifecycle event."""

    kind: EventKind
    turn_step: int
    total_step: int
    duration_ms: float
    response_type: str
    tool_name: str
    arguments: dict
    result: dict
    status: str
    error_type: str
    message: str
    task_id: str
    plan_revision: int
    step_id: str
    step_description: str
    previous_status: str
    step_count: int
    replan_count: int
    reason: str
    checkpoint_size_bytes: int


EventHandler = Callable[[AgentEvent], None]
