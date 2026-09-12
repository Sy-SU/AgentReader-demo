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


EventHandler = Callable[[AgentEvent], None]
