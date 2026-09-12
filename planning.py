"""Pure data contract and state transitions for V3 task plans.

This module deliberately has no LLM, Tool, filesystem, or Runtime dependency.
The Runtime will be the only caller allowed to persist the returned plan values.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
import re


PLAN_FORMAT_VERSION = 1
MAX_PLAN_STEPS = 8
MAX_PLAN_GOAL_CHARS = 2_000
MAX_STEP_DESCRIPTION_CHARS = 500
MAX_TASK_LLM_STEPS = 32
MAX_TASK_TOOL_CALLS = 24
MAX_TASK_REPLANS = 2
MAX_EVIDENCE_REFS_PER_STEP = 24

TASK_STATUSES = frozenset(
    {"running", "blocked", "completed", "failed", "cancelled"}
)
STEP_STATUSES = frozenset(
    {"pending", "running", "completed", "blocked", "failed"}
)

_PLAN_KEYS = frozenset(
    {
        "version",
        "task_id",
        "goal",
        "status",
        "revision",
        "current_step_id",
        "steps",
        "budget",
    }
)
_STEP_KEYS = frozenset(
    {"id", "description", "status", "attempts", "evidence_refs"}
)
_BUDGET_LIMITS = {
    "llm_steps": MAX_TASK_LLM_STEPS,
    "tool_calls": MAX_TASK_TOOL_CALLS,
    "replans": MAX_TASK_REPLANS,
}
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EVIDENCE_REF_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
)


class PlanValidationError(ValueError):
    """Raised when a plan or requested plan transition is invalid."""


def create_plan(
    task_id: str,
    goal: str,
    step_descriptions: list[str],
) -> dict:
    """Create a validated revision-1 plan from trusted Runtime inputs."""
    normalized_task_id = _normalize_nonempty_string(task_id, "task_id")
    normalized_goal = _normalize_nonempty_string(goal, "goal")
    normalized_descriptions = normalize_step_descriptions(step_descriptions)

    steps = []
    for index, description in enumerate(normalized_descriptions, start=1):
        steps.append(
            {
                "id": f"step-{index:03d}",
                "description": description,
                "status": "pending",
                "attempts": 0,
                "evidence_refs": [],
            }
        )

    plan = {
        "version": PLAN_FORMAT_VERSION,
        "task_id": normalized_task_id,
        "goal": normalized_goal,
        "status": "running",
        "revision": 1,
        "current_step_id": None,
        "steps": steps,
        "budget": {
            "llm_steps": 0,
            "tool_calls": 0,
            "replans": 0,
        },
    }
    validate_plan(plan)
    return plan


def normalize_step_descriptions(value: object) -> list[str]:
    """Validate and normalize the only Plan content supplied by the LLM."""
    if not isinstance(value, list):
        raise PlanValidationError("step_descriptions must be a list.")
    if not 1 <= len(value) <= MAX_PLAN_STEPS:
        raise PlanValidationError(
            f"a plan must contain between 1 and {MAX_PLAN_STEPS} steps."
        )

    normalized = []
    for index, description in enumerate(value, start=1):
        normalized_description = _normalize_nonempty_string(
            description,
            f"step {index} description",
        )
        if len(normalized_description) > MAX_STEP_DESCRIPTION_CHARS:
            raise PlanValidationError(
                f"step {index} description must contain at most "
                f"{MAX_STEP_DESCRIPTION_CHARS} characters."
            )
        normalized.append(normalized_description)
    return normalized


def validate_plan(plan: object) -> None:
    """Validate the complete, versioned V3 plan data contract."""
    if not isinstance(plan, dict):
        raise PlanValidationError("plan must be a dictionary.")
    _require_exact_keys(plan, _PLAN_KEYS, "plan")

    if not _is_int(plan["version"]) or plan["version"] != PLAN_FORMAT_VERSION:
        raise PlanValidationError(
            f"plan version must be {PLAN_FORMAT_VERSION}."
        )

    _validate_identifier(plan["task_id"], "task_id")
    _validate_bounded_string(
        plan["goal"],
        "goal",
        MAX_PLAN_GOAL_CHARS,
    )

    status = plan["status"]
    if not isinstance(status, str) or status not in TASK_STATUSES:
        raise PlanValidationError(f"invalid task status: {status!r}.")

    revision = plan["revision"]
    if not _is_int(revision) or revision < 1:
        raise PlanValidationError("revision must be a positive integer.")

    steps = plan["steps"]
    if not isinstance(steps, list):
        raise PlanValidationError("steps must be a list.")
    if not 1 <= len(steps) <= MAX_PLAN_STEPS:
        raise PlanValidationError(
            f"a plan must contain between 1 and {MAX_PLAN_STEPS} steps."
        )

    step_ids: set[str] = set()
    active_steps = []
    for index, step in enumerate(steps, start=1):
        _validate_step(step, index)
        step_id = step["id"]
        if step_id in step_ids:
            raise PlanValidationError(f"duplicate step id: {step_id!r}.")
        step_ids.add(step_id)
        if step["status"] in {"running", "blocked"}:
            active_steps.append(step)

    if len(active_steps) > 1:
        raise PlanValidationError(
            "a plan can contain at most one running or blocked step."
        )

    current_step_id = plan["current_step_id"]
    if current_step_id is not None:
        _validate_identifier(current_step_id, "current_step_id")
        if current_step_id not in step_ids:
            raise PlanValidationError(
                "current_step_id must reference an existing step."
            )

    if active_steps:
        if current_step_id != active_steps[0]["id"]:
            raise PlanValidationError(
                "current_step_id must reference the active step."
            )
    elif current_step_id is not None:
        raise PlanValidationError(
            "current_step_id must be null when no step is active."
        )

    if status == "running":
        if active_steps and active_steps[0]["status"] != "running":
            raise PlanValidationError(
                "a running task cannot contain a blocked active step."
            )
        if all(step["status"] == "completed" for step in steps):
            raise PlanValidationError(
                "a task with all steps completed must be completed."
            )
    elif status == "blocked":
        if len(active_steps) != 1 or active_steps[0]["status"] != "blocked":
            raise PlanValidationError(
                "a blocked task must have exactly one blocked current step."
            )
    elif status == "completed":
        if current_step_id is not None or not all(
            step["status"] == "completed" for step in steps
        ):
            raise PlanValidationError(
                "a completed task must have every step completed and no current step."
            )
    elif active_steps:
        raise PlanValidationError(
            "a failed or cancelled task cannot have an active step."
        )

    _validate_budget(plan["budget"])


def record_plan_usage(
    plan: dict,
    *,
    llm_steps: int = 0,
    tool_calls: int = 0,
    replans: int = 0,
) -> dict:
    """Return a new plan with trusted Runtime usage added to its budget."""
    validate_plan(plan)
    increments = {
        "llm_steps": llm_steps,
        "tool_calls": tool_calls,
        "replans": replans,
    }
    for name, value in increments.items():
        if not _is_int(value) or value < 0:
            raise PlanValidationError(
                f"usage increment {name} must be a non-negative integer."
            )

    updated = deepcopy(plan)
    for name, value in increments.items():
        updated["budget"][name] += value
    validate_plan(updated)
    return updated


def record_current_step_evidence(
    plan: dict,
    evidence_refs: Sequence[str],
) -> dict:
    """Attach trusted Runtime Tool Result references to the running step."""
    updated, step = _copy_running_plan_and_current_step(plan)
    _append_evidence_refs(step, evidence_refs)
    validate_plan(updated)
    return updated


def start_next_step(plan: dict) -> dict:
    """Start the first pending step and return a new plan value."""
    validate_plan(plan)
    if plan["status"] != "running":
        raise PlanValidationError("only a running task can start a step.")
    if plan["current_step_id"] is not None:
        raise PlanValidationError("the plan already has an active step.")
    if any(step["status"] == "failed" for step in plan["steps"]):
        raise PlanValidationError(
            "a failed step must be resolved by replanning before execution continues."
        )

    updated = deepcopy(plan)
    next_step = next(
        (step for step in updated["steps"] if step["status"] == "pending"),
        None,
    )
    if next_step is None:
        raise PlanValidationError("the plan has no pending step to start.")

    next_step["status"] = "running"
    next_step["attempts"] += 1
    updated["current_step_id"] = next_step["id"]
    validate_plan(updated)
    return updated


def complete_current_step(
    plan: dict,
    evidence_refs: Sequence[str] | None = None,
) -> dict:
    """Complete the running step and complete the task when it was the last one."""
    updated, step = _copy_running_plan_and_current_step(plan)
    _append_evidence_refs(step, evidence_refs)
    step["status"] = "completed"
    updated["current_step_id"] = None
    if all(item["status"] == "completed" for item in updated["steps"]):
        updated["status"] = "completed"
    validate_plan(updated)
    return updated


def block_current_step(
    plan: dict,
    evidence_refs: Sequence[str] | None = None,
) -> dict:
    """Block the running step while waiting for required user information."""
    updated, step = _copy_running_plan_and_current_step(plan)
    _append_evidence_refs(step, evidence_refs)
    step["status"] = "blocked"
    updated["status"] = "blocked"
    validate_plan(updated)
    return updated


def resume_blocked_step(plan: dict) -> dict:
    """Resume the blocked current step after the user supplies information."""
    validate_plan(plan)
    if plan["status"] != "blocked":
        raise PlanValidationError("only a blocked task can resume its step.")

    updated = deepcopy(plan)
    step = _find_current_step(updated)
    if step["status"] != "blocked":
        raise PlanValidationError("the current step is not blocked.")
    step["status"] = "running"
    step["attempts"] += 1
    updated["status"] = "running"
    validate_plan(updated)
    return updated


def fail_current_step(
    plan: dict,
    evidence_refs: Sequence[str] | None = None,
) -> dict:
    """Mark the running step failed while leaving the task available to replan."""
    updated, step = _copy_running_plan_and_current_step(plan)
    _append_evidence_refs(step, evidence_refs)
    step["status"] = "failed"
    updated["current_step_id"] = None
    validate_plan(updated)
    return updated


def fail_plan(plan: dict) -> dict:
    """Move a running or blocked task to its terminal failed state."""
    return _finish_plan(plan, "failed")


def cancel_plan(plan: dict) -> dict:
    """Move a running or blocked task to its terminal cancelled state."""
    return _finish_plan(plan, "cancelled")


def _validate_step(step: object, index: int) -> None:
    if not isinstance(step, dict):
        raise PlanValidationError(f"step {index} must be a dictionary.")
    _require_exact_keys(step, _STEP_KEYS, f"step {index}")
    _validate_identifier(step["id"], f"step {index} id")
    _validate_bounded_string(
        step["description"],
        f"step {index} description",
        MAX_STEP_DESCRIPTION_CHARS,
    )

    status = step["status"]
    if not isinstance(status, str) or status not in STEP_STATUSES:
        raise PlanValidationError(
            f"invalid status for step {step['id']!r}: {status!r}."
        )

    attempts = step["attempts"]
    if not _is_int(attempts) or attempts < 0:
        raise PlanValidationError("step attempts must be a non-negative integer.")
    if status == "pending" and attempts != 0:
        raise PlanValidationError("a pending step must have zero attempts.")
    if status != "pending" and attempts < 1:
        raise PlanValidationError(
            "a non-pending step must have at least one attempt."
        )

    evidence_refs = step["evidence_refs"]
    _validate_evidence_refs(evidence_refs)
    if status == "pending" and evidence_refs:
        raise PlanValidationError(
            "a pending step cannot contain evidence references."
        )


def _validate_budget(budget: object) -> None:
    if not isinstance(budget, dict):
        raise PlanValidationError("budget must be a dictionary.")
    _require_exact_keys(budget, frozenset(_BUDGET_LIMITS), "budget")
    for name, limit in _BUDGET_LIMITS.items():
        value = budget[name]
        if not _is_int(value) or not 0 <= value <= limit:
            raise PlanValidationError(
                f"budget {name} must be an integer between 0 and {limit}."
            )


def _validate_evidence_refs(evidence_refs: object) -> None:
    if not isinstance(evidence_refs, list):
        raise PlanValidationError("evidence_refs must be a list.")
    if len(evidence_refs) > MAX_EVIDENCE_REFS_PER_STEP:
        raise PlanValidationError(
            "a step contains too many evidence references."
        )
    for reference in evidence_refs:
        if not isinstance(reference, str) or not _EVIDENCE_REF_PATTERN.fullmatch(
            reference
        ):
            raise PlanValidationError(
                f"invalid evidence reference: {reference!r}."
            )
    if len(set(evidence_refs)) != len(evidence_refs):
        raise PlanValidationError("evidence references must be unique.")


def _copy_running_plan_and_current_step(plan: dict) -> tuple[dict, dict]:
    validate_plan(plan)
    if plan["status"] != "running":
        raise PlanValidationError(
            "the current step can change only while the task is running."
        )
    updated = deepcopy(plan)
    step = _find_current_step(updated)
    if step["status"] != "running":
        raise PlanValidationError("the current step is not running.")
    return updated, step


def _find_current_step(plan: dict) -> dict:
    current_step_id = plan["current_step_id"]
    if current_step_id is None:
        raise PlanValidationError("the plan has no current step.")
    return next(
        step for step in plan["steps"] if step["id"] == current_step_id
    )


def _append_evidence_refs(
    step: dict,
    evidence_refs: Sequence[str] | None,
) -> None:
    if evidence_refs is None:
        return
    if isinstance(evidence_refs, (str, bytes)) or not isinstance(
        evidence_refs, Sequence
    ):
        raise PlanValidationError(
            "new evidence references must be a sequence of strings."
        )
    combined = [*step["evidence_refs"], *evidence_refs]
    _validate_evidence_refs(combined)
    step["evidence_refs"] = combined


def _finish_plan(plan: dict, terminal_status: str) -> dict:
    validate_plan(plan)
    if plan["status"] not in {"running", "blocked"}:
        raise PlanValidationError(
            f"only a running or blocked task can become {terminal_status}."
        )

    updated = deepcopy(plan)
    if updated["current_step_id"] is not None:
        step = _find_current_step(updated)
        step["status"] = "failed"
    updated["current_step_id"] = None
    updated["status"] = terminal_status
    validate_plan(updated)
    return updated


def _require_exact_keys(
    value: dict,
    expected_keys: frozenset[str],
    label: str,
) -> None:
    actual_keys = set(value)
    if actual_keys == expected_keys:
        return
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    raise PlanValidationError(
        f"{label} fields do not match the contract; "
        f"missing={missing}, extra={extra}."
    )


def _normalize_nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError(f"{label} must be a non-empty string.")
    return value.strip()


def _validate_bounded_string(value: object, label: str, limit: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PlanValidationError(f"{label} must be a non-empty string.")
    if value != value.strip():
        raise PlanValidationError(f"{label} must not have surrounding whitespace.")
    if len(value) > limit:
        raise PlanValidationError(
            f"{label} must contain at most {limit} characters."
        )


def _validate_identifier(value: object, label: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise PlanValidationError(f"{label} is not a valid identifier.")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
