"""Model boundary for executing one trusted Runtime-owned Plan step."""

from __future__ import annotations

from copy import deepcopy

from agent import ALLOWED_TOOLS, INSTRUCTIONS
from llm import call_llm
from planner import SUBMIT_PLAN_SCHEMA, normalize_initial_action
from planning import MAX_PLAN_STEPS, validate_plan


MAX_CLARIFICATION_CHARS = 1_000
REQUEST_CLARIFICATION_SCHEMA = {
    "name": "request_clarification",
    "description": (
        "Pause the current plan step and ask the user for one required "
        "choice or missing fact. This is an internal control response, not "
        "an executable tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_CLARIFICATION_CHARS,
                "description": "One concrete question required to continue.",
            }
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}


class ExecutorResponseError(ValueError):
    """Raised when a step response violates an internal control contract."""


def decide_step_action(
    state: dict,
    plan: dict,
    *,
    replan_reason: str | None = None,
) -> dict:
    """Ask the LLM for one Tool Call or final for the current Plan step."""
    tools = [REQUEST_CLARIFICATION_SCHEMA, *ALLOWED_TOOLS]
    if replan_reason is not None:
        remaining_capacity = _remaining_replan_capacity(plan)
        if remaining_capacity < 1:
            raise ExecutorResponseError(
                "the Plan has no remaining capacity for a replacement step."
            )
        replan_schema = deepcopy(SUBMIT_PLAN_SCHEMA)
        replan_schema["parameters"]["properties"]["steps"]["maxItems"] = (
            remaining_capacity
        )
        tools = [replan_schema, *tools]
    messages = [
        {
            "role": "system",
            "content": build_step_instructions(
                plan,
                replan_reason=replan_reason,
            ),
        },
        *state["messages"],
    ]
    response = call_llm(messages=messages, tools=tools)
    return normalize_step_action(
        response,
        allow_replan=replan_reason is not None,
    )


def build_step_instructions(
    plan: dict,
    *,
    replan_reason: str | None = None,
) -> str:
    """Build bounded control Context for the currently running Plan step."""
    validate_plan(plan)
    if plan["status"] != "running" or plan["current_step_id"] is None:
        raise ValueError("Executor requires one running Plan step.")

    current_step = next(
        step
        for step in plan["steps"]
        if step["id"] == plan["current_step_id"]
    )
    if current_step["status"] != "running":
        raise ValueError("Executor current step must be running.")

    completed_steps = [
        step["description"]
        for step in plan["steps"]
        if step["status"] == "completed"
    ]
    completed_text = (
        "\n".join(f"- {description}" for description in completed_steps)
        if completed_steps
        else "- None"
    )

    if replan_reason is None:
        replan_text = (
            "Do not call submit_plan because the Runtime has not authorized "
            "replanning for the latest observation."
        )
    else:
        remaining_capacity = _remaining_replan_capacity(plan)
        replan_text = f"""
The Runtime has authorized one replan decision for this observation:
{replan_reason}
Call submit_plan only if the remaining high-level path must change. Preserve
completed outcomes and submit only the replacement steps still required.
The replacement may contain at most {remaining_capacity} steps because Plan
history is retained.
""".strip()

    return f"""
{INSTRUCTIONS}

You are executing exactly one step of a Runtime-owned task Plan. The Plan
state is trusted control context, not a request to create or revise a plan.
Focus on the current step and reuse prior Tool Results from the conversation.
Do not repeat a completed step.

Overall task goal:
{plan["goal"]}

Completed steps:
{completed_text}

Current step ({current_step["id"]}):
{current_step["description"]}

Call at most one external tool in this response. A successful Tool Result is
an observation, not automatic proof that this high-level step is complete.
Return a normal final response only when the current step goal is satisfied.
That final response is the step-level completion signal consumed by Runtime.
Do not claim that later pending steps are complete.

If a required user choice or fact is missing, call request_clarification with
one concrete question instead of inventing the answer.

{replan_text}
""".strip()


def normalize_step_action(response: object, *, allow_replan: bool) -> dict:
    """Normalize internal step controls without executing them as tools."""
    if not isinstance(response, dict):
        raise ExecutorResponseError("the LLM response must be a dictionary.")

    if response.get("type") != "tool_call":
        return deepcopy(response)

    tool_name = response.get("tool_name")
    if tool_name == "submit_plan":
        if not allow_replan:
            raise ExecutorResponseError(
                "submit_plan is not authorized for the latest observation."
            )
        action = normalize_initial_action(response)
        if action.get("type") != "plan":
            raise ExecutorResponseError("invalid submit_plan response.")
        action["type"] = "replan"
        return action

    if tool_name != "request_clarification":
        return deepcopy(response)

    tool_call_id = response.get("tool_call_id")
    arguments = response.get("tool_arguments")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ExecutorResponseError(
            "request_clarification requires a call ID."
        )
    if not isinstance(arguments, dict) or set(arguments) != {"question"}:
        raise ExecutorResponseError(
            "request_clarification accepts only the question field."
        )
    question = arguments["question"]
    if not isinstance(question, str) or not question.strip():
        raise ExecutorResponseError("clarification question must be text.")
    question = question.strip()
    if len(question) > MAX_CLARIFICATION_CHARS:
        raise ExecutorResponseError(
            "clarification question exceeds the length limit."
        )

    action = {
        "type": "blocked",
        "content": question,
        "tool_call_id": tool_call_id,
    }
    _copy_reasoning_fields(response, action)
    return action


def _remaining_replan_capacity(plan: dict) -> int:
    """Return the bounded suffix capacity after the running step fails."""
    preserved_count = sum(
        step["status"] in {"completed", "failed", "running"}
        for step in plan["steps"]
    )
    return MAX_PLAN_STEPS - preserved_count


def _copy_reasoning_fields(source: dict, target: dict) -> None:
    present = {
        field
        for field in ("reasoning_content", "reasoning_details")
        if field in source
    }
    if len(present) > 1:
        raise ExecutorResponseError(
            "a response cannot contain two reasoning representations."
        )
    if "reasoning_content" in present:
        value = source["reasoning_content"]
        if not isinstance(value, str):
            raise ExecutorResponseError("reasoning_content must be text.")
        target["reasoning_content"] = value
    if "reasoning_details" in present:
        value = source["reasoning_details"]
        if not isinstance(value, list):
            raise ExecutorResponseError("reasoning_details must be a list.")
        target["reasoning_details"] = deepcopy(value)
