"""Initial V3 planning decision without executing a Plan or a Tool."""

from __future__ import annotations

from copy import deepcopy

from agent import ALLOWED_TOOLS, INSTRUCTIONS
from llm import call_llm
from planning import (
    MAX_PLAN_STEPS,
    MAX_STEP_DESCRIPTION_CHARS,
    PlanValidationError,
    normalize_step_descriptions,
)


SUBMIT_PLAN_SCHEMA = {
    "name": "submit_plan",
    "description": (
        "Submit an ordered high-level plan for a genuinely multi-step task. "
        "This is an internal planning response, not an executable tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_PLAN_STEPS,
                "items": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_STEP_DESCRIPTION_CHARS,
                },
                "description": (
                    "Ordered outcome-oriented step descriptions. Do not "
                    "include state, counters, IDs, URLs, paths, or evidence."
                ),
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    },
}


PLANNING_INSTRUCTIONS = f"""
{INSTRUCTIONS}

For a genuinely composite request with multiple papers, multiple evidence
requirements, or multiple deliverables, you may call submit_plan as the first
decision. Keep the plan outcome-oriented and ordered. Do not encode a fixed
sequence of tool names. Submit between 1 and {MAX_PLAN_STEPS} concise steps.

The submit_plan call may contain only step descriptions. Never provide task
IDs, step IDs, statuses, attempts, budgets, evidence references, URLs, or local
paths. Planning does not grant permission for saving, downloading, or another
side effect that the user did not request.

For a simple search, save, download, or single-paper question, keep using the
existing tool_call or final response directly. Do not create a plan merely
because submit_plan is available.
""".strip()


INITIAL_ACTION_TOOLS = [SUBMIT_PLAN_SCHEMA, *ALLOWED_TOOLS]


class PlannerResponseError(ValueError):
    """Raised when the initial model decision violates the planner contract."""


def decide_initial_action(state: dict) -> dict:
    """Ask once for a plan, ordinary Tool Call, or final answer."""
    messages = [
        {"role": "system", "content": PLANNING_INSTRUCTIONS},
        *state["messages"],
    ]
    response = call_llm(messages=messages, tools=INITIAL_ACTION_TOOLS)
    return normalize_initial_action(response)


def normalize_initial_action(response: object) -> dict:
    """Convert an internal submit_plan Tool Call into a trusted action shape."""
    if not isinstance(response, dict):
        raise PlannerResponseError("the LLM response must be a dictionary.")

    response_type = response.get("type")
    if response_type == "final":
        if not isinstance(response.get("content"), str):
            raise PlannerResponseError("a final response must contain text.")
        return deepcopy(response)

    if response_type != "tool_call":
        raise PlannerResponseError(
            "the initial decision must be plan, tool_call, or final."
        )

    tool_name = response.get("tool_name")
    tool_call_id = response.get("tool_call_id")
    tool_arguments = response.get("tool_arguments")
    if not isinstance(tool_name, str) or not tool_name:
        raise PlannerResponseError("a tool call must contain a tool name.")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise PlannerResponseError("a tool call must contain a call ID.")
    if not isinstance(tool_arguments, dict):
        raise PlannerResponseError("tool arguments must be a dictionary.")

    if tool_name != "submit_plan":
        return deepcopy(response)

    if set(tool_arguments) != {"steps"}:
        raise PlannerResponseError(
            "submit_plan accepts only the steps field."
        )
    try:
        step_descriptions = normalize_step_descriptions(
            tool_arguments["steps"]
        )
    except PlanValidationError as error:
        raise PlannerResponseError(
            f"invalid submit_plan steps: {error}"
        ) from error

    action = {
        "type": "plan",
        "content": None,
        "tool_call_id": tool_call_id,
        "step_descriptions": step_descriptions,
    }
    _copy_reasoning_fields(response, action)
    return action


def _copy_reasoning_fields(source: dict, target: dict) -> None:
    """Preserve normalized opaque reasoning when submit_plan is internalized."""
    present = {
        field
        for field in ("reasoning_content", "reasoning_details")
        if field in source
    }
    if len(present) > 1:
        raise PlannerResponseError(
            "a response cannot contain two reasoning representations."
        )
    if "reasoning_content" in present:
        value = source["reasoning_content"]
        if not isinstance(value, str):
            raise PlannerResponseError("reasoning_content must be text.")
        target["reasoning_content"] = value
    if "reasoning_details" in present:
        value = source["reasoning_details"]
        if not isinstance(value, list):
            raise PlannerResponseError("reasoning_details must be a list.")
        target["reasoning_details"] = deepcopy(value)
