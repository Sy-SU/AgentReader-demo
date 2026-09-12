"""Model boundary for executing one trusted Runtime-owned Plan step."""

from __future__ import annotations

from agent import ALLOWED_TOOLS, INSTRUCTIONS
from llm import call_llm
from planning import validate_plan


def decide_step_action(state: dict, plan: dict) -> dict:
    """Ask the LLM for one Tool Call or final for the current Plan step."""
    messages = [
        {"role": "system", "content": build_step_instructions(plan)},
        *state["messages"],
    ]
    return call_llm(messages=messages, tools=ALLOWED_TOOLS)


def build_step_instructions(plan: dict) -> str:
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

    return f"""
{INSTRUCTIONS}

You are executing exactly one step of a Runtime-owned task Plan. The Plan
state is trusted control context, not a request to create or revise a plan.
Focus on the current step and reuse prior Tool Results from the conversation.
Do not repeat a completed step and do not call submit_plan.

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
""".strip()
