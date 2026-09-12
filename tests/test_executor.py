import unittest
from copy import deepcopy
from unittest.mock import patch

from agent import ALLOWED_TOOLS
from executor import (
    REQUEST_CLARIFICATION_SCHEMA,
    ExecutorResponseError,
    build_step_instructions,
    decide_step_action,
    normalize_step_action,
)
from planner import SUBMIT_PLAN_SCHEMA
from planning import complete_current_step, create_plan, start_next_step
from state import create_state


FINAL = {
    "type": "final",
    "content": "Step complete.",
    "tool_call_id": None,
    "tool_name": None,
    "tool_arguments": None,
}


class ExecutorTests(unittest.TestCase):
    @patch("executor.call_llm", return_value=FINAL)
    def test_executor_injects_only_the_current_trusted_step(self, call_llm):
        state = create_state("Compare paper A and paper B")
        first_running = start_next_step(
            create_plan(
                "task-1",
                "Compare paper A and paper B",
                ["Identify both papers", "Compare their methods"],
            )
        )
        first_done = complete_current_step(first_running)
        second_running = start_next_step(first_done)
        state_snapshot = deepcopy(state)
        plan_snapshot = deepcopy(second_running)

        response = decide_step_action(state, second_running)

        self.assertEqual(response, FINAL)
        self.assertEqual(state, state_snapshot)
        self.assertEqual(second_running, plan_snapshot)
        kwargs = call_llm.call_args.kwargs
        self.assertEqual(
            kwargs["tools"],
            [REQUEST_CLARIFICATION_SCHEMA, *ALLOWED_TOOLS],
        )
        self.assertNotIn("submit_plan", str(kwargs["tools"]))
        instructions = kwargs["messages"][0]["content"]
        self.assertIn("Compare paper A and paper B", instructions)
        self.assertIn("- Identify both papers", instructions)
        self.assertIn("Current step (step-002)", instructions)
        self.assertIn("Compare their methods", instructions)
        self.assertIn("step-level completion signal", instructions)
        self.assertIn("request_clarification", instructions)
        self.assertIn("has not authorized replanning", instructions)
        self.assertEqual(kwargs["messages"][1:], state["messages"])

    @patch(
        "executor.call_llm",
        return_value={
            "type": "tool_call",
            "content": None,
            "tool_call_id": "replan-call-1",
            "tool_name": "submit_plan",
            "tool_arguments": {"steps": ["Use an alternate source"]},
        },
    )
    def test_executor_exposes_and_normalizes_only_authorized_replan(
        self,
        call_llm,
    ):
        plan = start_next_step(create_plan("task-1", "goal", ["one"]))
        state = create_state("goal")

        action = decide_step_action(
            state,
            plan,
            replan_reason="search returned no candidates",
        )

        self.assertEqual(action["type"], "replan")
        self.assertEqual(
            action["step_descriptions"],
            ["Use an alternate source"],
        )
        kwargs = call_llm.call_args.kwargs
        replan_schema = kwargs["tools"][0]
        self.assertEqual(replan_schema["name"], "submit_plan")
        self.assertEqual(
            replan_schema["parameters"]["properties"]["steps"]["maxItems"],
            7,
        )
        self.assertEqual(
            SUBMIT_PLAN_SCHEMA["parameters"]["properties"]["steps"][
                "maxItems"
            ],
            8,
        )
        self.assertIn(
            "search returned no candidates",
            kwargs["messages"][0]["content"],
        )
        self.assertIn(
            "at most 7 steps",
            kwargs["messages"][0]["content"],
        )

    def test_submit_plan_is_rejected_without_runtime_authorization(self):
        response = {
            "type": "tool_call",
            "content": None,
            "tool_call_id": "replan-call-1",
            "tool_name": "submit_plan",
            "tool_arguments": {"steps": ["replacement"]},
        }

        with self.assertRaisesRegex(ExecutorResponseError, "not authorized"):
            normalize_step_action(response, allow_replan=False)

    def test_request_clarification_becomes_an_internal_block_action(self):
        response = {
            "type": "tool_call",
            "content": None,
            "tool_call_id": "clarify-call-1",
            "tool_name": "request_clarification",
            "tool_arguments": {"question": "  Which paper do you mean?  "},
            "reasoning_content": "opaque reasoning",
        }

        action = normalize_step_action(response, allow_replan=False)

        self.assertEqual(
            action,
            {
                "type": "blocked",
                "content": "Which paper do you mean?",
                "tool_call_id": "clarify-call-1",
                "reasoning_content": "opaque reasoning",
            },
        )

    def test_executor_rejects_a_plan_without_a_running_step(self):
        plan = create_plan("task-1", "goal", ["one"])

        with self.assertRaisesRegex(ValueError, "running Plan step"):
            build_step_instructions(plan)


if __name__ == "__main__":
    unittest.main()
