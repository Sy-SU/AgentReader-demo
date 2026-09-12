import unittest
from copy import deepcopy
from unittest.mock import patch

from agent import ALLOWED_TOOLS
from planner import (
    INITIAL_ACTION_TOOLS,
    PLANNING_INSTRUCTIONS,
    SUBMIT_PLAN_SCHEMA,
    PlannerResponseError,
    decide_initial_action,
    normalize_initial_action,
)
from runtime import TOOL_REGISTRY
from state import create_state


PLAN_CALL = {
    "type": "tool_call",
    "content": None,
    "tool_call_id": "plan-call-1",
    "tool_name": "submit_plan",
    "tool_arguments": {
        "steps": [
            "  Identify the two requested papers  ",
            "Collect method and experiment evidence for each paper",
            "Compare the papers with page-level citations",
        ]
    },
}


class PlannerContractTests(unittest.TestCase):
    def test_submit_plan_schema_exposes_only_bounded_descriptions(self):
        self.assertEqual(SUBMIT_PLAN_SCHEMA["name"], "submit_plan")
        parameters = SUBMIT_PLAN_SCHEMA["parameters"]
        self.assertEqual(set(parameters["properties"]), {"steps"})
        self.assertEqual(parameters["required"], ["steps"])
        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(parameters["properties"]["steps"]["minItems"], 1)
        self.assertEqual(parameters["properties"]["steps"]["maxItems"], 8)
        self.assertEqual(
            parameters["properties"]["steps"]["items"]["maxLength"],
            500,
        )

    def test_submit_plan_is_not_an_executable_tool(self):
        self.assertNotIn("submit_plan", TOOL_REGISTRY)
        self.assertNotIn(
            "submit_plan",
            {schema["name"] for schema in ALLOWED_TOOLS},
        )
        self.assertIn(
            "submit_plan",
            {schema["name"] for schema in INITIAL_ACTION_TOOLS},
        )

    @patch("planner.call_llm", return_value=PLAN_CALL)
    def test_initial_decision_uses_fake_llm_and_normalizes_plan(self, fake_llm):
        state = create_state("Compare two papers with evidence")
        original_state = deepcopy(state)

        action = decide_initial_action(state)

        self.assertEqual(state, original_state)
        self.assertEqual(
            action,
            {
                "type": "plan",
                "content": None,
                "tool_call_id": "plan-call-1",
                "step_descriptions": [
                    "Identify the two requested papers",
                    "Collect method and experiment evidence for each paper",
                    "Compare the papers with page-level citations",
                ],
            },
        )
        call = fake_llm.call_args
        self.assertEqual(call.kwargs["messages"][0]["role"], "system")
        self.assertEqual(
            call.kwargs["messages"][0]["content"],
            PLANNING_INSTRUCTIONS,
        )
        self.assertEqual(
            call.kwargs["messages"][1:],
            state["messages"],
        )
        self.assertIn(SUBMIT_PLAN_SCHEMA, call.kwargs["tools"])

    def test_simple_tool_call_passes_through_without_forcing_a_plan(self):
        tool_call = {
            "type": "tool_call",
            "content": None,
            "tool_call_id": "search-call-1",
            "tool_name": "search_paper",
            "tool_arguments": {"query": "Attention Is All You Need"},
        }

        action = normalize_initial_action(tool_call)
        action["tool_arguments"]["query"] = "changed"

        self.assertEqual(
            tool_call["tool_arguments"]["query"],
            "Attention Is All You Need",
        )

    def test_final_response_passes_through_without_forcing_a_plan(self):
        final = {
            "type": "final",
            "content": "Please provide the second paper title.",
            "tool_call_id": None,
            "tool_name": None,
            "tool_arguments": None,
        }

        self.assertEqual(normalize_initial_action(final), final)

    def test_submit_plan_rejects_runtime_owned_or_extra_fields(self):
        for extra_field, value in (
            ("status", "completed"),
            ("task_id", "model-task"),
            ("budget", {"llm_steps": 0}),
            ("evidence_refs", ["made-up"]),
        ):
            invalid = deepcopy(PLAN_CALL)
            invalid["tool_arguments"][extra_field] = value
            with self.subTest(extra_field=extra_field):
                with self.assertRaisesRegex(
                    PlannerResponseError,
                    "only the steps field",
                ):
                    normalize_initial_action(invalid)

    def test_submit_plan_rejects_invalid_step_lists(self):
        invalid_steps = [
            [],
            ["step"] * 9,
            [""],
            ["x" * 501],
            [{"description": "model supplied structure"}],
            "not-a-list",
        ]

        for steps in invalid_steps:
            invalid = deepcopy(PLAN_CALL)
            invalid["tool_arguments"]["steps"] = steps
            with self.subTest(steps=steps):
                with self.assertRaisesRegex(
                    PlannerResponseError,
                    "invalid submit_plan steps",
                ):
                    normalize_initial_action(invalid)

    def test_malformed_initial_responses_are_rejected(self):
        invalid_responses = [
            None,
            {"type": "unknown"},
            {"type": "final", "content": None},
            {
                "type": "tool_call",
                "tool_call_id": "call-1",
                "tool_name": "search_paper",
                "tool_arguments": [],
            },
        ]

        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(PlannerResponseError):
                    normalize_initial_action(response)


if __name__ == "__main__":
    unittest.main()
