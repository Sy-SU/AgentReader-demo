import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from planning import (
    MAX_TASK_LLM_STEPS,
    MAX_TASK_TOOL_CALLS,
    create_plan,
    record_plan_usage,
)
from runtime import run_agent
from state import append_user_message, create_state


PLAN_ACTION = {
    "type": "plan",
    "content": None,
    "tool_call_id": "plan-call-1",
    "step_descriptions": [
        "Identify both papers",
        "Collect page-level evidence",
        "Compare the methods and experiments",
    ],
}
STEP_FINAL = {
    "type": "final",
    "content": "Current step completed.",
    "tool_call_id": None,
    "tool_name": None,
    "tool_arguments": None,
}
SEARCH_CALL = {
    "type": "tool_call",
    "content": None,
    "tool_call_id": "unsafe/call id",
    "tool_name": "search_paper",
    "tool_arguments": {"query": "paper A"},
}


class RuntimePlanningTests(unittest.TestCase):
    def test_new_state_has_no_task_or_plan(self):
        state = create_state("Compare two papers")

        self.assertIsNone(state["task_id"])
        self.assertIsNone(state["plan"])

    @patch("runtime.uuid4", return_value=SimpleNamespace(hex="abc123"))
    @patch("runtime.decide_next_action", return_value=PLAN_ACTION)
    def test_runtime_creates_trusted_plan_without_executing_tool(
        self,
        decide,
        _uuid,
    ):
        state = create_state("Compare paper A and paper B")
        events = []

        answer = run_agent(state, on_event=events.append)

        decide.assert_called_once_with(state, allow_planning=True)
        self.assertEqual(state["task_id"], "task-abc123")
        self.assertEqual(state["plan"]["task_id"], "task-abc123")
        self.assertEqual(
            state["plan"]["goal"],
            "Compare paper A and paper B",
        )
        self.assertEqual(state["plan"]["budget"]["llm_steps"], 1)
        self.assertEqual(state["plan"]["budget"]["tool_calls"], 0)
        self.assertTrue(
            all(
                step["status"] == "pending"
                for step in state["plan"]["steps"]
            )
        )
        self.assertFalse(
            any(message["role"] == "tool" for message in state["messages"])
        )
        self.assertNotIn("submit_plan", str(state["messages"]))
        self.assertIn("已创建任务计划", answer)
        self.assertEqual(state["messages"][-1]["content"], answer)
        self.assertEqual(state["step"], 1)
        self.assertEqual(
            [event["kind"] for event in events],
            ["llm_started", "llm_finished", "turn_finished"],
        )
        self.assertEqual(events[1]["response_type"], "plan")
        self.assertEqual(events[2]["status"], "planned")

    @patch("runtime.uuid4", return_value=SimpleNamespace(hex="latest"))
    @patch("runtime.decide_next_action", return_value=PLAN_ACTION)
    def test_plan_goal_uses_latest_user_turn(self, _decide, _uuid):
        state = create_state("First simple request")
        append_user_message(state, "Now compare paper A and paper B")

        run_agent(state)

        self.assertEqual(
            state["plan"]["goal"],
            "Now compare paper A and paper B",
        )

    @patch("runtime.decide_plan_step", return_value=STEP_FINAL)
    @patch("runtime.decide_next_action")
    def test_active_plan_uses_executor_without_silent_replanning(
        self,
        decide,
        decide_step,
    ):
        state = create_state("Compare two papers")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Compare two papers",
            ["Identify papers", "Compare evidence"],
        )
        original_descriptions = [
            step["description"] for step in state["plan"]["steps"]
        ]

        answer = run_agent(state, max_steps=1)

        decide.assert_not_called()
        decide_step.assert_called_once()
        self.assertEqual(state["task_id"], "task-existing")
        self.assertEqual(
            [step["description"] for step in state["plan"]["steps"]],
            original_descriptions,
        )
        self.assertEqual(state["plan"]["steps"][0]["status"], "completed")
        self.assertEqual(state["plan"]["steps"][1]["status"], "pending")
        self.assertIn("max_steps=1", answer)

    @patch("runtime.decide_next_action")
    def test_executor_completes_two_steps_and_records_runtime_evidence(
        self,
        initial_decision,
    ):
        state = create_state("Compare paper A and paper B")
        state["task_id"] = "task-existing"
        state["plan"] = record_plan_usage(
            create_plan(
                "task-existing",
                "Compare paper A and paper B",
                ["Identify both papers", "Compare their methods"],
            ),
            llm_steps=1,
        )
        state["step"] = 1
        search = Mock(return_value={"found": False, "papers": []})
        step_plans = []

        def decide_step(_state, plan):
            step_plans.append(deepcopy(plan))
            return [SEARCH_CALL, STEP_FINAL, {**STEP_FINAL, "content": "Done."}][
                len(step_plans) - 1
            ]

        with (
            patch("runtime.decide_plan_step", side_effect=decide_step),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=3)

        initial_decision.assert_not_called()
        search.assert_called_once_with(query="paper A")
        self.assertEqual(answer, "Done.")
        self.assertEqual(state["plan"]["status"], "completed")
        self.assertEqual(
            [step["status"] for step in state["plan"]["steps"]],
            ["completed", "completed"],
        )
        self.assertEqual(
            state["plan"]["steps"][0]["evidence_refs"],
            ["tool-result-001"],
        )
        self.assertEqual(state["plan"]["steps"][1]["evidence_refs"], [])
        self.assertEqual(
            state["plan"]["budget"],
            {"llm_steps": 4, "tool_calls": 1, "replans": 0},
        )
        self.assertEqual(state["step"], 4)
        tool_message = next(
            message for message in state["messages"] if message["role"] == "tool"
        )
        self.assertEqual(tool_message["evidence_ref"], "tool-result-001")
        self.assertNotIn("unsafe/call id", str(state["plan"]))
        self.assertEqual(step_plans[0]["current_step_id"], "step-001")
        self.assertEqual(step_plans[1]["current_step_id"], "step-001")
        self.assertEqual(step_plans[2]["current_step_id"], "step-002")

    def test_tool_result_does_not_complete_the_current_step(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find evidence",
            ["Search for the evidence"],
        )
        search = Mock(return_value={"found": False, "papers": []})

        with (
            patch("runtime.decide_plan_step", return_value=SEARCH_CALL),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=1)

        self.assertIn("max_steps=1", answer)
        self.assertEqual(state["plan"]["status"], "running")
        self.assertEqual(state["plan"]["steps"][0]["status"], "running")
        self.assertEqual(
            state["plan"]["steps"][0]["evidence_refs"],
            ["tool-result-001"],
        )

    @patch("runtime.decide_plan_step")
    def test_executor_stops_before_exceeding_llm_budget(self, decide_step):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = record_plan_usage(
            create_plan(
                "task-existing",
                "Find evidence",
                ["Search for evidence"],
            ),
            llm_steps=MAX_TASK_LLM_STEPS,
        )

        answer = run_agent(state)

        decide_step.assert_not_called()
        self.assertEqual(state["plan"]["status"], "failed")
        self.assertEqual(
            state["plan"]["budget"]["llm_steps"],
            MAX_TASK_LLM_STEPS,
        )
        self.assertIn("LLM 决策上限", answer)

    def test_executor_stops_before_exceeding_tool_budget(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = record_plan_usage(
            create_plan(
                "task-existing",
                "Find evidence",
                ["Search for evidence"],
            ),
            tool_calls=MAX_TASK_TOOL_CALLS,
        )
        search = Mock(return_value={"found": False, "papers": []})

        with (
            patch("runtime.decide_plan_step", return_value=SEARCH_CALL),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            answer = run_agent(state)

        search.assert_not_called()
        self.assertEqual(state["plan"]["status"], "failed")
        self.assertEqual(
            state["plan"]["budget"]["tool_calls"],
            MAX_TASK_TOOL_CALLS,
        )
        self.assertIn("Tool 执行上限", answer)
        self.assertFalse(
            any(message.get("tool_call") for message in state["messages"])
        )

    @patch("runtime.decide_next_action")
    def test_simple_final_keeps_v2_path_without_creating_plan(self, decide):
        decide.return_value = {
            "type": "final",
            "content": "Please provide another title.",
            "tool_call_id": None,
            "tool_name": None,
            "tool_arguments": None,
        }
        state = create_state("Find one paper")

        answer = run_agent(state)

        decide.assert_called_once_with(state, allow_planning=True)
        self.assertEqual(answer, "Please provide another title.")
        self.assertIsNone(state["task_id"])
        self.assertIsNone(state["plan"])

    @patch("runtime.decide_next_action")
    def test_invalid_plan_action_does_not_partially_update_state(self, decide):
        decide.return_value = {
            **PLAN_ACTION,
            "task_id": "model-controlled-task",
        }
        state = create_state("Compare two papers")
        events = []

        with self.assertRaisesRegex(ValueError, "plan action contract"):
            run_agent(state, on_event=events.append)

        self.assertIsNone(state["task_id"])
        self.assertIsNone(state["plan"])
        self.assertEqual(events[-1]["kind"], "run_failed")


if __name__ == "__main__":
    unittest.main()
