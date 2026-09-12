import unittest
from copy import deepcopy
from unittest.mock import patch

from agent import ALLOWED_TOOLS
from executor import build_step_instructions, decide_step_action
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
        self.assertEqual(kwargs["tools"], ALLOWED_TOOLS)
        self.assertNotIn("submit_plan", str(kwargs["tools"]))
        instructions = kwargs["messages"][0]["content"]
        self.assertIn("Compare paper A and paper B", instructions)
        self.assertIn("- Identify both papers", instructions)
        self.assertIn("Current step (step-002)", instructions)
        self.assertIn("Compare their methods", instructions)
        self.assertIn("step-level completion signal", instructions)
        self.assertEqual(kwargs["messages"][1:], state["messages"])

    def test_executor_rejects_a_plan_without_a_running_step(self):
        plan = create_plan("task-1", "goal", ["one"])

        with self.assertRaisesRegex(ValueError, "running Plan step"):
            build_step_instructions(plan)


if __name__ == "__main__":
    unittest.main()
