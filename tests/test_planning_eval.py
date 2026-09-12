import io
import json
import unittest
from contextlib import redirect_stdout

from evals.planning import (
    cli_main,
    evaluate_planning_runtime,
    format_planning_report,
)


class PlanningEvaluationTests(unittest.TestCase):
    def test_two_paper_success_metrics_are_deterministic(self):
        first = evaluate_planning_runtime()
        second = evaluate_planning_runtime()

        self.assertEqual(first, second)
        self.assertTrue(first["passed"])
        self.assertEqual(first["task_status"], "completed")
        self.assertEqual(first["plan_revision"], 1)
        self.assertEqual(
            first["metrics"],
            {
                "task_completed": True,
                "replan_count": 0,
                "llm_decisions": 6,
                "tool_executions": 2,
                "checkpoint_save_count": 13,
                "max_checkpoint_bytes": 4572,
            },
        )
        self.assertEqual(
            first["observations"]["search_queries"],
            ["Attention Is All You Need", "BERT"],
        )
        self.assertEqual(
            first["observations"]["evidence_refs"],
            ["tool-result-001", "tool-result-002"],
        )
        self.assertTrue(first["observations"]["checkpoint_cleared"])

    def test_cli_supports_human_and_json_reports(self):
        human_output = io.StringIO()
        with redirect_stdout(human_output):
            self.assertEqual(cli_main([]), 0)
        report = human_output.getvalue()
        self.assertIn("Planning Runtime Evaluation", report)
        self.assertIn("Task status: completed", report)

        json_output = io.StringIO()
        with redirect_stdout(json_output):
            self.assertEqual(cli_main(["--json"]), 0)
        result = json.loads(json_output.getvalue())
        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics"]["tool_executions"], 2)
        self.assertIn("Checkpoint saves", format_planning_report(result))


if __name__ == "__main__":
    unittest.main()
