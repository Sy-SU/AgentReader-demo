import unittest

from evals.execution import (
    ExecutionEvaluationTracker,
    format_execution_evaluation,
)


class ExecutionEvaluationTests(unittest.TestCase):
    def test_completed_tool_turn_scores_100(self):
        tracker = ExecutionEvaluationTracker()
        for event in (
            {"kind": "llm_started", "turn_step": 1},
            {
                "kind": "llm_finished",
                "turn_step": 1,
                "duration_ms": 12.5,
            },
            {"kind": "tool_started", "tool_name": "search_paper"},
            {
                "kind": "tool_finished",
                "tool_name": "search_paper",
                "duration_ms": 20,
                "result": {"found": True, "papers": []},
            },
            {"kind": "llm_started", "turn_step": 2},
            {
                "kind": "llm_finished",
                "turn_step": 2,
                "duration_ms": 7.5,
            },
            {"kind": "turn_finished", "status": "completed"},
        ):
            tracker.observe(event)

        result = tracker.evaluate()

        self.assertEqual(result["score"], 100.0)
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["dimensions"]["tool_success"], 100.0)
        self.assertEqual(result["metrics"]["llm_duration_ms"], 20.0)
        self.assertNotIn("papers", result)
        self.assertIn("不代表答案正确率", format_execution_evaluation(result))

    def test_tool_and_budget_errors_reduce_the_score(self):
        tracker = ExecutionEvaluationTracker()
        for event in (
            {"kind": "llm_started"},
            {"kind": "llm_finished"},
            {"kind": "tool_started"},
            {
                "kind": "tool_finished",
                "result": {
                    "error": {
                        "type": "duplicate_search_query",
                        "message": "duplicate",
                    }
                },
            },
            {"kind": "turn_finished", "status": "completed"},
        ):
            tracker.observe(event)

        result = tracker.evaluate()

        self.assertEqual(result["score"], 70.0)
        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["metrics"]["tool_errors"], 1)
        self.assertEqual(result["metrics"]["runtime_guard_errors"], 1)

    def test_runtime_failure_is_unhealthy(self):
        tracker = ExecutionEvaluationTracker()
        tracker.observe({"kind": "llm_started"})
        tracker.observe(
            {
                "kind": "run_failed",
                "error_type": "RuntimeError",
                "message": "provider unavailable",
            }
        )

        result = tracker.evaluate()

        self.assertLess(result["score"], 20)
        self.assertEqual(result["status"], "unhealthy")
        self.assertEqual(result["metrics"]["run_failures"], 1)

    def test_cancelled_turn_is_not_scored(self):
        tracker = ExecutionEvaluationTracker()
        tracker.observe({"kind": "turn_finished", "status": "cancelled"})

        result = tracker.evaluate()

        self.assertIsNone(result["score"])
        self.assertEqual(result["reason"], "user_cancelled")
        self.assertIn("未评分", format_execution_evaluation(result))


if __name__ == "__main__":
    unittest.main()
