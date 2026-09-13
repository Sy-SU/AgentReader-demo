import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.agent_tasks import (
    DEFAULT_CASES_PATH,
    cli_main,
    evaluate_agent_task_suite,
    format_agent_task_suite,
    load_agent_task_cases,
)


class AgentTaskSuiteTests(unittest.TestCase):
    def test_default_suite_runs_three_runtime_cases_deterministically(self):
        first = evaluate_agent_task_suite()
        second = evaluate_agent_task_suite()

        self.assertEqual(first, second)
        self.assertTrue(first["passed"])
        self.assertEqual(first["case_count"], 3)
        self.assertEqual(first["passed_count"], 3)
        self.assertEqual(first["mean_task_success_score"], 100.0)
        self.assertTrue(
            all(case["runtime"]["checkpoint_cleared"] for case in first["cases"])
        )

    def test_case_filter_keeps_requested_order(self):
        report = evaluate_agent_task_suite(
            case_ids=["unknown-paper-safe-stop", "exact-title-search"]
        )

        self.assertEqual(
            [case["case_id"] for case in report["cases"]],
            ["unknown-paper-safe-stop", "exact-title-search"],
        )

    def test_cli_supports_human_json_debug_and_case_filter(self):
        human_output = io.StringIO()
        with redirect_stdout(human_output):
            self.assertEqual(
                cli_main(["--case", "two-paper-evidence-comparison"]),
                0,
            )
        self.assertIn("Passed: 1/1", human_output.getvalue())
        self.assertNotIn("evidence_pages=", human_output.getvalue())

        debug_output = io.StringIO()
        with redirect_stdout(debug_output):
            self.assertEqual(
                cli_main(
                    [
                        "--case",
                        "two-paper-evidence-comparison",
                        "--debug",
                    ]
                ),
                0,
            )
        self.assertIn("evidence_pages=", debug_output.getvalue())
        self.assertNotIn("benchmark evidence", debug_output.getvalue())

        json_output = io.StringIO()
        with redirect_stdout(json_output):
            self.assertEqual(cli_main(["--case", "exact-title-search", "--json"]), 0)
        payload = json.loads(json_output.getvalue())
        self.assertEqual(payload["case_count"], 1)
        self.assertEqual(payload["cases"][0]["score"], 100.0)

    def test_cli_rejects_unknown_case_without_traceback(self):
        error_output = io.StringIO()
        with redirect_stderr(error_output):
            self.assertEqual(cli_main(["--case", "missing-case"]), 2)
        self.assertIn("Unknown Agent task cases", error_output.getvalue())

    def test_loader_rejects_version_and_duplicate_ids(self):
        payload = json.loads(DEFAULT_CASES_PATH.read_text(encoding="utf-8"))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            invalid_version = {**payload, "dataset_version": 99}
            path.write_text(json.dumps(invalid_version), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "version"):
                load_agent_task_cases(path)

            duplicate = {**payload, "cases": [payload["cases"][0]] * 2}
            path.write_text(json.dumps(duplicate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "IDs must be unique"):
                load_agent_task_cases(path)

    def test_report_discloses_fixed_provider_boundary(self):
        report = evaluate_agent_task_suite(case_ids=["exact-title-search"])
        rendered = format_agent_task_suite(report, debug=True)

        self.assertIn("Provider decisions", rendered)
        self.assertIn("task success 100.0/100", rendered)


if __name__ == "__main__":
    unittest.main()
