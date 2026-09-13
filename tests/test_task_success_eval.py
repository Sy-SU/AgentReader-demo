import unittest

from evals.task_success import (
    evaluate_task_success,
    format_task_success_report,
    normalize_task_contract,
)


PAPER_A = "arxiv:1706.03762"
PAPER_B = "arxiv:1810.04805"


def contract(
    *,
    candidates=None,
    downloads=None,
    evidence=None,
    no_results=False,
):
    candidates = candidates or []
    downloads = downloads or []
    evidence = evidence or []
    required = {"search_paper": 1}
    allowed = ["search_paper"]
    if downloads:
        required["download_paper"] = len(downloads)
        allowed.append("download_paper")
    if evidence:
        required["retrieve_paper_chunks"] = len(evidence)
        allowed.append("retrieve_paper_chunks")
    return {
        "id": "test-case",
        "expectations": {
            "terminal_status": "completed",
            "required_tools": required,
            "allowed_tools": allowed,
            "candidate_ids": candidates,
            "download_ids": downloads,
            "evidence_ids": evidence,
            "expect_no_results": no_results,
        },
    }


def complete_state(tool_results, *, plan=True):
    messages = [{"role": "user", "content": "benchmark task"}]
    for index, (name, result) in enumerate(tool_results, start=1):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_call": {
                        "id": call_id,
                        "name": name,
                        "arguments": {},
                    },
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": result,
                },
            ]
        )
    messages.append({"role": "assistant", "content": "done"})
    return {
        "messages": messages,
        "plan": {"status": "completed"} if plan else None,
    }


class TaskSuccessEvaluationTests(unittest.TestCase):
    def test_complete_evidence_task_scores_100_without_retaining_text_or_path(self):
        state = complete_state(
            [
                (
                    "search_paper",
                    {
                        "found": True,
                        "papers": [
                            {"candidate_id": PAPER_A},
                            {"candidate_id": PAPER_B},
                        ],
                    },
                ),
                (
                    "download_paper",
                    {
                        "downloaded": True,
                        "paper_id": PAPER_A,
                        "local_path": "/secret/a.pdf",
                    },
                ),
                (
                    "download_paper",
                    {
                        "cached": True,
                        "paper_id": PAPER_B,
                        "local_path": "/secret/b.pdf",
                    },
                ),
                (
                    "retrieve_paper_chunks",
                    {
                        "found": True,
                        "paper_id": PAPER_A,
                        "matches": [{"page": 2, "text": "secret A text"}],
                    },
                ),
                (
                    "retrieve_paper_chunks",
                    {
                        "found": True,
                        "paper_id": PAPER_B,
                        "matches": [{"page": 3, "text": "secret B text"}],
                    },
                ),
            ]
        )

        report = evaluate_task_success(
            contract(
                candidates=[PAPER_A, PAPER_B],
                downloads=[PAPER_A, PAPER_B],
                evidence=[PAPER_A, PAPER_B],
            ),
            state,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["score"], 100.0)
        self.assertEqual(
            report["observations"]["evidence_pages"],
            {PAPER_A: [2], PAPER_B: [3]},
        )
        self.assertNotIn("secret", str(report))
        self.assertNotIn("local_path", str(report))

    def test_missing_candidate_download_and_valid_page_are_separate_failures(self):
        state = complete_state(
            [
                (
                    "search_paper",
                    {"found": True, "papers": [{"candidate_id": PAPER_A}]},
                ),
                (
                    "download_paper",
                    {"downloaded": True, "paper_id": PAPER_A},
                ),
                (
                    "retrieve_paper_chunks",
                    {
                        "found": True,
                        "paper_id": PAPER_A,
                        "matches": [{"page": None, "text": "text"}],
                    },
                ),
            ]
        )

        report = evaluate_task_success(
            contract(
                candidates=[PAPER_A, PAPER_B],
                downloads=[PAPER_A, PAPER_B],
                evidence=[PAPER_A, PAPER_B],
            ),
            state,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            report["dimensions"]["candidate_discovery"]["score"], 50.0
        )
        self.assertEqual(
            report["dimensions"]["paper_selection"]["score"], 50.0
        )
        self.assertEqual(report["dimensions"]["page_evidence"]["score"], 0.0)

    def test_normal_no_result_and_safe_stop_are_scored_without_text_judgment(self):
        state = complete_state(
            [("search_paper", {"found": False, "count": 0, "papers": []})],
            plan=False,
        )

        report = evaluate_task_success(
            contract(no_results=True),
            state,
        )

        self.assertTrue(report["passed"])
        self.assertIsNone(report["dimensions"]["candidate_discovery"])
        self.assertEqual(report["dimensions"]["no_result_safety"]["score"], 100.0)

    def test_tool_error_and_unfinished_plan_reduce_independent_dimensions(self):
        state = complete_state(
            [
                (
                    "search_paper",
                    {"error": {"type": "search_failed", "message": "secret"}},
                )
            ]
        )
        state["plan"]["status"] = "blocked"

        report = evaluate_task_success(contract(no_results=True), state)

        self.assertEqual(report["dimensions"]["completion"]["score"], 0.0)
        self.assertFalse(report["dimensions"]["tool_use"]["passed"])
        self.assertEqual(report["observations"]["tool_error_types"], ["search_failed"])
        self.assertNotIn("secret", str(report))

    def test_invalid_contract_relationships_and_unpaired_calls_are_rejected(self):
        bad = contract(candidates=[PAPER_A], downloads=[PAPER_B])
        with self.assertRaisesRegex(ValueError, "download_ids must be candidates"):
            normalize_task_contract(bad)

        state = complete_state([])
        state["messages"].insert(
            -1,
            {
                "role": "assistant",
                "content": None,
                "tool_call": {"id": "unpaired", "name": "search_paper"},
            },
        )
        with self.assertRaisesRegex(ValueError, "without Tool Results"):
            evaluate_task_success(contract(no_results=True), state)

    def test_non_assistant_tool_calls_and_unbounded_observations_are_rejected(self):
        wrong_role = complete_state([], plan=False)
        wrong_role["messages"].insert(
            -1,
            {
                "role": "user",
                "content": "not a trusted call",
                "tool_call": {"id": "bad", "name": "search_paper"},
            },
        )
        with self.assertRaisesRegex(ValueError, "Tool Call"):
            evaluate_task_success(contract(no_results=True), wrong_role)

        oversized_id = "x" * 161
        state = complete_state(
            [
                (
                    "search_paper",
                    {
                        "found": True,
                        "papers": [{"candidate_id": oversized_id}],
                    },
                )
            ],
            plan=False,
        )
        report = evaluate_task_success(contract(no_results=True), state)
        self.assertEqual(report["observations"]["candidate_ids"], [])

    def test_formatter_adds_bounded_observations_only_in_debug_mode(self):
        report = evaluate_task_success(
            contract(no_results=True),
            complete_state(
                [("search_paper", {"found": False, "papers": []})],
                plan=False,
            ),
        )

        normal = format_task_success_report(report)
        debug = format_task_success_report(report, debug=True)

        self.assertNotIn("tools=", normal)
        self.assertIn("tools=", debug)
        self.assertIn("no-result=100.0", debug)


if __name__ == "__main__":
    unittest.main()
