import copy
import io
import json
import unittest
from contextlib import redirect_stdout

from evals.retrieval import (
    cli_main,
    compare_retrieval_methods,
    evaluate_retrieval_dataset,
    format_comparison_report,
    format_evaluation_report,
    load_retrieval_dataset,
    validate_retrieval_dataset,
)


class RetrievalEvaluationTests(unittest.TestCase):
    def test_default_dataset_locks_the_guarded_tfidf_baseline(self):
        result = evaluate_retrieval_dataset(load_retrieval_dataset())

        self.assertEqual(result["scoring_method"], "tfidf_cosine")
        self.assertEqual(result["minimum_query_term_coverage"], 0.5)
        self.assertEqual(result["top_k"], 3)
        self.assertEqual(result["case_count"], 10)
        self.assertEqual(result["answerable_count"], 8)
        self.assertEqual(result["no_answer_count"], 2)
        self.assertEqual(
            result["metrics"],
            {
                "recall_at_k": 0.875,
                "hit_rate_at_k": 0.875,
                "mrr": 0.875,
                "no_answer_accuracy": 1.0,
            },
        )

    def test_baseline_exposes_only_the_remaining_cross_language_limit(self):
        result = evaluate_retrieval_dataset(load_retrieval_dataset())
        cases = {case["id"]: case for case in result["cases"]}

        self.assertEqual(result["by_tag"]["lexical"]["metrics"]["mrr"], 1.0)
        self.assertEqual(
            result["by_tag"]["cross-language"]["metrics"]["mrr"],
            0.0,
        )
        self.assertFalse(cases["cross-language-position"]["passed"])
        self.assertEqual(
            cases["cross-language-position"]["retrieved_pages"], []
        )
        self.assertTrue(cases["no-answer-common-term"]["passed"])
        self.assertEqual(
            cases["no-answer-common-term"]["retrieved_pages"], []
        )
        self.assertEqual(
            cases["no-answer-common-term"]["query_term_coverage"],
            0.333333,
        )
        self.assertTrue(
            cases["no-answer-common-term"][
                "rejected_low_query_coverage"
            ]
        )

    def test_same_dataset_comparison_selects_guarded_tfidf(self):
        comparison = compare_retrieval_methods(load_retrieval_dataset())
        configurations = {
            result["configuration_id"]: result
            for result in comparison["configurations"]
        }

        self.assertEqual(
            set(configurations),
            {
                "tfidf_raw",
                "bm25_raw",
                "tfidf_guarded",
                "bm25_guarded",
            },
        )
        for configuration_id in ("tfidf_raw", "bm25_raw"):
            self.assertEqual(
                configurations[configuration_id]["metrics"],
                {
                    "recall_at_k": 0.875,
                    "hit_rate_at_k": 0.875,
                    "mrr": 0.875,
                    "no_answer_accuracy": 0.5,
                },
            )
        for configuration_id in ("tfidf_guarded", "bm25_guarded"):
            self.assertEqual(
                configurations[configuration_id]["metrics"],
                {
                    "recall_at_k": 0.875,
                    "hit_rate_at_k": 0.875,
                    "mrr": 0.875,
                    "no_answer_accuracy": 1.0,
                },
            )
        self.assertEqual(
            comparison["decision"]["selected_configuration"],
            "tfidf_guarded",
        )
        self.assertEqual(
            comparison["decision"]["selected_scoring_method"],
            "tfidf_cosine",
        )

    def test_dataset_validation_rejects_ambiguous_ground_truth(self):
        original = load_retrieval_dataset()

        duplicate_id = copy.deepcopy(original)
        duplicate_id["cases"][1]["id"] = duplicate_id["cases"][0]["id"]
        with self.assertRaisesRegex(ValueError, "ids must be unique"):
            validate_retrieval_dataset(duplicate_id)

        unknown_page = copy.deepcopy(original)
        unknown_page["cases"][0]["expected_pages"] = [99]
        with self.assertRaisesRegex(ValueError, "invalid expected_pages"):
            validate_retrieval_dataset(unknown_page)

    def test_cli_supports_human_and_json_reports(self):
        human_output = io.StringIO()
        with redirect_stdout(human_output):
            self.assertEqual(cli_main([]), 0)
        report = human_output.getvalue()
        self.assertIn("Recall@3: 0.875", report)
        self.assertIn("cross-language-position", report)

        json_output = io.StringIO()
        with redirect_stdout(json_output):
            self.assertEqual(cli_main(["--json"]), 0)
        result = json.loads(json_output.getvalue())
        self.assertEqual(result["metrics"]["no_answer_accuracy"], 1.0)

        self.assertIn(
            "Failed cases: 1",
            format_evaluation_report(result),
        )

        comparison_output = io.StringIO()
        with redirect_stdout(comparison_output):
            self.assertEqual(cli_main(["--compare"]), 0)
        comparison_report = comparison_output.getvalue()
        self.assertIn("tfidf_raw", comparison_report)
        self.assertIn("bm25_guarded", comparison_report)
        self.assertIn("Selected: tfidf_guarded", comparison_report)

        comparison = compare_retrieval_methods(load_retrieval_dataset())
        self.assertIn(
            "Selected: tfidf_guarded",
            format_comparison_report(comparison),
        )


if __name__ == "__main__":
    unittest.main()
