import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.litsearch import (
    evaluate_litsearch,
    format_litsearch_report,
    load_litsearch_queries,
    load_litsearch_results,
)


FIXTURE = [
    {
        "query_set": "inline_acl",
        "query": "Find broad work on two topics.",
        "specificity": 0,
        "quality": 2,
        "corpusids": [101, 102],
        "retrieved": [999, 101, 102],
    },
    {
        "query_set": "author_iclr",
        "query": "Find one specific paper.",
        "specificity": 1,
        "quality": 2,
        "corpusids": [201],
        "retrieved": [999, 201],
    },
    {
        "query_set": "author_iclr",
        "query": "Find another specific paper.",
        "specificity": 1,
        "quality": 1,
        "corpusids": [301],
        "retrieved": [],
    },
]


class LitSearchEvaluationTests(unittest.TestCase):
    def test_scores_official_broad_and_specific_cutoffs(self):
        result = evaluate_litsearch(FIXTURE)

        self.assertEqual(result["case_count"], 3)
        self.assertEqual(
            result["official_metrics"],
            {
                "broad_recall_at_20": 1.0,
                "specific_recall_at_5": 0.5,
                "specific_recall_at_20": 0.5,
            },
        )
        self.assertEqual(result["all_queries"]["recall_at_5"], 0.666667)
        self.assertEqual(result["by_query_set"]["author_iclr"]["case_count"], 2)

    def test_accepts_official_paragraph_result_pairs(self):
        fixture = [
            {
                **FIXTURE[0],
                "retrieved": [[101, 4], [101, 8], [102, 1]],
            }
        ]

        result = evaluate_litsearch(fixture)

        self.assertEqual(result["all_queries"]["recall_at_5"], 1.0)
        self.assertEqual(
            result["cases"][0]["retrieved_corpusids"],
            [101, 101, 102],
        )

    def test_loads_dataset_viewer_rows_and_jsonl(self):
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            viewer_path = directory_path / "viewer.json"
            viewer_path.write_text(
                json.dumps(
                    {"rows": [{"row_idx": 0, "row": FIXTURE[0]}]}
                ),
                encoding="utf-8",
            )
            jsonl_path = directory_path / "results.jsonl"
            jsonl_path.write_text(
                json.dumps(FIXTURE[1]) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(len(load_litsearch_results(viewer_path)), 1)
            self.assertEqual(len(load_litsearch_results(jsonl_path)), 1)

    def test_rejects_missing_predictions(self):
        invalid = [{key: value for key, value in FIXTURE[0].items() if key != "retrieved"}]

        with self.assertRaisesRegex(ValueError, "retrieved"):
            evaluate_litsearch(invalid)

    def test_rejects_quality_outside_official_scale(self):
        invalid = [{**FIXTURE[0], "quality": 3}]

        with self.assertRaisesRegex(ValueError, "quality must be 1 or 2"):
            evaluate_litsearch(invalid)

    def test_query_loader_accepts_records_before_retrieval(self):
        with TemporaryDirectory() as directory:
            query_path = Path(directory) / "queries.json"
            query = {
                key: value
                for key, value in FIXTURE[0].items()
                if key != "retrieved"
            }
            query_path.write_text(json.dumps([query]), encoding="utf-8")

            loaded = load_litsearch_queries(query_path)

        self.assertEqual(loaded, [query])

    def test_report_and_cli_have_human_and_json_outputs(self):
        report = format_litsearch_report(evaluate_litsearch(FIXTURE))
        self.assertIn("Broad Recall@20: 1.000", report)
        self.assertIn("Specific Recall@5: 0.500", report)

        with TemporaryDirectory() as directory:
            result_path = Path(directory) / "results.json"
            result_path.write_text(json.dumps(FIXTURE), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "evaluate_litsearch.py",
                    str(result_path),
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["benchmark"], "LitSearch")
        self.assertEqual(payload["case_count"], 3)


if __name__ == "__main__":
    unittest.main()
