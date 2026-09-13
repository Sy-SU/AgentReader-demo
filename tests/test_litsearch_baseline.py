import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.litsearch import load_litsearch_results
from evals.litsearch_baseline import (
    RETRIEVAL_METHOD,
    format_litsearch_baseline_report,
    load_litsearch_corpus,
    run_litsearch_baseline,
)


CORPUS = [
    {
        "corpusid": 101,
        "title": "Attention Is All You Need",
        "abstract": (
            "A Transformer encoder decoder uses self attention for machine "
            "translation."
        ),
    },
    {
        "corpusid": 102,
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": (
            "BERT uses masked language modeling and next sentence prediction."
        ),
    },
    {
        "corpusid": 103,
        "title": "Convolutional Image Classification",
        "abstract": "A residual convolutional network for image recognition.",
    },
]

QUERIES = [
    {
        "query_set": "inline_acl",
        "query": "transformer self attention machine translation",
        "specificity": 0,
        "quality": 2,
        "corpusids": [101],
    },
    {
        "query_set": "author_iclr",
        "query": "masked language modeling next sentence prediction",
        "specificity": 1,
        "quality": 2,
        "corpusids": [102],
    },
]


class LitSearchBaselineTests(unittest.TestCase):
    def test_same_corpus_baseline_generates_scorable_ranked_ids(self):
        report = run_litsearch_baseline(CORPUS, QUERIES)

        self.assertEqual(report["retrieval_method"], RETRIEVAL_METHOD)
        self.assertFalse(report["official_baseline_compatible"])
        self.assertEqual(report["corpus_document_count"], 3)
        self.assertEqual(report["query_count"], 2)
        self.assertEqual(report["retrieval_results"][0]["retrieved"][0], 101)
        self.assertEqual(report["retrieval_results"][1]["retrieved"][0], 102)
        self.assertEqual(
            report["evaluation"]["official_metrics"],
            {
                "broad_recall_at_20": 1.0,
                "specific_recall_at_5": 1.0,
                "specific_recall_at_20": 1.0,
            },
        )

    def test_limit_is_deterministic_and_report_discloses_comparability(self):
        report = run_litsearch_baseline(CORPUS, QUERIES, limit=1)
        rendered = format_litsearch_baseline_report(report)

        self.assertEqual(report["query_count"], 1)
        self.assertEqual(
            report["retrieval_results"][0]["query"],
            QUERIES[0]["query"],
        )
        self.assertIn("Comparable to official retriever numbers: no", rendered)
        self.assertIn("Specific Recall@5: N/A", rendered)

    def test_top_k_cannot_undercut_official_recall_at_20(self):
        with self.assertRaisesRegex(ValueError, "Recall@20 is valid"):
            run_litsearch_baseline(CORPUS, QUERIES, top_k=19)

    def test_corpus_loader_supports_jsonl_and_dataset_viewer_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl_path = root / "corpus.jsonl"
            jsonl_path.write_text(
                "\n".join(json.dumps(record) for record in CORPUS) + "\n",
                encoding="utf-8",
            )
            viewer_path = root / "viewer.json"
            viewer_path.write_text(
                json.dumps(
                    {"rows": [{"row_idx": 0, "row": CORPUS[0]}]}
                ),
                encoding="utf-8",
            )

            self.assertEqual(load_litsearch_corpus(jsonl_path), CORPUS)
            self.assertEqual(load_litsearch_corpus(viewer_path), [CORPUS[0]])

    def test_duplicate_corpus_ids_are_rejected(self):
        duplicate = [CORPUS[0], {**CORPUS[1], "corpusid": 101}]

        with self.assertRaisesRegex(ValueError, "repeats corpusid 101"):
            run_litsearch_baseline(duplicate, QUERIES)

    def test_empty_official_text_is_kept_in_the_corpus_id_space(self):
        corpus = [
            CORPUS[0],
            {"corpusid": 104, "title": None, "abstract": None},
        ]

        report = run_litsearch_baseline(corpus, QUERIES[:1])

        self.assertEqual(report["corpus_document_count"], 2)
        self.assertEqual(report["retrieval_results"][0]["retrieved"], [101, 104])

    def test_cli_runs_retrieval_scores_and_writes_official_shape(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            corpus_path = root / "corpus.jsonl"
            query_path = root / "queries.json"
            output_path = root / "results.jsonl"
            corpus_path.write_text(
                "\n".join(json.dumps(record) for record in CORPUS) + "\n",
                encoding="utf-8",
            )
            query_path.write_text(json.dumps(QUERIES), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "run_litsearch_baseline.py",
                    str(corpus_path),
                    str(query_path),
                    "--output",
                    str(output_path),
                    "--json",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            saved_results = load_litsearch_results(output_path)

        self.assertEqual(payload["retrieval_method"], RETRIEVAL_METHOD)
        self.assertEqual(len(saved_results), 2)
        self.assertEqual(saved_results[0]["retrieved"][0], 101)


if __name__ == "__main__":
    unittest.main()
