import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.litsearch import load_litsearch_queries
from evals.litsearch_baseline import load_litsearch_corpus
from evals.litsearch_data import (
    _project_corpus,
    _project_query,
    _select_parquet_files,
    _write_validated_jsonl,
)


class LitSearchDataTests(unittest.TestCase):
    def test_selects_only_query_and_clean_corpus_shards(self):
        files = [
            "query/full-00000-of-00001.parquet",
            *[
                f"corpus_clean/full-{index:05d}-of-00006.parquet"
                for index in range(6)
            ],
            "corpus_s2orc/full-00000-of-00008.parquet",
            "README.md",
        ]

        query, corpus = _select_parquet_files(reversed(files))

        self.assertEqual(query, ["query/full-00000-of-00001.parquet"])
        self.assertEqual(len(corpus), 6)
        self.assertTrue(all(path.startswith("corpus_clean/") for path in corpus))

    def test_unexpected_official_layout_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one query and six"):
            _select_parquet_files(["query/only.parquet"])

    def test_writes_minimal_validated_projections(self):
        query = {
            "query": "transformer attention",
            "query_set": "inline_acl",
            "specificity": 0,
            "quality": 2,
            "corpusids": [101],
            "unused": "not exported",
        }
        corpus = {
            "corpusid": 101,
            "title": "Attention Is All You Need",
            "abstract": "A Transformer based on attention.",
            "full_paper": "not exported",
        }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            query_path = root / "queries.jsonl"
            corpus_path = root / "corpus.jsonl"
            _write_validated_jsonl(
                [query], query_path, _project_query, load_litsearch_queries, 1
            )
            _write_validated_jsonl(
                [corpus], corpus_path, _project_corpus, load_litsearch_corpus, 1
            )

            saved_query = json.loads(query_path.read_text(encoding="utf-8"))
            saved_corpus = json.loads(corpus_path.read_text(encoding="utf-8"))

        self.assertNotIn("unused", saved_query)
        self.assertNotIn("full_paper", saved_corpus)
        self.assertEqual(saved_query["corpusids"], [101])
        self.assertEqual(saved_corpus["corpusid"], 101)

    def test_projection_count_must_match_expected_count(self):
        query = {
            "query": "transformer attention",
            "query_set": "inline_acl",
            "specificity": 0,
            "quality": 2,
            "corpusids": [101],
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "queries.jsonl"
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                _write_validated_jsonl(
                    [query],
                    output,
                    _project_query,
                    load_litsearch_queries,
                    2,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
