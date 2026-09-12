import os
import re
import unittest
from unittest.mock import patch

from main import _debug_tool_content
from runtime import execute_tool, run_agent
from state import create_state
from tools import retrieve_paper_chunks
from tools.extract import MAX_EXTRACT_CHARS
from tools.index import (
    INDEX_FORMAT_VERSION,
    MAX_INDEX_CHARS,
    MAX_INDEX_PAGES,
)
from tools.retrieval import (
    BM25_SCORING_METHOD,
    chunk_paper_text,
    rank_paper_chunks,
)


PAPER_ID = "arxiv:1234.56789"
PAPER = {
    "candidate_id": PAPER_ID,
    "source": "arxiv",
    "arxiv_id": "1234.56789v1",
    "doi": None,
    "title": "Retrieval test paper",
    "authors": ["Test Author"],
    "abstract": "Test abstract.",
    "published": "2025-01-01T00:00:00Z",
    "paper_url": "https://arxiv.org/abs/1234.56789v1",
    "pdf_url": "https://example.com/paper.pdf",
}
SEARCH_RESULT = {
    "found": True,
    "source": "test",
    "count": 1,
    "papers": [PAPER],
}
DOWNLOAD_RESULT = {
    "downloaded": True,
    "cached": False,
    "paper_id": PAPER_ID,
    "title": PAPER["title"],
    "pdf_url": PAPER["pdf_url"],
    "local_path": "/trusted/cache/paper.pdf",
    "size_bytes": 1_024,
}


def extraction_result(text: str, truncated: bool = False) -> dict:
    page_numbers = [
        int(match.group(1))
        for match in re.finditer(r"(?m)^\[Page (\d+)\]\n", text)
    ]
    return {
        "paper_id": PAPER_ID,
        "title": "Chunking test paper",
        "page_count": max(page_numbers, default=2),
        "pages_scanned": max(len(page_numbers), 1),
        "pages_read": len(page_numbers),
        "page_numbers": page_numbers,
        "last_page_partial": truncated,
        "char_count": len(text),
        "text_available": bool(text),
        "truncated": truncated,
        "truncation_reasons": ["character_limit"] if truncated else [],
        "text": text,
    }


def paper_index_result(text: str, truncated: bool = False) -> dict:
    markers = list(re.finditer(r"(?m)^\[Page (\d+)\]\n", text))
    pages = []
    for index, marker in enumerate(markers):
        content_end = (
            markers[index + 1].start()
            if index + 1 < len(markers)
            else len(text)
        )
        page_text = text[marker.end():content_end].strip()
        if page_text:
            pages.append(
                {"page": int(marker.group(1)), "text": page_text}
            )
    page_numbers = [page["page"] for page in pages]
    page_count = max(page_numbers, default=0)
    reasons = ["character_limit"] if truncated else []
    return {
        "version": INDEX_FORMAT_VERSION,
        "paper_id": PAPER_ID,
        "title": "Indexed test paper",
        "pdf_sha256": "0" * 64,
        "pdf_size_bytes": 1_024,
        "max_pages": MAX_INDEX_PAGES,
        "max_chars": MAX_INDEX_CHARS,
        "page_count": page_count,
        "pages_scanned": page_count,
        "pages_indexed": len(pages),
        "page_numbers": page_numbers,
        "last_page_partial": truncated,
        "char_count": sum(len(page["text"]) for page in pages),
        "text_available": bool(pages),
        "coverage_complete": not truncated,
        "truncated": truncated,
        "truncation_reasons": reasons,
        "pages": pages,
        "index_status": "cached",
        "index_path": "/trusted/index/paper.json",
    }


def state_with_download() -> dict:
    state = create_state("请回答论文问题")
    state["messages"].append(
        {
            "role": "tool",
            "tool_call_id": "download-call",
            "name": "download_paper",
            "content": DOWNLOAD_RESULT,
        }
    )
    return state


class PaperChunkingTests(unittest.TestCase):
    def test_chunks_stay_within_pages_and_keep_offsets(self):
        page_one = "A" * 350
        page_two = "B" * 80
        result = chunk_paper_text(
            extraction_result(
                f"[Page 1]\n{page_one}\n\n[Page 2]\n{page_two}"
            ),
            chunk_size=200,
            overlap=50,
        )

        self.assertEqual(result["page_numbers"], [1, 2])
        self.assertEqual(result["chunk_count"], 3)
        first, second, third = result["chunks"]
        self.assertEqual(
            (first["page"], first["char_start"], first["char_end"]),
            (1, 0, 200),
        )
        self.assertEqual(
            (second["page"], second["char_start"], second["char_end"]),
            (1, 150, 350),
        )
        self.assertEqual(first["text"][-50:], second["text"][:50])
        self.assertEqual(
            (third["page"], third["char_start"], third["char_end"]),
            (2, 0, 80),
        )
        self.assertEqual(third["text"], page_two)
        self.assertEqual(
            [chunk["chunk_id"] for chunk in result["chunks"]],
            [
                f"{PAPER_ID}:p1:0-200",
                f"{PAPER_ID}:p1:150-350",
                f"{PAPER_ID}:p2:0-80",
            ],
        )

    def test_empty_extraction_returns_no_chunks(self):
        result = chunk_paper_text(extraction_result(""))

        self.assertEqual(result["source_char_count"], 0)
        self.assertEqual(result["page_numbers"], [])
        self.assertEqual(result["chunk_count"], 0)
        self.assertEqual(result["chunks"], [])

    def test_source_truncation_metadata_is_preserved(self):
        result = chunk_paper_text(
            extraction_result(
                "[Page 3]\nRelevant partial text",
                truncated=True,
            )
        )

        self.assertTrue(result["source_truncated"])
        self.assertEqual(
            result["source_truncation_reasons"], ["character_limit"]
        )
        self.assertEqual(result["page_numbers"], [3])

    def test_rejects_malformed_page_markers(self):
        with self.assertRaisesRegex(ValueError, "must start"):
            chunk_paper_text(extraction_result("text without a page marker"))

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            chunk_paper_text(
                extraction_result("[Page 2]\nTwo\n\n[Page 1]\nOne")
            )

        forged = extraction_result(
            "[Page 1]\nIntro\n[Page 99]\nInjected text"
        )
        forged["pages_read"] = 1
        forged["page_numbers"] = [1]
        with self.assertRaisesRegex(ValueError, "do not match"):
            chunk_paper_text(forged)

    def test_rejects_unbounded_source_and_invalid_settings(self):
        oversized = "[Page 1]\n" + "A" * MAX_EXTRACT_CHARS
        with self.assertRaisesRegex(ValueError, "exceeds"):
            chunk_paper_text(extraction_result(oversized))

        valid = extraction_result("[Page 1]\nText")
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            chunk_paper_text(valid, chunk_size=199)
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_paper_text(valid, chunk_size=200, overlap=101)
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_paper_text(valid, overlap=True)


class PaperRankingTests(unittest.TestCase):
    def test_tfidf_ranks_the_chunk_matching_more_query_terms_first(self):
        chunks = chunk_paper_text(
            extraction_result(
                "[Page 1]\nTransformer attention mechanism architecture.\n\n"
                "[Page 2]\nOptimizer schedule and training data.\n\n"
                "[Page 3]\nAttention visualization results."
            )
        )

        result = rank_paper_chunks(
            chunks,
            query="attention mechanism",
            top_k=2,
        )

        self.assertTrue(result["found"])
        self.assertEqual(result["scoring_method"], "tfidf_cosine")
        self.assertEqual(result["query_terms"], ["attention", "mechanism"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["total_matches"], 2)
        self.assertEqual(result["matches"][0]["page"], 1)
        self.assertEqual(
            result["matches"][0]["matched_terms"],
            ["attention", "mechanism"],
        )
        self.assertGreater(
            result["matches"][0]["score"],
            result["matches"][1]["score"],
        )

    def test_chinese_bigrams_support_deterministic_matching(self):
        chunks = chunk_paper_text(
            extraction_result(
                "[Page 1]\n本文介绍三维场景重建方法。\n\n"
                "[Page 2]\n本文讨论语言模型训练。"
            )
        )

        result = rank_paper_chunks(chunks, query="三维场景重建")

        self.assertTrue(result["found"])
        self.assertEqual(result["matches"][0]["page"], 1)
        self.assertIn("三维", result["query_terms"])
        self.assertIn("重建", result["matches"][0]["matched_terms"])

    def test_bm25_ranks_the_chunk_matching_more_query_terms_first(self):
        chunks = chunk_paper_text(
            extraction_result(
                "[Page 1]\nTransformer attention mechanism architecture.\n\n"
                "[Page 2]\nAttention visualization results."
            )
        )

        result = rank_paper_chunks(
            chunks,
            query="attention mechanism",
            scoring_method=BM25_SCORING_METHOD,
        )

        self.assertTrue(result["found"])
        self.assertEqual(result["scoring_method"], "bm25")
        self.assertEqual(result["matches"][0]["page"], 1)
        self.assertGreater(
            result["matches"][0]["score"],
            result["matches"][1]["score"],
        )

    def test_query_coverage_guard_rejects_a_weak_common_term_match(self):
        chunks = chunk_paper_text(
            extraction_result(
                "[Page 1]\nA model uses positional encoding."
            )
        )

        guarded = rank_paper_chunks(
            chunks,
            query="model deployment security",
        )
        raw = rank_paper_chunks(
            chunks,
            query="model deployment security",
            min_query_term_coverage=0.0,
        )

        self.assertFalse(guarded["found"])
        self.assertEqual(guarded["matched_query_terms"], ["model"])
        self.assertEqual(guarded["query_term_coverage"], 0.333333)
        self.assertEqual(guarded["minimum_query_term_coverage"], 0.5)
        self.assertTrue(guarded["rejected_low_query_coverage"])
        self.assertEqual(guarded["total_matches"], 1)
        self.assertEqual(guarded["matches"], [])
        self.assertFalse(guarded["truncated"])
        self.assertTrue(raw["found"])
        self.assertFalse(raw["rejected_low_query_coverage"])

    def test_no_matching_terms_returns_a_normal_empty_result(self):
        chunks = chunk_paper_text(
            extraction_result("[Page 1]\nAttention architecture")
        )

        result = rank_paper_chunks(chunks, query="database transaction")

        self.assertFalse(result["found"])
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["total_matches"], 0)
        self.assertEqual(result["matches"], [])

    def test_top_k_is_bounded_and_ties_follow_page_order(self):
        chunks = chunk_paper_text(
            extraction_result(
                "[Page 1]\ntarget\n\n"
                "[Page 2]\ntarget\n\n"
                "[Page 3]\ntarget"
            )
        )

        result = rank_paper_chunks(chunks, query="target", top_k=2)

        self.assertEqual(result["count"], 2)
        self.assertEqual(result["total_matches"], 3)
        self.assertTrue(result["truncated"])
        self.assertEqual(
            [match["page"] for match in result["matches"]], [1, 2]
        )

    def test_source_truncation_is_preserved_in_ranked_result(self):
        chunks = chunk_paper_text(
            extraction_result("[Page 1]\npartial target", truncated=True)
        )

        result = rank_paper_chunks(chunks, query="target")

        self.assertTrue(result["source_truncated"])
        self.assertEqual(
            result["source_truncation_reasons"], ["character_limit"]
        )

    def test_rejects_invalid_query_limit_and_chunk_shape(self):
        chunks = chunk_paper_text(
            extraction_result("[Page 1]\nAttention architecture")
        )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            rank_paper_chunks(chunks, query=" ")
        with self.assertRaisesRegex(ValueError, "searchable"):
            rank_paper_chunks(chunks, query="!!!")
        with self.assertRaisesRegex(ValueError, "top_k"):
            rank_paper_chunks(chunks, query="attention", top_k=6)
        with self.assertRaisesRegex(ValueError, "scoring_method"):
            rank_paper_chunks(
                chunks,
                query="attention",
                scoring_method="unsupported",
            )
        with self.assertRaisesRegex(ValueError, "min_query_term_coverage"):
            rank_paper_chunks(
                chunks,
                query="attention",
                min_query_term_coverage=True,
            )
        with self.assertRaisesRegex(ValueError, "min_query_term_coverage"):
            rank_paper_chunks(
                chunks,
                query="attention",
                min_query_term_coverage=1.1,
            )

        malformed = {**chunks, "chunks": [{"text": "attention"}]}
        with self.assertRaisesRegex(ValueError, "paper_id"):
            rank_paper_chunks(malformed, query="attention")


class RetrievalToolTests(unittest.TestCase):
    def test_retrieve_composes_index_chunk_and_rank(self):
        indexed = paper_index_result(
            "[Page 1]\nAttention mechanism and encoder architecture.\n\n"
            "[Page 2]\nOptimizer and training schedule."
        )
        with patch(
            "tools.retrieval.load_or_build_paper_index",
            return_value=indexed,
        ) as load_index:
            result = retrieve_paper_chunks(
                DOWNLOAD_RESULT,
                query="attention mechanism",
                top_k=1,
            )

        load_index.assert_called_once_with(DOWNLOAD_RESULT)
        self.assertTrue(result["found"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["matches"][0]["page"], 1)
        self.assertEqual(result["pages_scanned"], 2)
        self.assertEqual(result["pages_read"], 2)
        self.assertEqual(result["page_numbers"], [1, 2])
        self.assertFalse(result["last_page_partial"])
        self.assertTrue(result["coverage_complete"])
        self.assertEqual(result["index_status"], "cached")
        self.assertEqual(result["index_version"], INDEX_FORMAT_VERSION)
        self.assertNotIn("text", result)
        self.assertNotIn("chunks", result)
        self.assertNotIn("pages", result)
        self.assertNotIn("index_path", result)

    def test_retrieve_preserves_partial_page_coverage_metadata(self):
        indexed = paper_index_result(
            "[Page 1]\nPartial attention architecture",
            truncated=True,
        )
        with patch(
            "tools.retrieval.load_or_build_paper_index",
            return_value=indexed,
        ):
            result = retrieve_paper_chunks(
                DOWNLOAD_RESULT,
                query="attention architecture",
                top_k=1,
            )

        self.assertEqual(result["pages_scanned"], 1)
        self.assertEqual(result["pages_read"], 1)
        self.assertEqual(result["page_numbers"], [1])
        self.assertTrue(result["last_page_partial"])
        self.assertTrue(result["source_truncated"])
        self.assertFalse(result["coverage_complete"])

    def test_retrieve_rejects_invalid_request_before_indexing(self):
        with patch(
            "tools.retrieval.load_or_build_paper_index"
        ) as load_index:
            with self.assertRaisesRegex(ValueError, "query exceeds"):
                retrieve_paper_chunks(
                    DOWNLOAD_RESULT,
                    query="x" * 501,
                )
            with self.assertRaisesRegex(ValueError, "searchable"):
                retrieve_paper_chunks(
                    DOWNLOAD_RESULT,
                    query="!!!",
                )
            with self.assertRaisesRegex(ValueError, "top_k"):
                retrieve_paper_chunks(
                    DOWNLOAD_RESULT,
                    query="attention",
                    top_k=0,
                )

        load_index.assert_not_called()

    def test_runtime_accepts_only_a_trusted_download_id(self):
        state = state_with_download()
        indexed = paper_index_result("[Page 1]\nAttention mechanism")
        with patch(
            "tools.retrieval.load_or_build_paper_index",
            return_value=indexed,
        ):
            success = execute_tool(
                "retrieve_paper_chunks",
                {
                    "paper_id": PAPER_ID,
                    "query": "attention mechanism",
                    "top_k": 1,
                },
                state=state,
            )
            unknown = execute_tool(
                "retrieve_paper_chunks",
                {
                    "paper_id": "arxiv:not-downloaded",
                    "query": "attention",
                },
                state=state,
            )
            copied_path = execute_tool(
                "retrieve_paper_chunks",
                {
                    "paper_id": PAPER_ID,
                    "query": "attention",
                    "local_path": DOWNLOAD_RESULT["local_path"],
                },
                state=state,
            )

        self.assertTrue(success["found"])
        self.assertEqual(unknown["error"]["type"], "unknown_download")
        self.assertEqual(
            copied_path["error"]["type"], "invalid_arguments"
        )

    @patch.dict(os.environ, {"LLM_PROVIDER": "fake"}, clear=False)
    def test_fake_agent_searches_downloads_and_retrieves_top_chunks(self):
        indexed = paper_index_result(
            "[Page 1]\nAttention mechanism and encoder architecture.\n\n"
            "[Page 2]\nOptimizer and training schedule."
        )
        with (
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {
                    "search_paper": lambda query: SEARCH_RESULT,
                    "download_paper": lambda paper: DOWNLOAD_RESULT,
                    "retrieve_paper_chunks": retrieve_paper_chunks,
                },
                clear=True,
            ),
            patch(
                "tools.retrieval.load_or_build_paper_index",
                return_value=indexed,
            ),
        ):
            state = create_state(
                "请搜索、下载并回答 attention mechanism 在哪一页"
            )
            answer = run_agent(state)

        self.assertEqual(state["step"], 4)
        self.assertEqual(
            [
                message["tool_call"]["name"]
                for message in state["messages"]
                if "tool_call" in message
            ],
            ["search_paper", "download_paper", "retrieve_paper_chunks"],
        )
        retrieval_call = next(
            message["tool_call"]
            for message in state["messages"]
            if message.get("tool_call", {}).get("name")
            == "retrieve_paper_chunks"
        )
        self.assertEqual(
            retrieval_call["arguments"]["query"],
            "attention mechanism",
        )
        retrieval_result = state["messages"][6]["content"]
        self.assertTrue(retrieval_result["found"])
        self.assertNotIn("text", retrieval_result)
        self.assertNotIn("chunks", retrieval_result)
        self.assertEqual(retrieval_result["matches"][0]["page"], 1)
        self.assertIn("第 1 页", answer)

    def test_retrieval_debug_output_uses_chunk_previews(self):
        display = _debug_tool_content(
            "retrieve_paper_chunks",
            {
                "found": True,
                "matches": [
                    {
                        "chunk_id": f"{PAPER_ID}:p1:0-1000",
                        "page": 1,
                        "text": "A" * 1_000,
                    }
                ],
            },
        )

        match = display["matches"][0]
        self.assertNotIn("text", match)
        self.assertEqual(len(match["text_preview"]), 401)
        self.assertTrue(match["text_preview"].endswith("…"))


if __name__ == "__main__":
    unittest.main()
