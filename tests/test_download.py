import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from runtime import execute_tool, run_agent
from state import create_state
from tools import download_paper
from tools.download import MAX_PDF_SIZE_BYTES


PDF_BYTES = b"%PDF-1.7\nminimal test PDF"
PAPER = {
    "candidate_id": "arxiv:1234.56789",
    "source": "arxiv",
    "arxiv_id": "1234.56789v1",
    "doi": None,
    "title": "TASA (test record)",
    "authors": ["Test Author"],
    "abstract": "Test abstract.",
    "published": "2025-01-01T00:00:00Z",
    "paper_url": "https://arxiv.org/abs/1234.56789v1",
    "pdf_url": "https://example.com/tasa.pdf",
}
SEARCH_RESULT = {
    "found": True,
    "source": "test",
    "count": 1,
    "papers": [PAPER],
}


class FakeResponse:
    def __init__(self, data=PDF_BYTES, headers=None, url=None):
        self.data = data
        self.headers = headers or {"Content-Type": "application/pdf"}
        self.url = url or PAPER["pdf_url"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def geturl(self):
        return self.url

    def read(self, size=-1):
        return self.data if size < 0 else self.data[:size]


def state_with_search_result(paper=None):
    result = {**SEARCH_RESULT, "papers": [paper or PAPER]}
    state = create_state("下载论文")
    state["messages"].append(
        {
            "role": "tool",
            "tool_call_id": "test-search-call",
            "name": "search_paper",
            "content": result,
        }
    )
    return state


def state_with_library_result():
    state = create_state("下载已保存的论文")
    state["messages"].append(
        {
            "role": "tool",
            "tool_call_id": "test-list-call",
            "name": "list_library",
            "content": {
                "count": 1,
                "total": 1,
                "truncated": False,
                "papers": [
                    {
                        "id": PAPER["candidate_id"],
                        "title": PAPER["title"],
                        "authors": PAPER["authors"],
                        "source": PAPER["source"],
                        "published": PAPER["published"],
                        "paper_url": PAPER["paper_url"],
                        "pdf_url": PAPER["pdf_url"],
                        "saved_at": "2026-09-11T00:00:00+00:00",
                    }
                ],
                "library_path": "/test/library.json",
            },
        }
    )
    return state


class DownloadPaperTests(unittest.TestCase):
    def test_downloads_trusted_pdf_and_reuses_valid_cache(self):
        state = state_with_search_result()

        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            with (
                patch.dict(
                    os.environ,
                    {"PAPER_CACHE_DIR": str(cache_dir)},
                    clear=False,
                ),
                patch(
                    "tools.download.urlopen",
                    return_value=FakeResponse(),
                ) as urlopen,
            ):
                first = execute_tool(
                    "download_paper",
                    {"paper_id": "arxiv:1234.56789"},
                    state=state,
                )
                second = execute_tool(
                    "download_paper",
                    {"paper_id": "arxiv:1234.56789"},
                    state=state,
                )

            cached_file = Path(first["local_path"])
            self.assertEqual(cached_file.read_bytes(), PDF_BYTES)
            self.assertEqual(cached_file.parent, cache_dir.resolve())
            self.assertFalse(
                cached_file.with_suffix(f"{cached_file.suffix}.tmp").exists()
            )
            self.assertTrue(first["downloaded"])
            self.assertFalse(first["cached"])
            self.assertFalse(second["downloaded"])
            self.assertTrue(second["cached"])
            self.assertEqual(first["local_path"], second["local_path"])
            self.assertEqual(urlopen.call_count, 1)

    def test_runtime_rejects_paper_outside_trusted_results(self):
        state = state_with_search_result()

        with (
            TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": directory},
                clear=False,
            ),
            patch("tools.download.urlopen") as urlopen,
        ):
            result = execute_tool(
                "download_paper",
                {"paper_id": "arxiv:not-in-results"},
                state=state,
            )
            copied_url = execute_tool(
                "download_paper",
                {
                    "paper_id": "arxiv:1234.56789",
                    "pdf_url": PAPER["pdf_url"],
                },
                state=state,
            )

        self.assertEqual(result["error"]["type"], "unknown_paper")
        self.assertEqual(
            copied_url["error"]["type"],
            "invalid_arguments",
        )
        urlopen.assert_not_called()

    def test_download_accepts_trusted_list_library_record(self):
        state = state_with_library_result()

        with TemporaryDirectory() as directory:
            with (
                patch.dict(
                    os.environ,
                    {"PAPER_CACHE_DIR": directory},
                    clear=False,
                ),
                patch(
                    "tools.download.urlopen",
                    return_value=FakeResponse(),
                ),
            ):
                result = execute_tool(
                    "download_paper",
                    {"paper_id": "arxiv:1234.56789"},
                    state=state,
                )

            self.assertTrue(result["downloaded"])
            self.assertEqual(result["paper_id"], "arxiv:1234.56789")
            self.assertTrue(Path(result["local_path"]).is_file())

    def test_rejects_unsafe_url_without_network_or_cache_file(self):
        unsafe_paper = {
            **PAPER,
            "pdf_url": "http://127.0.0.1/private.pdf",
        }
        state = state_with_search_result(unsafe_paper)

        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            with (
                patch.dict(
                    os.environ,
                    {"PAPER_CACHE_DIR": str(cache_dir)},
                    clear=False,
                ),
                patch("tools.download.urlopen") as urlopen,
            ):
                result = execute_tool(
                    "download_paper",
                    {"paper_id": "arxiv:1234.56789"},
                    state=state,
                )

            self.assertEqual(
                result["error"]["type"],
                "tool_execution_error",
            )
            self.assertIn("must use HTTPS", result["error"]["message"])
            self.assertFalse(cache_dir.exists())
            urlopen.assert_not_called()

    def test_rejects_non_pdf_response_without_cache_file(self):
        state = state_with_search_result()

        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            response = FakeResponse(
                data=b"<html>not a PDF</html>",
                headers={"Content-Type": "text/html"},
            )
            with (
                patch.dict(
                    os.environ,
                    {"PAPER_CACHE_DIR": str(cache_dir)},
                    clear=False,
                ),
                patch("tools.download.urlopen", return_value=response),
            ):
                result = execute_tool(
                    "download_paper",
                    {"paper_id": "arxiv:1234.56789"},
                    state=state,
                )

            self.assertEqual(
                result["error"]["type"],
                "tool_execution_error",
            )
            self.assertIn("Content-Type", result["error"]["message"])
            self.assertFalse(cache_dir.exists())

    def test_rejects_declared_oversized_pdf_without_cache_file(self):
        state = state_with_search_result()

        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            response = FakeResponse(
                headers={
                    "Content-Type": "application/pdf",
                    "Content-Length": str(MAX_PDF_SIZE_BYTES + 1),
                }
            )
            with (
                patch.dict(
                    os.environ,
                    {"PAPER_CACHE_DIR": str(cache_dir)},
                    clear=False,
                ),
                patch("tools.download.urlopen", return_value=response),
            ):
                result = execute_tool(
                    "download_paper",
                    {"paper_id": "arxiv:1234.56789"},
                    state=state,
                )

            self.assertEqual(
                result["error"]["type"],
                "tool_execution_error",
            )
            self.assertIn("size limit", result["error"]["message"])
            self.assertFalse(cache_dir.exists())

    @patch.dict(os.environ, {"LLM_PROVIDER": "fake"}, clear=False)
    def test_fake_agent_can_search_then_download(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            with (
                patch.dict(
                    os.environ,
                    {"PAPER_CACHE_DIR": str(cache_dir)},
                    clear=False,
                ),
                patch.dict(
                    "runtime.TOOL_REGISTRY",
                    {
                        "search_paper": lambda query: SEARCH_RESULT,
                        "download_paper": download_paper,
                    },
                    clear=True,
                ),
                patch(
                    "tools.download.urlopen",
                    return_value=FakeResponse(),
                ),
            ):
                state = create_state("请搜索并下载 TASA 的 PDF")
                answer = run_agent(state)

            self.assertEqual(state["step"], 3)
            self.assertEqual(
                state["messages"][3]["tool_call"]["name"],
                "download_paper",
            )
            self.assertTrue(state["messages"][4]["content"]["downloaded"])
            self.assertIn("PDF 已下载", answer)
            self.assertEqual(len(list(cache_dir.glob("*.pdf"))), 1)


if __name__ == "__main__":
    unittest.main()
