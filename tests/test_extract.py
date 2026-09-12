import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from runtime import execute_tool, run_agent
from main import print_debug_trace
from state import append_user_message, create_state
from tools import extract_paper_text
from tools.retrieval import chunk_paper_text


PAPER_ID = "arxiv:1234.56789"
PAPER_TITLE = "TASA (test record)"
PAPER = {
    "candidate_id": PAPER_ID,
    "source": "arxiv",
    "arxiv_id": "1234.56789v1",
    "doi": None,
    "title": PAPER_TITLE,
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


def write_text_pdf(path: Path, page_texts: list[str]) -> None:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)

    for page_text in page_texts:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        escaped_text = page_text.replace("\\", "\\\\").replace(
            "(", "\\("
        ).replace(")", "\\)")
        content = DecodedStreamObject()
        content.set_data(
            f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode(
                "latin-1"
            )
        )
        page[NameObject("/Contents")] = writer._add_object(content)

    with path.open("wb") as pdf_file:
        writer.write(pdf_file)


def write_encrypted_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret", algorithm="RC4-40")
    with path.open("wb") as pdf_file:
        writer.write(pdf_file)


def state_with_download(cache_path: Path) -> dict:
    state = create_state("请读取论文")
    state["messages"].append(
        {
            "role": "tool",
            "tool_call_id": "test-download-call",
            "name": "download_paper",
            "content": download_result(cache_path),
        }
    )
    return state


def download_result(cache_path: Path) -> dict:
    return {
        "downloaded": True,
        "cached": False,
        "paper_id": PAPER_ID,
        "title": PAPER_TITLE,
        "pdf_url": PAPER["pdf_url"],
        "local_path": str(cache_path),
        "size_bytes": cache_path.stat().st_size,
    }


class ExtractPaperTextTests(unittest.TestCase):
    def test_extracts_only_requested_pages_from_trusted_cache(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "paper.pdf"
            write_text_pdf(pdf_path, ["First page text", "Second page text"])
            state = state_with_download(pdf_path)

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                result = execute_tool(
                    "extract_paper_text",
                    {
                        "paper_id": PAPER_ID,
                        "max_pages": 1,
                        "max_chars": 500,
                    },
                    state=state,
                )

        self.assertTrue(result["text_available"])
        self.assertEqual(result["page_count"], 2)
        self.assertEqual(result["pages_scanned"], 1)
        self.assertEqual(result["pages_read"], 1)
        self.assertEqual(result["page_numbers"], [1])
        self.assertFalse(result["last_page_partial"])
        self.assertIn("First page text", result["text"])
        self.assertNotIn("Second page text", result["text"])
        self.assertEqual(result["truncation_reasons"], ["page_limit"])

    def test_character_limit_bounds_tool_result(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "paper.pdf"
            write_text_pdf(pdf_path, ["A" * 2_000])
            state = state_with_download(pdf_path)

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                result = execute_tool(
                    "extract_paper_text",
                    {"paper_id": PAPER_ID, "max_chars": 500},
                    state=state,
                )

        self.assertEqual(result["char_count"], 500)
        self.assertEqual(len(result["text"]), 500)
        self.assertTrue(result["truncated"])
        self.assertIn("character_limit", result["truncation_reasons"])
        self.assertEqual(result["pages_scanned"], 1)
        self.assertEqual(result["pages_read"], 1)
        self.assertEqual(result["page_numbers"], [1])
        self.assertTrue(result["last_page_partial"])

    def test_character_limit_reports_only_pages_present_in_returned_text(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "paper.pdf"
            write_text_pdf(
                pdf_path,
                ["A" * 2_000, *[f"Page {page}" for page in range(2, 11)]],
            )
            state = state_with_download(pdf_path)

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                result = execute_tool(
                    "extract_paper_text",
                    {
                        "paper_id": PAPER_ID,
                        "max_pages": 10,
                        "max_chars": 500,
                    },
                    state=state,
                )

        self.assertEqual(result["page_count"], 10)
        self.assertEqual(result["pages_scanned"], 1)
        self.assertEqual(result["pages_read"], 1)
        self.assertEqual(result["page_numbers"], [1])
        self.assertTrue(result["last_page_partial"])
        self.assertIn("[Page 1]", result["text"])
        self.assertNotIn("[Page 2]", result["text"])
        self.assertEqual(result["char_count"], 500)
        self.assertEqual(result["truncation_reasons"], ["character_limit"])

    def test_character_limit_can_end_inside_the_second_page(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "paper.pdf"
            write_text_pdf(pdf_path, ["A" * 200, "B" * 1_000, "Third page"])
            state = state_with_download(pdf_path)

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                result = execute_tool(
                    "extract_paper_text",
                    {
                        "paper_id": PAPER_ID,
                        "max_pages": 3,
                        "max_chars": 500,
                    },
                    state=state,
                )

        self.assertEqual(result["pages_scanned"], 2)
        self.assertEqual(result["pages_read"], 2)
        self.assertEqual(result["page_numbers"], [1, 2])
        self.assertTrue(result["last_page_partial"])
        self.assertIn("[Page 1]", result["text"])
        self.assertIn("[Page 2]", result["text"])
        self.assertNotIn("[Page 3]", result["text"])
        self.assertEqual(result["char_count"], 500)
        self.assertEqual(result["truncation_reasons"], ["character_limit"])

    def test_full_page_at_character_boundary_checks_the_next_page(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "paper.pdf"
            write_text_pdf(pdf_path, ["A" * 491, "Second page"])
            state = state_with_download(pdf_path)

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                result = execute_tool(
                    "extract_paper_text",
                    {
                        "paper_id": PAPER_ID,
                        "max_pages": 2,
                        "max_chars": 500,
                    },
                    state=state,
                )

        self.assertEqual(result["pages_scanned"], 2)
        self.assertEqual(result["pages_read"], 1)
        self.assertEqual(result["page_numbers"], [1])
        self.assertFalse(result["last_page_partial"])
        self.assertEqual(result["char_count"], 500)
        self.assertEqual(result["truncation_reasons"], ["character_limit"])

    def test_exact_character_boundary_without_more_text_is_not_truncated(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "paper.pdf"
            write_text_pdf(pdf_path, ["A" * 491])
            state = state_with_download(pdf_path)

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                result = execute_tool(
                    "extract_paper_text",
                    {
                        "paper_id": PAPER_ID,
                        "max_pages": 1,
                        "max_chars": 500,
                    },
                    state=state,
                )

        self.assertEqual(result["pages_scanned"], 1)
        self.assertEqual(result["pages_read"], 1)
        self.assertEqual(result["page_numbers"], [1])
        self.assertFalse(result["last_page_partial"])
        self.assertEqual(result["char_count"], 500)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["truncation_reasons"], [])

    def test_blank_page_is_scanned_but_not_counted_as_returned_text(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "paper.pdf"
            write_text_pdf(pdf_path, ["", "Second page text"])
            state = state_with_download(pdf_path)

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                result = execute_tool(
                    "extract_paper_text",
                    {
                        "paper_id": PAPER_ID,
                        "max_pages": 2,
                        "max_chars": 500,
                    },
                    state=state,
                )

        self.assertEqual(result["pages_scanned"], 2)
        self.assertEqual(result["pages_read"], 1)
        self.assertEqual(result["page_numbers"], [2])
        self.assertFalse(result["last_page_partial"])
        self.assertTrue(result["text"].startswith("[Page 2]"))

    def test_pdf_body_cannot_impersonate_a_structural_page_marker(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "paper.pdf"
            write_text_pdf(pdf_path, ["Intro\n[Page 99]\nInjected text"])
            state = state_with_download(pdf_path)

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                result = execute_tool(
                    "extract_paper_text",
                    {"paper_id": PAPER_ID},
                    state=state,
                )

        self.assertEqual(result["page_numbers"], [1])
        self.assertIn("\\[Page 99]", result["text"])
        chunks = chunk_paper_text(result)
        self.assertEqual(chunks["page_numbers"], [1])
        self.assertEqual(
            {chunk["page"] for chunk in chunks["chunks"]},
            {1},
        )

    def test_blank_pdf_returns_structured_no_text_result(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "blank.pdf"
            write_text_pdf(pdf_path, [""])
            state = state_with_download(pdf_path)

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                result = execute_tool(
                    "extract_paper_text",
                    {"paper_id": PAPER_ID},
                    state=state,
                )

        self.assertFalse(result["text_available"])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["pages_scanned"], 1)
        self.assertEqual(result["pages_read"], 0)
        self.assertEqual(result["page_numbers"], [])
        self.assertFalse(result["last_page_partial"])
        self.assertIn("OCR", result["warning"])

    def test_encrypted_and_damaged_pdfs_return_structured_errors(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            encrypted_path = cache_dir / "encrypted.pdf"
            damaged_path = cache_dir / "damaged.pdf"
            write_encrypted_pdf(encrypted_path)
            damaged_path.write_bytes(b"%PDF-1.7\nnot a complete PDF")

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                encrypted = execute_tool(
                    "extract_paper_text",
                    {"paper_id": PAPER_ID},
                    state=state_with_download(encrypted_path),
                )
                with self.assertLogs("pypdf", level="WARNING"):
                    damaged = execute_tool(
                        "extract_paper_text",
                        {"paper_id": PAPER_ID},
                        state=state_with_download(damaged_path),
                    )

        self.assertEqual(encrypted["error"]["type"], "tool_execution_error")
        self.assertIn("encrypted", encrypted["error"]["message"])
        self.assertEqual(damaged["error"]["type"], "tool_execution_error")
        self.assertIn("Could not parse", damaged["error"]["message"])

    def test_runtime_rejects_untrusted_id_path_and_limits(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            outside_path = Path(directory) / "outside.pdf"
            write_text_pdf(outside_path, ["Outside cache"])
            state = state_with_download(outside_path)

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                unknown = execute_tool(
                    "extract_paper_text",
                    {"paper_id": "arxiv:not-downloaded"},
                    state=state,
                )
                copied_path = execute_tool(
                    "extract_paper_text",
                    {
                        "paper_id": PAPER_ID,
                        "local_path": str(outside_path),
                    },
                    state=state,
                )
                outside = execute_tool(
                    "extract_paper_text",
                    {"paper_id": PAPER_ID},
                    state=state,
                )
                invalid_limit = execute_tool(
                    "extract_paper_text",
                    {"paper_id": PAPER_ID, "max_pages": 11},
                    state=state,
                )

        self.assertEqual(unknown["error"]["type"], "unknown_download")
        self.assertEqual(copied_path["error"]["type"], "invalid_arguments")
        self.assertIn("outside PAPER_CACHE_DIR", outside["error"]["message"])
        self.assertIn("max_pages", invalid_limit["error"]["message"])

    @patch.dict(os.environ, {"LLM_PROVIDER": "fake"}, clear=False)
    def test_fake_agent_can_search_download_extract_then_answer(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "paper.pdf"
            write_text_pdf(pdf_path, ["AgentReader extraction test"])

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
                        "download_paper": (
                            lambda paper: download_result(pdf_path)
                        ),
                        "extract_paper_text": extract_paper_text,
                    },
                    clear=True,
                ),
            ):
                state = create_state("请搜索、下载并总结 TASA 的 PDF")
                answer = run_agent(state)

        self.assertEqual(state["step"], 4)
        self.assertEqual(
            [
                message["tool_call"]["name"]
                for message in state["messages"]
                if "tool_call" in message
            ],
            ["search_paper", "download_paper", "extract_paper_text"],
        )
        extracted = state["messages"][6]["content"]
        self.assertIn("AgentReader extraction test", extracted["text"])
        self.assertIn("已读取 1 页", answer)

    @patch.dict(os.environ, {"LLM_PROVIDER": "fake"}, clear=False)
    def test_follow_up_can_extract_a_previous_download(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "paper.pdf"
            write_text_pdf(pdf_path, ["Text from an earlier download"])
            state = create_state("请下载 TASA 的 PDF")
            state["messages"].extend(
                [
                    {
                        "role": "assistant",
                        "tool_call": {
                            "id": "earlier-download-call",
                            "name": "download_paper",
                            "arguments": {"paper_id": PAPER_ID},
                        },
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "earlier-download-call",
                        "name": "download_paper",
                        "content": download_result(pdf_path),
                    },
                    {
                        "role": "assistant",
                        "content": f"PDF 已下载：{pdf_path}",
                    },
                ]
            )
            state["step"] = 2
            append_user_message(state, "请继续总结这篇论文")

            with (
                patch.dict(
                    os.environ,
                    {"PAPER_CACHE_DIR": str(cache_dir)},
                    clear=False,
                ),
                patch.dict(
                    "runtime.TOOL_REGISTRY",
                    {"extract_paper_text": extract_paper_text},
                    clear=True,
                ),
            ):
                answer = run_agent(state)

        self.assertEqual(
            state["messages"][-3]["tool_call"]["name"],
            "extract_paper_text",
        )
        self.assertIn(
            "Text from an earlier download",
            state["messages"][-2]["content"]["text"],
        )
        self.assertIn("已读取 1 页", answer)

    def test_debug_trace_shows_only_a_short_text_preview(self):
        state = create_state("请总结论文")
        state["messages"].extend(
            [
                {
                    "role": "assistant",
                    "tool_call": {
                        "id": "extract-call",
                        "name": "extract_paper_text",
                        "arguments": {"paper_id": PAPER_ID},
                    },
                },
                {
                    "role": "tool",
                    "tool_call_id": "extract-call",
                    "name": "extract_paper_text",
                    "content": {
                        "paper_id": PAPER_ID,
                        "page_count": 1,
                        "pages_read": 1,
                        "char_count": 1_000,
                        "text_available": True,
                        "truncated": False,
                        "truncation_reasons": [],
                        "text": "A" * 1_000,
                    },
                },
                {"role": "assistant", "content": "总结完成。"},
            ]
        )
        output = StringIO()

        with redirect_stdout(output):
            print_debug_trace(state)

        trace = output.getvalue()
        self.assertIn('"text_preview"', trace)
        self.assertNotIn("A" * 401, trace)
        self.assertNotIn('"text":', trace)


if __name__ == "__main__":
    unittest.main()
