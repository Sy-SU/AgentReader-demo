import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from tools.index import load_or_build_paper_index, validate_paper_index
from tools.retrieval import retrieve_paper_chunks


PAPER_ID = "arxiv:1234.56789"


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


def download_result(pdf_path: Path) -> dict:
    return {
        "downloaded": True,
        "cached": False,
        "paper_id": PAPER_ID,
        "title": "Index test paper",
        "pdf_url": "https://example.com/paper.pdf",
        "local_path": str(pdf_path),
        "size_bytes": pdf_path.stat().st_size,
    }


class PaperIndexTests(unittest.TestCase):
    def test_lazy_index_covers_page_after_old_ten_page_limit_and_reuses_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_dir = root / "pdfs"
            index_dir = root / "indexes"
            pdf_dir.mkdir()
            pdf_path = pdf_dir / "paper.pdf"
            page_texts = [
                f"ordinary content page {page}" for page in range(1, 12)
            ]
            page_texts.append("unique tail evidence on page twelve")
            write_text_pdf(pdf_path, page_texts)

            with patch.dict(
                os.environ,
                {
                    "PAPER_CACHE_DIR": str(pdf_dir),
                    "PAPER_INDEX_DIR": str(index_dir),
                },
                clear=False,
            ):
                first = retrieve_paper_chunks(
                    download_result(pdf_path),
                    query="unique tail evidence",
                    top_k=1,
                )
                with patch(
                    "tools.index._build_index_document",
                    side_effect=AssertionError("cache should be reused"),
                ):
                    second = retrieve_paper_chunks(
                        download_result(pdf_path),
                        query="unique tail evidence",
                        top_k=1,
                    )

            self.assertTrue(first["found"])
            self.assertEqual(first["matches"][0]["page"], 12)
            self.assertEqual(first["pages_read"], 12)
            self.assertTrue(first["coverage_complete"])
            self.assertFalse(first["source_truncated"])
            self.assertEqual(first["index_status"], "built")
            self.assertEqual(second["index_status"], "cached")

            index_paths = list(index_dir.glob("*.json"))
            self.assertEqual(len(index_paths), 1)
            with index_paths[0].open("r", encoding="utf-8") as index_file:
                persisted = json.load(index_file)
            validate_paper_index(persisted)
            self.assertNotIn("index_status", persisted)
            self.assertNotIn("index_path", persisted)

    def test_pdf_change_invalidates_and_rebuilds_index(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_dir = root / "pdfs"
            index_dir = root / "indexes"
            pdf_dir.mkdir()
            pdf = pdf_dir / "paper.pdf"
            write_text_pdf(pdf, ["old evidence"])

            with patch.dict(
                os.environ,
                {
                    "PAPER_CACHE_DIR": str(pdf_dir),
                    "PAPER_INDEX_DIR": str(index_dir),
                },
                clear=False,
            ):
                first = load_or_build_paper_index(download_result(pdf))
                write_text_pdf(pdf, ["new replacement evidence"])
                second = load_or_build_paper_index(download_result(pdf))

            self.assertEqual(first["index_status"], "built")
            self.assertEqual(second["index_status"], "rebuilt")
            self.assertEqual(second["index_rebuild_reason"], "stale_index")
            self.assertNotEqual(first["pdf_sha256"], second["pdf_sha256"])
            self.assertEqual(
                second["pages"][0]["text"],
                "new replacement evidence",
            )

    def test_corrupt_index_is_rebuilt_atomically(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_dir = root / "pdfs"
            index_dir = root / "indexes"
            pdf_dir.mkdir()
            pdf = pdf_dir / "paper.pdf"
            write_text_pdf(pdf, ["recoverable evidence"])

            with patch.dict(
                os.environ,
                {
                    "PAPER_CACHE_DIR": str(pdf_dir),
                    "PAPER_INDEX_DIR": str(index_dir),
                },
                clear=False,
            ):
                first = load_or_build_paper_index(download_result(pdf))
                index_path = Path(first["index_path"])
                index_path.write_text("{broken", encoding="utf-8")
                rebuilt = load_or_build_paper_index(download_result(pdf))

            self.assertEqual(rebuilt["index_status"], "rebuilt")
            self.assertEqual(
                rebuilt["index_rebuild_reason"], "invalid_index"
            )
            with index_path.open("r", encoding="utf-8") as index_file:
                persisted = json.load(index_file)
            validate_paper_index(persisted)
            self.assertEqual(list(index_dir.glob("*.tmp")), [])

    def test_stale_index_contract_is_rebuilt(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_dir = root / "pdfs"
            index_dir = root / "indexes"
            pdf_dir.mkdir()
            pdf = pdf_dir / "paper.pdf"
            write_text_pdf(pdf, ["versioned evidence"])

            with patch.dict(
                os.environ,
                {
                    "PAPER_CACHE_DIR": str(pdf_dir),
                    "PAPER_INDEX_DIR": str(index_dir),
                },
                clear=False,
            ):
                first = load_or_build_paper_index(download_result(pdf))
                index_path = Path(first["index_path"])
                persisted = json.loads(index_path.read_text(encoding="utf-8"))
                persisted["version"] = 0
                index_path.write_text(
                    json.dumps(persisted),
                    encoding="utf-8",
                )
                rebuilt = load_or_build_paper_index(download_result(pdf))

            self.assertEqual(rebuilt["index_status"], "rebuilt")
            self.assertEqual(
                rebuilt["index_rebuild_reason"], "invalid_index"
            )
            validate_paper_index(rebuilt)

    def test_index_discloses_page_and_character_limits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_dir = root / "pdfs"
            index_dir = root / "indexes"
            pdf_dir.mkdir()
            page_limited_pdf = pdf_dir / "page-limited.pdf"
            char_limited_pdf = pdf_dir / "char-limited.pdf"
            write_text_pdf(page_limited_pdf, ["one", "two", "three"])
            write_text_pdf(char_limited_pdf, ["A" * 30])

            environment = {
                "PAPER_CACHE_DIR": str(pdf_dir),
                "PAPER_INDEX_DIR": str(index_dir),
            }
            with patch.dict(os.environ, environment, clear=False):
                with patch("tools.index.MAX_INDEX_PAGES", 2):
                    page_limited = load_or_build_paper_index(
                        {
                            **download_result(page_limited_pdf),
                            "paper_id": "paper:page-limit",
                        }
                    )
                with patch("tools.index.MAX_INDEX_CHARS", 10):
                    char_limited = load_or_build_paper_index(
                        {
                            **download_result(char_limited_pdf),
                            "paper_id": "paper:char-limit",
                        }
                    )

            self.assertEqual(page_limited["page_numbers"], [1, 2])
            self.assertEqual(
                page_limited["truncation_reasons"], ["page_limit"]
            )
            self.assertFalse(page_limited["coverage_complete"])
            self.assertEqual(char_limited["char_count"], 10)
            self.assertTrue(char_limited["last_page_partial"])
            self.assertEqual(
                char_limited["truncation_reasons"], ["character_limit"]
            )


if __name__ == "__main__":
    unittest.main()
