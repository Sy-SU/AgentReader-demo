"""Persistent, bounded full-document text indexes for cached PDFs."""

import hashlib
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .download import validate_cached_pdf_path
from .extract import normalize_extracted_page_text


INDEX_FORMAT_VERSION = 1
MAX_INDEX_PAGES = 200
MAX_INDEX_CHARS = 2_000_000
MAX_INDEX_FILE_BYTES = 32 * 1024 * 1024
DEFAULT_INDEX_DIR = Path(__file__).resolve().parent.parent / "data/indexes"


def load_or_build_paper_index(download_result: dict) -> dict:
    """Load a current index or atomically rebuild it from a trusted PDF."""
    if not isinstance(download_result, dict):
        raise ValueError("download_result must be a dictionary.")

    paper_id = download_result.get("paper_id")
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("download_result has no valid paper_id.")
    paper_id = paper_id.strip()

    pdf_path = validate_cached_pdf_path(download_result.get("local_path"))
    pdf_size_bytes = pdf_path.stat().st_size
    pdf_sha256 = _file_sha256(pdf_path)
    index_path = _index_path(paper_id)
    rebuild_reason = None

    if index_path.exists():
        try:
            cached_index = _read_index(index_path)
            validate_paper_index(cached_index)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            rebuild_reason = "invalid_index"
        else:
            if _matches_pdf(
                cached_index,
                paper_id=paper_id,
                pdf_sha256=pdf_sha256,
                pdf_size_bytes=pdf_size_bytes,
            ):
                return _with_runtime_metadata(
                    cached_index,
                    title=download_result.get("title"),
                    index_path=index_path,
                    index_status="cached",
                )
            rebuild_reason = "stale_index"

    index_document = _build_index_document(
        pdf_path=pdf_path,
        paper_id=paper_id,
        title=download_result.get("title"),
        pdf_sha256=pdf_sha256,
        pdf_size_bytes=pdf_size_bytes,
    )
    validate_paper_index(index_document)
    _write_index(index_path, index_document)
    index_status = "rebuilt" if rebuild_reason else "built"
    return _with_runtime_metadata(
        index_document,
        title=download_result.get("title"),
        index_path=index_path,
        index_status=index_status,
        rebuild_reason=rebuild_reason,
    )


def _build_index_document(
    pdf_path: Path,
    paper_id: str,
    title: object,
    pdf_sha256: str,
    pdf_size_bytes: int,
) -> dict:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PDF indexing requires pypdf. Run "
            "'conda env update -f environment.yml --prune'."
        ) from error

    reader = None
    try:
        reader = PdfReader(pdf_path, strict=False)
        if reader.is_encrypted:
            raise RuntimeError("Cached PDF is encrypted and cannot be indexed.")

        page_count = len(reader.pages)
        page_scan_limit = min(page_count, MAX_INDEX_PAGES)
        pages = []
        pages_scanned = 0
        char_count = 0
        character_limited = False
        last_page_partial = False

        for page_index in range(page_scan_limit):
            try:
                raw_text = reader.pages[page_index].extract_text() or ""
            except Exception as error:
                raise RuntimeError(
                    f"Could not index PDF page {page_index + 1}: {error}"
                ) from error
            pages_scanned += 1

            page_text = normalize_extracted_page_text(raw_text)
            if not page_text:
                continue

            remaining_chars = MAX_INDEX_CHARS - char_count
            if remaining_chars <= 0:
                character_limited = True
                break

            indexed_text = page_text[:remaining_chars]
            pages.append({"page": page_index + 1, "text": indexed_text})
            char_count += len(indexed_text)
            if len(indexed_text) < len(page_text):
                character_limited = True
                last_page_partial = True
                break
    except PdfReadError as error:
        raise RuntimeError(f"Could not parse cached PDF: {error}") from error
    except OSError as error:
        raise RuntimeError(f"Could not read cached PDF: {error}") from error
    finally:
        if reader is not None:
            reader.close()

    truncation_reasons = []
    if page_scan_limit < page_count:
        truncation_reasons.append("page_limit")
    if character_limited:
        truncation_reasons.append("character_limit")

    page_numbers = [page["page"] for page in pages]
    index_document = {
        "version": INDEX_FORMAT_VERSION,
        "paper_id": paper_id,
        "title": title if isinstance(title, str) else None,
        "pdf_sha256": pdf_sha256,
        "pdf_size_bytes": pdf_size_bytes,
        "max_pages": MAX_INDEX_PAGES,
        "max_chars": MAX_INDEX_CHARS,
        "page_count": page_count,
        "pages_scanned": pages_scanned,
        "pages_indexed": len(pages),
        "page_numbers": page_numbers,
        "last_page_partial": last_page_partial,
        "char_count": char_count,
        "text_available": bool(pages),
        "coverage_complete": not truncation_reasons,
        "truncated": bool(truncation_reasons),
        "truncation_reasons": truncation_reasons,
        "pages": pages,
    }
    if not pages:
        index_document["warning"] = (
            "No extractable text was found while indexing the PDF. "
            "The PDF may contain scanned images and OCR is not implemented."
        )
    return index_document


def validate_paper_index(index_document: object) -> None:
    """Validate the persisted index contract before retrieval uses it."""
    if not isinstance(index_document, dict):
        raise ValueError("Paper index must be a JSON object.")
    if index_document.get("version") != INDEX_FORMAT_VERSION:
        raise ValueError("Paper index version is stale.")

    paper_id = index_document.get("paper_id")
    pdf_sha256 = index_document.get("pdf_sha256")
    pdf_size_bytes = index_document.get("pdf_size_bytes")
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("Paper index has no valid paper_id.")
    if not isinstance(pdf_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", pdf_sha256
    ):
        raise ValueError("Paper index has no valid PDF hash.")
    if (
        isinstance(pdf_size_bytes, bool)
        or not isinstance(pdf_size_bytes, int)
        or pdf_size_bytes < 0
    ):
        raise ValueError("Paper index has no valid PDF size.")
    if (
        index_document.get("max_pages") != MAX_INDEX_PAGES
        or index_document.get("max_chars") != MAX_INDEX_CHARS
    ):
        raise ValueError("Paper index extraction bounds are stale.")

    page_count = _non_negative_int(index_document.get("page_count"))
    pages_scanned = _non_negative_int(index_document.get("pages_scanned"))
    pages_indexed = _non_negative_int(index_document.get("pages_indexed"))
    char_count = _non_negative_int(index_document.get("char_count"))
    if pages_scanned > page_count or pages_indexed > pages_scanned:
        raise ValueError("Paper index page counters are inconsistent.")
    if pages_scanned > MAX_INDEX_PAGES or char_count > MAX_INDEX_CHARS:
        raise ValueError("Paper index exceeds configured bounds.")

    pages = index_document.get("pages")
    page_numbers = index_document.get("page_numbers")
    if not isinstance(pages, list) or not isinstance(page_numbers, list):
        raise ValueError("Paper index pages must be lists.")
    if len(pages) != pages_indexed or len(page_numbers) != pages_indexed:
        raise ValueError("Paper index page metadata is inconsistent.")

    parsed_page_numbers = []
    calculated_char_count = 0
    for page in pages:
        if not isinstance(page, dict) or set(page) != {"page", "text"}:
            raise ValueError("Each indexed page must contain page and text.")
        page_number = page["page"]
        text = page["text"]
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or not 1 <= page_number <= page_count
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError("Paper index contains an invalid page.")
        parsed_page_numbers.append(page_number)
        calculated_char_count += len(text)

    if parsed_page_numbers != sorted(set(parsed_page_numbers)):
        raise ValueError("Indexed page numbers must be strictly increasing.")
    if (
        page_numbers != parsed_page_numbers
        or char_count != calculated_char_count
    ):
        raise ValueError("Paper index text metadata is inconsistent.")

    reasons = index_document.get("truncation_reasons")
    if (
        not isinstance(reasons, list)
        or reasons != list(dict.fromkeys(reasons))
        or any(
            reason not in {"page_limit", "character_limit"}
            for reason in reasons
        )
        or not isinstance(index_document.get("truncated"), bool)
        or not isinstance(index_document.get("coverage_complete"), bool)
        or not isinstance(index_document.get("text_available"), bool)
        or bool(reasons) != bool(index_document.get("truncated"))
        or index_document.get("coverage_complete") != (not reasons)
        or index_document.get("text_available") != bool(pages)
        or not isinstance(index_document.get("last_page_partial"), bool)
        or (
            index_document.get("last_page_partial")
            and "character_limit" not in reasons
        )
    ):
        raise ValueError("Paper index coverage metadata is inconsistent.")


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Paper index counters must be non-negative integers.")
    return value


def _matches_pdf(
    index_document: dict,
    paper_id: str,
    pdf_sha256: str,
    pdf_size_bytes: int,
) -> bool:
    return (
        index_document["paper_id"] == paper_id
        and index_document["pdf_sha256"] == pdf_sha256
        and index_document["pdf_size_bytes"] == pdf_size_bytes
    )


def _read_index(index_path: Path) -> dict:
    index_root = _index_dir().resolve()
    if not index_path.resolve().is_relative_to(index_root):
        raise ValueError("Paper index path is outside PAPER_INDEX_DIR.")
    if not index_path.is_file():
        raise ValueError("Paper index path is not a regular file.")
    if index_path.stat().st_size > MAX_INDEX_FILE_BYTES:
        raise ValueError("Paper index exceeds the file size limit.")
    with index_path.open("r", encoding="utf-8") as index_file:
        return json.load(index_file)


def _write_index(index_path: Path, index_document: dict) -> None:
    temporary_path = None
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=index_path.parent,
            prefix=f".{index_path.stem}-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                index_document,
                temporary_file,
                ensure_ascii=False,
                indent=2,
            )
            temporary_file.write("\n")
        if temporary_path.stat().st_size > MAX_INDEX_FILE_BYTES:
            raise RuntimeError(
                "Generated paper index exceeds the file size limit."
            )
        temporary_path.replace(index_path)
    except OSError as error:
        raise RuntimeError(f"Could not write paper index: {error}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _file_sha256(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with pdf_path.open("rb") as pdf_file:
            for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RuntimeError(f"Could not hash cached PDF: {error}") from error
    return digest.hexdigest()


def _index_path(paper_id: str) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id)
    safe_stem = safe_stem.strip("._-")[:80] or "paper"
    digest = hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:8]
    return _index_dir() / f"{safe_stem}-{digest}.json"


def _index_dir() -> Path:
    configured_path = os.getenv("PAPER_INDEX_DIR")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_INDEX_DIR


def _with_runtime_metadata(
    index_document: dict,
    title: object,
    index_path: Path,
    index_status: str,
    rebuild_reason: str | None = None,
) -> dict:
    result: dict[str, Any] = dict(index_document)
    result["title"] = title if isinstance(title, str) else None
    result["index_path"] = str(index_path)
    result["index_status"] = index_status
    if rebuild_reason:
        result["index_rebuild_reason"] = rebuild_reason
    return result
