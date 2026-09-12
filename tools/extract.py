"""Bounded text extraction from a trusted cached PDF."""

import re

from .download import validate_cached_pdf_path


DEFAULT_MAX_PAGES = 5
MAX_EXTRACT_PAGES = 10
DEFAULT_MAX_CHARS = 12_000
MIN_EXTRACT_CHARS = 500
MAX_EXTRACT_CHARS = 20_000


def extract_paper_text(
    download_result: dict,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict:
    """Extract bounded text from one trusted download result."""
    if not isinstance(download_result, dict):
        raise ValueError("download_result must be a dictionary.")

    paper_id = download_result.get("paper_id")
    local_path = download_result.get("local_path")
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("download_result has no valid paper_id.")
    _validate_limits(max_pages=max_pages, max_chars=max_chars)
    cache_path = validate_cached_pdf_path(local_path)

    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PDF text extraction requires pypdf. Run "
            "'conda env update -f environment.yml --prune'."
        ) from error

    reader = None
    try:
        reader = PdfReader(cache_path, strict=False)
        if reader.is_encrypted:
            raise RuntimeError("Cached PDF is encrypted and cannot be read.")

        page_count = len(reader.pages)
        page_scan_limit = min(page_count, max_pages)
        pages_scanned = 0
        page_numbers = []
        page_sections = []
        text_length = 0
        character_limited = False
        last_page_partial = False
        for page_index in range(page_scan_limit):
            try:
                page_text = reader.pages[page_index].extract_text() or ""
            except Exception as error:
                raise RuntimeError(
                    f"Could not extract text from PDF page {page_index + 1}: "
                    f"{error}"
                ) from error
            pages_scanned += 1

            normalized_text = normalize_extracted_page_text(page_text)
            if not normalized_text:
                continue

            page_number = page_index + 1
            separator = "\n\n" if page_sections else ""
            header = f"[Page {page_number}]\n"
            remaining_chars = max_chars - text_length
            fixed_length = len(separator) + len(header)

            if remaining_chars <= fixed_length:
                character_limited = True
                break

            available_page_chars = remaining_chars - fixed_length
            returned_page_text = normalized_text[:available_page_chars]
            page_sections.append(
                f"{separator}{header}{returned_page_text}"
            )
            page_numbers.append(page_number)
            text_length += fixed_length + len(returned_page_text)

            if len(returned_page_text) < len(normalized_text):
                character_limited = True
                last_page_partial = True
                break

        text = "".join(page_sections).rstrip()
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

    result = {
        "paper_id": paper_id.strip(),
        "title": download_result.get("title"),
        "page_count": page_count,
        "pages_scanned": pages_scanned,
        "pages_read": len(page_numbers),
        "page_numbers": page_numbers,
        "last_page_partial": last_page_partial,
        "char_count": len(text),
        "text_available": bool(text),
        "truncated": bool(truncation_reasons),
        "truncation_reasons": truncation_reasons,
        "text": text,
    }
    if not text:
        result["warning"] = (
            "No extractable text was found in the selected pages. "
            "The PDF may contain scanned images and OCR is not implemented."
        )
    return result


def _validate_limits(max_pages: int, max_chars: int) -> None:
    if (
        isinstance(max_pages, bool)
        or not isinstance(max_pages, int)
        or not 1 <= max_pages <= MAX_EXTRACT_PAGES
    ):
        raise ValueError(
            f"max_pages must be an integer from 1 to {MAX_EXTRACT_PAGES}."
        )
    if (
        isinstance(max_chars, bool)
        or not isinstance(max_chars, int)
        or not MIN_EXTRACT_CHARS <= max_chars <= MAX_EXTRACT_CHARS
    ):
        raise ValueError(
            "max_chars must be an integer from "
            f"{MIN_EXTRACT_CHARS} to {MAX_EXTRACT_CHARS}."
        )


def normalize_extracted_page_text(value: str) -> str:
    """Normalize one PDF page and escape structural marker lookalikes."""
    normalized = value.replace("\x00", "").replace(
        "\r\n", "\n"
    ).replace("\r", "\n").strip()
    return re.sub(
        r"(?m)^\[Page \d+\]$",
        lambda match: f"\\{match.group(0)}",
        normalized,
    )
