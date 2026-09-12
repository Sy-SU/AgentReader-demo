"""Bounded PDF download and local cache handling."""

import hashlib
import ipaddress
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .library import paper_id


DEFAULT_PDF_CACHE_DIR = (
    Path(__file__).resolve().parent.parent / "data/pdfs"
)
PDF_DOWNLOAD_TIMEOUT_SECONDS = 20
MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024
PDF_MAGIC = b"%PDF-"
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/octet-stream",
    "binary/octet-stream",
}


def download_paper(paper: dict) -> dict:
    """Download one trusted paper PDF or return its valid cached file."""
    if not isinstance(paper, dict):
        raise ValueError("paper must be a dictionary.")

    pdf_url = paper.get("pdf_url")
    if not isinstance(pdf_url, str) or not pdf_url.strip():
        raise ValueError("The selected paper does not have a PDF URL.")
    pdf_url = pdf_url.strip()
    _validate_pdf_url(pdf_url)

    resolved_paper_id = _resolved_paper_id(paper)
    cache_path = _cache_path(resolved_paper_id)
    if cache_path.exists():
        size_bytes = _validate_cached_pdf(cache_path)
        return _download_result(
            paper=paper,
            resolved_paper_id=resolved_paper_id,
            pdf_url=pdf_url,
            cache_path=cache_path,
            size_bytes=size_bytes,
            downloaded=False,
        )

    pdf_data, final_url = _fetch_pdf(pdf_url)
    _validate_pdf_data(pdf_data)
    _write_pdf(cache_path, pdf_data)
    return _download_result(
        paper=paper,
        resolved_paper_id=resolved_paper_id,
        pdf_url=final_url,
        cache_path=cache_path,
        size_bytes=len(pdf_data),
        downloaded=True,
    )


def validate_cached_pdf_path(value: str) -> Path:
    """Return a validated PDF path contained by the configured cache."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Cached PDF path must be a non-empty string.")

    cache_root = _cache_dir().resolve()
    cache_path = Path(value).expanduser().resolve()
    if not cache_path.is_relative_to(cache_root):
        raise ValueError("Cached PDF path is outside PAPER_CACHE_DIR.")
    if cache_path.suffix.lower() != ".pdf":
        raise ValueError("Cached paper must be a PDF file.")

    _validate_cached_pdf(cache_path)
    return cache_path


def _fetch_pdf(pdf_url: str) -> tuple[bytes, str]:
    request = Request(
        pdf_url,
        headers={
            "Accept": "application/pdf",
            "User-Agent": "AgentReaderDemo/1.0",
        },
    )

    try:
        with urlopen(
            request,
            timeout=PDF_DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            final_url = response.geturl()
            _validate_pdf_url(final_url)
            _validate_response_headers(response.headers)
            pdf_data = response.read(MAX_PDF_SIZE_BYTES + 1)
    except HTTPError as error:
        raise RuntimeError(
            f"PDF server returned HTTP {error.code}."
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Could not download PDF: {error}") from error

    if len(pdf_data) > MAX_PDF_SIZE_BYTES:
        raise RuntimeError(
            f"PDF exceeds the {MAX_PDF_SIZE_BYTES}-byte size limit."
        )
    return pdf_data, final_url


def _validate_pdf_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("PDF URL must use HTTPS and include a hostname.")
    if parsed.username or parsed.password:
        raise ValueError("PDF URL must not contain credentials.")

    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("PDF URL must not target localhost.")

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("PDF URL must not target a private IP address.")


def _validate_response_headers(headers: object) -> None:
    content_type = str(headers.get("Content-Type", ""))
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type and media_type not in ALLOWED_CONTENT_TYPES:
        raise RuntimeError(
            f"PDF URL returned unsupported Content-Type '{media_type}'."
        )

    content_length = headers.get("Content-Length")
    if content_length is None:
        return
    try:
        declared_size = int(content_length)
    except (TypeError, ValueError):
        return
    if declared_size > MAX_PDF_SIZE_BYTES:
        raise RuntimeError(
            f"PDF exceeds the {MAX_PDF_SIZE_BYTES}-byte size limit."
        )


def _validate_pdf_data(pdf_data: bytes) -> None:
    if not pdf_data.startswith(PDF_MAGIC):
        raise RuntimeError("Downloaded content is not a valid PDF file.")


def _validate_cached_pdf(cache_path: Path) -> int:
    if not cache_path.is_file():
        raise RuntimeError("PDF cache path is not a regular file.")

    try:
        size_bytes = cache_path.stat().st_size
        with cache_path.open("rb") as cached_file:
            magic = cached_file.read(len(PDF_MAGIC))
    except OSError as error:
        raise RuntimeError(f"Could not read cached PDF: {error}") from error

    if size_bytes > MAX_PDF_SIZE_BYTES or magic != PDF_MAGIC:
        raise RuntimeError("Cached PDF failed validation.")
    return size_bytes


def _resolved_paper_id(paper: dict) -> str:
    known_id = paper.get("candidate_id") or paper.get("id")
    if isinstance(known_id, str) and known_id.strip():
        return known_id.strip()
    return paper_id(paper)


def _cache_path(resolved_paper_id: str) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved_paper_id)
    safe_stem = safe_stem.strip("._-")[:80] or "paper"
    digest = hashlib.sha256(
        resolved_paper_id.encode("utf-8")
    ).hexdigest()[:8]
    return _cache_dir() / f"{safe_stem}-{digest}.pdf"


def _cache_dir() -> Path:
    configured_path = os.getenv("PAPER_CACHE_DIR")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_PDF_CACHE_DIR


def _write_pdf(cache_path: Path, pdf_data: bytes) -> None:
    temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_bytes(pdf_data)
        temporary_path.replace(cache_path)
    except OSError as error:
        raise RuntimeError(f"Could not write PDF cache: {error}") from error
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _download_result(
    paper: dict,
    resolved_paper_id: str,
    pdf_url: str,
    cache_path: Path,
    size_bytes: int,
    downloaded: bool,
) -> dict:
    return {
        "downloaded": downloaded,
        "cached": not downloaded,
        "paper_id": resolved_paper_id,
        "title": paper.get("title"),
        "pdf_url": pdf_url,
        "local_path": str(cache_path),
        "size_bytes": size_bytes,
    }
