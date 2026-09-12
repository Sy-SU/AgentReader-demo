"""Paper identity and local JSON library persistence."""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_LIBRARY_PATH = (
    Path(__file__).resolve().parent.parent / "data/library.json"
)
DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 50


def save_paper(paper: dict) -> dict:
    """Save one paper to the local JSON library without duplicating it."""
    normalized_paper = _normalize_paper_for_library(paper)
    resolved_paper_id = paper_id(normalized_paper)
    library_path = _library_path()
    library = _load_library(library_path)

    for existing in library["papers"]:
        if existing.get("id") == resolved_paper_id:
            return {
                "saved": False,
                "reason": "already_exists",
                "paper_id": resolved_paper_id,
                "title": existing.get("title"),
                "library_path": str(library_path),
                "total": len(library["papers"]),
            }

    record = {
        "id": resolved_paper_id,
        **normalized_paper,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    library["papers"].append(record)
    _write_library(library_path, library)

    return {
        "saved": True,
        "paper_id": resolved_paper_id,
        "title": record["title"],
        "library_path": str(library_path),
        "total": len(library["papers"]),
    }


def list_library(limit: int = DEFAULT_LIST_LIMIT) -> dict:
    """Return a bounded, newest-first view of the local paper library."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer.")
    if not 1 <= limit <= MAX_LIST_LIMIT:
        raise ValueError(
            f"limit must be between 1 and {MAX_LIST_LIMIT}."
        )

    library_path = _library_path()
    library = _load_library(library_path)
    records = library["papers"]

    if not all(isinstance(record, dict) for record in records):
        raise RuntimeError("Library contains an invalid paper record.")

    selected_records = list(reversed(records))[:limit]
    papers = [_paper_summary(record) for record in selected_records]
    return {
        "count": len(papers),
        "total": len(records),
        "truncated": len(papers) < len(records),
        "papers": papers,
        "library_path": str(library_path),
    }


def paper_id(paper: dict) -> str:
    """Return the stable ID shared by search candidates and library records."""
    arxiv_id = paper.get("arxiv_id")
    if arxiv_id:
        base_arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
        return f"arxiv:{base_arxiv_id.lower()}"

    doi = paper.get("doi")
    if doi:
        return f"doi:{doi.lower()}"

    title = str(paper.get("title") or "").strip().lower()
    paper_url = str(paper.get("paper_url") or "").strip().lower()
    identity = f"{title}|{paper_url}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"paper:{digest}"


def _paper_summary(record: dict) -> dict:
    return {
        "id": record.get("id"),
        "title": record.get("title"),
        "authors": record.get("authors") or [],
        "source": record.get("source"),
        "published": record.get("published"),
        "paper_url": record.get("paper_url"),
        "pdf_url": record.get("pdf_url"),
        "saved_at": record.get("saved_at"),
    }


def _normalize_paper_for_library(paper: dict) -> dict:
    if not isinstance(paper, dict):
        raise ValueError("paper must be a dictionary.")

    title = paper.get("title")
    paper_url = paper.get("paper_url")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("paper.title must be a non-empty string.")
    if not isinstance(paper_url, str) or not paper_url.strip():
        raise ValueError("paper.paper_url must be a non-empty string.")

    authors = paper.get("authors") or []
    if not isinstance(authors, list) or not all(
        isinstance(author, str) for author in authors
    ):
        raise ValueError("paper.authors must be a list of strings.")

    optional_fields = (
        "source",
        "arxiv_id",
        "doi",
        "abstract",
        "published",
        "pdf_url",
    )
    for field in optional_fields:
        value = paper.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"paper.{field} must be a string or null.")

    return {
        "source": paper.get("source"),
        "arxiv_id": paper.get("arxiv_id"),
        "doi": paper.get("doi"),
        "title": _collapse_whitespace(title),
        "authors": [
            _collapse_whitespace(author)
            for author in authors
            if _collapse_whitespace(author)
        ],
        "abstract": paper.get("abstract"),
        "published": paper.get("published"),
        "paper_url": paper_url.strip(),
        "pdf_url": paper.get("pdf_url"),
    }


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _library_path() -> Path:
    configured_path = os.getenv("PAPER_LIBRARY_PATH")
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    return DEFAULT_LIBRARY_PATH


def _load_library(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "papers": []}

    try:
        library = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read library: {error}") from error

    if (
        not isinstance(library, dict)
        or library.get("version") != 1
        or not isinstance(library.get("papers"), list)
    ):
        raise RuntimeError("Library has an invalid structure.")
    return library


def _write_library(path: Path, library: dict) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(library, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError as error:
        raise RuntimeError(f"Could not write library: {error}") from error
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
