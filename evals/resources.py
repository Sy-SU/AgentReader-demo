"""Reproducible resource measurements for full-document retrieval."""

import argparse
import json
import os
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Iterator

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from tools.index import MAX_INDEX_CHARS, MAX_INDEX_PAGES
from tools.retrieval import (
    MAX_TOP_K,
    chunk_paper_index,
    retrieve_paper_chunks,
)


RESOURCE_BENCHMARK_VERSION = 1
DEFAULT_REPETITIONS = 5
DEFAULT_PAGE_COUNT = 12
DEFAULT_PAGE_CHARS = 3_000
DEFAULT_TOP_K = 3
BENCHMARK_PAPER_ID = "benchmark:retrieval-resources-v1"
BENCHMARK_QUERY = "retrieval benchmark evidence"


def measure_retrieval_resources(
    repetitions: int = DEFAULT_REPETITIONS,
    page_count: int = DEFAULT_PAGE_COUNT,
    page_chars: int = DEFAULT_PAGE_CHARS,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Measure first retrieval, cached retrieval, disk, and payload sizes."""
    _validate_settings(
        repetitions=repetitions,
        page_count=page_count,
        page_chars=page_chars,
        top_k=top_k,
    )

    samples = [
        _measure_once(
            page_count=page_count,
            page_chars=page_chars,
            top_k=top_k,
        )
        for _ in range(repetitions)
    ]
    stable_fields = (
        "pdf_bytes",
        "indexed_text_chars",
        "index_file_bytes",
        "chunk_count",
        "returned_chunks",
        "matched_text_chars",
        "tool_result_json_chars",
        "tool_result_utf8_bytes",
    )
    for field in stable_fields:
        if len({sample[field] for sample in samples}) != 1:
            raise RuntimeError(
                f"Resource benchmark produced unstable {field!r}."
            )

    representative = samples[0]
    payload_bytes = representative["tool_result_utf8_bytes"]
    index_bytes = representative["index_file_bytes"]
    return {
        "benchmark_version": RESOURCE_BENCHMARK_VERSION,
        "description": (
            "Synthetic PDF retrieval resource benchmark; timing includes "
            "hashing, index load/build, chunking, and TF-IDF ranking."
        ),
        "fixture": {
            "paper_id": BENCHMARK_PAPER_ID,
            "page_count": page_count,
            "requested_page_chars": page_chars,
            "pdf_bytes": representative["pdf_bytes"],
            "indexed_text_chars": representative["indexed_text_chars"],
            "index_file_bytes": index_bytes,
            "chunk_count": representative["chunk_count"],
        },
        "timing_ms": {
            "repetitions": repetitions,
            "first_retrieval": _timing_summary(
                sample["first_retrieval_ms"] for sample in samples
            ),
            "cached_retrieval": _timing_summary(
                sample["cached_retrieval_ms"] for sample in samples
            ),
        },
        "context_payload": {
            "definition": (
                "UTF-8 JSON for one retrieve_paper_chunks Tool Result; "
                "excludes instructions and previous messages."
            ),
            "top_k": top_k,
            "returned_chunks": representative["returned_chunks"],
            "matched_text_chars": representative["matched_text_chars"],
            "json_chars": representative["tool_result_json_chars"],
            "utf8_bytes": payload_bytes,
            "share_of_index": round(payload_bytes / index_bytes, 6),
            "index_to_payload_ratio": round(index_bytes / payload_bytes, 3),
        },
    }


def format_resource_report(result: dict) -> str:
    """Format a compact terminal report for a resource measurement."""
    fixture = result["fixture"]
    timing = result["timing_ms"]
    context = result["context_payload"]
    first = timing["first_retrieval"]
    cached = timing["cached_retrieval"]
    return "\n".join(
        [
            "=== Retrieval Resource Measurement ===",
            (
                f"Fixture: {fixture['page_count']} pages | "
                f"{fixture['indexed_text_chars']} indexed chars | "
                f"{fixture['chunk_count']} chunks"
            ),
            (
                "First retrieval: "
                f"median {first['median']:.3f} ms "
                f"(min {first['min']:.3f}, max {first['max']:.3f})"
            ),
            (
                "Cached retrieval: "
                f"median {cached['median']:.3f} ms "
                f"(min {cached['min']:.3f}, max {cached['max']:.3f})"
            ),
            f"PDF: {_format_bytes(fixture['pdf_bytes'])}",
            f"Index: {_format_bytes(fixture['index_file_bytes'])}",
            (
                "Tool Result payload: "
                f"{_format_bytes(context['utf8_bytes'])} | "
                f"{context['returned_chunks']} chunks | "
                f"{context['share_of_index']:.1%} of index size"
            ),
            "Timing is machine-dependent; payload excludes prior Context.",
        ]
    )


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure bounded retrieval resource usage offline."
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Number of isolated build/cache measurements (default: 5).",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=DEFAULT_PAGE_COUNT,
        help="Number of pages in the generated PDF (default: 12).",
    )
    parser.add_argument(
        "--page-chars",
        type=int,
        default=DEFAULT_PAGE_CHARS,
        help="Approximate source characters per page (default: 3000).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks returned per retrieval (default: 3).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete machine-readable measurement.",
    )
    args = parser.parse_args(argv)

    try:
        result = measure_retrieval_resources(
            repetitions=args.repetitions,
            page_count=args.pages,
            page_chars=args.page_chars,
            top_k=args.top_k,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Resource measurement failed: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_resource_report(result))
    return 0


def _measure_once(page_count: int, page_chars: int, top_k: int) -> dict:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        pdf_dir = root / "pdfs"
        index_dir = root / "indexes"
        pdf_dir.mkdir()
        pdf_path = pdf_dir / "benchmark.pdf"
        _write_fixture_pdf(
            pdf_path,
            page_count=page_count,
            page_chars=page_chars,
        )
        download_result = {
            "downloaded": True,
            "cached": True,
            "paper_id": BENCHMARK_PAPER_ID,
            "title": "Synthetic Retrieval Resource Benchmark",
            "pdf_url": "https://example.com/benchmark.pdf",
            "local_path": str(pdf_path),
            "size_bytes": pdf_path.stat().st_size,
        }

        with _cache_environment(pdf_dir=pdf_dir, index_dir=index_dir):
            first_started = perf_counter()
            first = retrieve_paper_chunks(
                download_result,
                query=BENCHMARK_QUERY,
                top_k=top_k,
            )
            first_retrieval_ms = (perf_counter() - first_started) * 1_000

            cached_started = perf_counter()
            cached = retrieve_paper_chunks(
                download_result,
                query=BENCHMARK_QUERY,
                top_k=top_k,
            )
            cached_retrieval_ms = (perf_counter() - cached_started) * 1_000

        if first["index_status"] != "built":
            raise RuntimeError("First retrieval did not build an index.")
        if cached["index_status"] != "cached":
            raise RuntimeError("Second retrieval did not reuse the index.")

        index_paths = list(index_dir.glob("*.json"))
        if len(index_paths) != 1:
            raise RuntimeError("Benchmark expected exactly one index file.")
        index_path = index_paths[0]
        index_document = json.loads(index_path.read_text(encoding="utf-8"))
        chunk_result = chunk_paper_index(index_document)
        serialized_result = json.dumps(cached, ensure_ascii=False)

        return {
            "first_retrieval_ms": first_retrieval_ms,
            "cached_retrieval_ms": cached_retrieval_ms,
            "pdf_bytes": pdf_path.stat().st_size,
            "indexed_text_chars": index_document["char_count"],
            "index_file_bytes": index_path.stat().st_size,
            "chunk_count": chunk_result["chunk_count"],
            "returned_chunks": cached["count"],
            "matched_text_chars": sum(
                len(match["text"]) for match in cached["matches"]
            ),
            "tool_result_json_chars": len(serialized_result),
            "tool_result_utf8_bytes": len(serialized_result.encode("utf-8")),
        }


def _write_fixture_pdf(
    path: Path,
    page_count: int,
    page_chars: int,
) -> None:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)

    for page_number in range(1, page_count + 1):
        seed = (
            f"Page {page_number} retrieval benchmark evidence discusses "
            "attention architecture experiments and reproducible ranking. "
        )
        page_text = (seed * (page_chars // len(seed) + 1))[:page_chars]
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        escaped_text = (
            page_text.replace("\\", "\\\\")
            .replace("(", "\\(")
            .replace(")", "\\)")
        )
        content = DecodedStreamObject()
        content.set_data(
            f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode(
                "latin-1"
            )
        )
        page[NameObject("/Contents")] = writer._add_object(content)

    with path.open("wb") as pdf_file:
        writer.write(pdf_file)


@contextmanager
def _cache_environment(pdf_dir: Path, index_dir: Path) -> Iterator[None]:
    updates = {
        "PAPER_CACHE_DIR": str(pdf_dir),
        "PAPER_INDEX_DIR": str(index_dir),
    }
    previous = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _validate_settings(
    repetitions: int,
    page_count: int,
    page_chars: int,
    top_k: int,
) -> None:
    integer_settings = {
        "repetitions": repetitions,
        "page_count": page_count,
        "page_chars": page_chars,
        "top_k": top_k,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in integer_settings.values()
    ):
        raise ValueError("Resource measurement settings must be integers.")
    if not 1 <= repetitions <= 20:
        raise ValueError("repetitions must be from 1 to 20.")
    if not 1 <= page_count <= MAX_INDEX_PAGES:
        raise ValueError(
            f"page_count must be from 1 to {MAX_INDEX_PAGES}."
        )
    if not 200 <= page_chars <= 20_000:
        raise ValueError("page_chars must be from 200 to 20000.")
    if page_count * page_chars > MAX_INDEX_CHARS:
        raise ValueError("Generated text would exceed the index char limit.")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be from 1 to {MAX_TOP_K}.")


def _timing_summary(values: Iterator[float]) -> dict:
    samples = list(values)
    return {
        "median": round(statistics.median(samples), 3),
        "min": round(min(samples), 3),
        "max": round(max(samples), 3),
        "samples": [round(sample, 3) for sample in samples],
    }


def _format_bytes(value: int) -> str:
    if value < 1_024:
        return f"{value} B"
    return f"{value / 1_024:.1f} KiB"
