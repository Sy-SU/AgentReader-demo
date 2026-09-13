"""Run a lightweight same-corpus LitSearch retrieval baseline.

This module deliberately lives under ``evals``.  It does not replace the
production arXiv/Crossref ``search_paper`` tool.  Instead, it indexes an
export of LitSearch's fixed title/abstract corpus with SQLite FTS5 and writes
ranked Semantic Scholar corpus IDs in the official result shape.

The retriever is an AgentReader baseline, not the official LitSearch BM25
implementation.  The official code uses NLTK and ``rank_bm25``; this adapter
uses only Python's bundled SQLite FTS5 with the Porter tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from evals.litsearch import (
    evaluate_litsearch,
    format_litsearch_report,
    load_litsearch_queries,
    normalize_litsearch_queries,
)


LITSEARCH_BASELINE_VERSION = 1
RETRIEVAL_METHOD = "agentreader_sqlite_fts5_porter_v1"
DEFAULT_TOP_K = 20
MIN_TOP_K = 20
MAX_TOP_K = 200
MAX_CORPUS_RECORDS = 100_000
MAX_QUERY_RECORDS = 1_000
MAX_CORPUS_FILE_BYTES = 512 * 1024 * 1024
MAX_TITLE_CHARS = 20_000
MAX_ABSTRACT_CHARS = 200_000
MAX_QUERY_TOKENS = 128
_QUERY_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def load_litsearch_corpus(path: str | Path) -> list[dict]:
    """Load a projected corpusid/title/abstract JSON or JSONL export."""
    corpus_path = Path(path)
    try:
        file_size = corpus_path.stat().st_size
    except OSError:
        raise
    if file_size > MAX_CORPUS_FILE_BYTES:
        raise ValueError(
            "LitSearch corpus export is too large for the lightweight "
            "adapter. Export only corpusid, title, and abstract."
        )

    suffix = corpus_path.suffix.casefold()
    if suffix == ".jsonl":
        raw_records = _read_jsonl(corpus_path)
    elif suffix == ".json":
        try:
            payload = json.loads(corpus_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"LitSearch corpus JSON is invalid: {error.msg}."
            ) from error
        raw_records = _records_from_payload(payload)
    else:
        raise ValueError("LitSearch corpus must use .json or .jsonl.")

    records = []
    seen_corpusids = set()
    for index, raw_record in enumerate(raw_records):
        if index >= MAX_CORPUS_RECORDS:
            raise ValueError(
                f"LitSearch corpus exceeds {MAX_CORPUS_RECORDS} records."
            )
        record = _normalize_corpus_record(raw_record, index)
        corpusid = record["corpusid"]
        if corpusid in seen_corpusids:
            raise ValueError(
                f"LitSearch corpus repeats corpusid {corpusid}."
            )
        seen_corpusids.add(corpusid)
        records.append(record)

    if not records:
        raise ValueError("LitSearch corpus must contain at least one record.")
    return records


def run_litsearch_baseline(
    corpus: list[dict],
    queries: list[dict],
    *,
    top_k: int = DEFAULT_TOP_K,
    limit: int | None = None,
) -> dict:
    """Retrieve over one fixed corpus and score the generated IDs."""
    normalized_corpus = _validate_corpus_records(corpus)
    normalized_queries = _select_queries(queries, limit)
    normalized_top_k = _validate_top_k(top_k)

    connection = sqlite3.connect(":memory:")
    try:
        index_started = perf_counter()
        _build_index(connection, normalized_corpus)
        index_duration_ms = (perf_counter() - index_started) * 1000

        retrieval_started = perf_counter()
        fallback_corpusids = [
            record["corpusid"] for record in normalized_corpus
        ]
        results = [
            {
                **query,
                "retrieved": _retrieve(
                    connection,
                    query["query"],
                    normalized_top_k,
                    fallback_corpusids,
                ),
            }
            for query in normalized_queries
        ]
        retrieval_duration_ms = (perf_counter() - retrieval_started) * 1000
    finally:
        connection.close()

    return {
        "baseline_version": LITSEARCH_BASELINE_VERSION,
        "benchmark": "LitSearch",
        "retrieval_method": RETRIEVAL_METHOD,
        "official_baseline_compatible": False,
        "corpus_document_count": len(normalized_corpus),
        "query_count": len(normalized_queries),
        "top_k": normalized_top_k,
        "timings_ms": {
            "index": round(index_duration_ms, 1),
            "retrieval": round(retrieval_duration_ms, 1),
        },
        "evaluation": evaluate_litsearch(results),
        "retrieval_results": results,
    }


def write_litsearch_results(records: list[dict], path: str | Path) -> Path:
    """Atomically write official-shape retrieval records as JSON or JSONL."""
    output_path = Path(path)
    suffix = output_path.suffix.casefold()
    if suffix not in {".json", ".jsonl"}:
        raise ValueError("LitSearch output must use .json or .jsonl.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            if suffix == ".json":
                json.dump(records, temporary_file, ensure_ascii=False, indent=2)
                temporary_file.write("\n")
            else:
                for record in records:
                    temporary_file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def format_litsearch_baseline_report(report: dict) -> str:
    """Format retriever identity, timings, and the existing metric report."""
    return "\n".join(
        [
            "=== AgentReader LitSearch Baseline ===",
            f"Retriever: {report['retrieval_method']}",
            "Comparable to official retriever numbers: no",
            f"Corpus documents: {report['corpus_document_count']}",
            f"Queries: {report['query_count']}",
            f"Top K generated: {report['top_k']}",
            (
                "Timing: index "
                f"{report['timings_ms']['index']:.1f} ms | retrieval "
                f"{report['timings_ms']['retrieval']:.1f} ms"
            ),
            "",
            format_litsearch_report(report["evaluation"]),
        ]
    )


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run AgentReader's lightweight SQLite FTS5 baseline on exported "
            "LitSearch query and title/abstract JSON/JSONL files."
        )
    )
    parser.add_argument("corpus", type=Path, help="Projected LitSearch corpus.")
    parser.add_argument("queries", type=Path, help="Official LitSearch queries.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=(
            f"Retrieve {MIN_TOP_K}..{MAX_TOP_K} papers per query "
            "(default: 20)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N queries for a smoke test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally write official-shape .json or .jsonl results.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete machine-readable report.",
    )
    args = parser.parse_args(argv)

    try:
        corpus = load_litsearch_corpus(args.corpus)
        queries = load_litsearch_queries(args.queries)
        report = run_litsearch_baseline(
            corpus,
            queries,
            top_k=args.top_k,
            limit=args.limit,
        )
        if args.output is not None:
            write_litsearch_results(
                report["retrieval_results"],
                args.output,
            )
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"LitSearch baseline failed: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_litsearch_baseline_report(report))
        if args.output is not None:
            print(f"Results: {args.output}")
    return 0


def _read_jsonl(path: Path) -> list[object]:
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    "LitSearch corpus JSONL line "
                    f"{line_number} is invalid: {error.msg}."
                ) from error
    return records


def _records_from_payload(payload: object) -> list[object]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError(
        "LitSearch corpus JSON must be a record list or a dataset-viewer "
        "object with a rows list."
    )


def _normalize_corpus_record(record: object, index: int) -> dict:
    if isinstance(record, dict) and isinstance(record.get("row"), dict):
        record = record["row"]
    if not isinstance(record, dict):
        raise ValueError(f"LitSearch corpus record {index} must be an object.")

    corpusid = _positive_int(record.get("corpusid"))
    title = record.get("title")
    abstract = record.get("abstract")
    if corpusid is None:
        raise ValueError(
            f"LitSearch corpus record {index} needs a positive corpusid."
        )
    if title is None:
        title = ""
    if abstract is None:
        abstract = ""
    if not isinstance(title, str) or not isinstance(abstract, str):
        raise ValueError(
            f"LitSearch corpus record {index} title/abstract must be strings."
        )
    title = title.strip()
    abstract = abstract.strip()
    if len(title) > MAX_TITLE_CHARS or len(abstract) > MAX_ABSTRACT_CHARS:
        raise ValueError(
            f"LitSearch corpus record {index} exceeds text limits."
        )
    return {"corpusid": corpusid, "title": title, "abstract": abstract}


def _validate_corpus_records(corpus: object) -> list[dict]:
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("LitSearch corpus must be a non-empty list.")
    if len(corpus) > MAX_CORPUS_RECORDS:
        raise ValueError(
            f"LitSearch corpus exceeds {MAX_CORPUS_RECORDS} records."
        )
    normalized = []
    seen_corpusids = set()
    for index, record in enumerate(corpus):
        item = _normalize_corpus_record(record, index)
        if item["corpusid"] in seen_corpusids:
            raise ValueError(
                f"LitSearch corpus repeats corpusid {item['corpusid']}."
            )
        seen_corpusids.add(item["corpusid"])
        normalized.append(item)
    return normalized


def _select_queries(queries: object, limit: int | None) -> list[dict]:
    if not isinstance(queries, list) or not queries:
        raise ValueError("LitSearch queries must be a non-empty list.")
    if len(queries) > MAX_QUERY_RECORDS:
        raise ValueError(
            f"LitSearch queries exceed {MAX_QUERY_RECORDS} records."
        )
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("LitSearch query limit must be a positive integer.")
    normalized = normalize_litsearch_queries(queries)
    return normalized if limit is None else normalized[:limit]


def _validate_top_k(top_k: object) -> int:
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not MIN_TOP_K <= top_k <= MAX_TOP_K
    ):
        raise ValueError(
            "LitSearch top_k must be between "
            f"{MIN_TOP_K} and {MAX_TOP_K} so Recall@20 is valid."
        )
    return top_k


def _build_index(connection: sqlite3.Connection, corpus: list[dict]) -> None:
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE papers USING fts5(
                corpusid UNINDEXED,
                title,
                abstract,
                tokenize='porter unicode61'
            )
            """
        )
    except sqlite3.OperationalError as error:
        raise ValueError(
            "This Python SQLite build does not support FTS5."
        ) from error
    connection.executemany(
        "INSERT INTO papers(corpusid, title, abstract) VALUES (?, ?, ?)",
        (
            (record["corpusid"], record["title"], record["abstract"])
            for record in corpus
        ),
    )
    connection.commit()


def _retrieve(
    connection: sqlite3.Connection,
    query: str,
    top_k: int,
    fallback_corpusids: list[int],
) -> list[int]:
    match_query = _fts_match_query(query)
    if match_query is None:
        return fallback_corpusids[:top_k]
    rows = connection.execute(
        """
        SELECT CAST(corpusid AS INTEGER) AS corpusid,
               bm25(papers, 0.0, 1.0, 1.0) AS score
        FROM papers
        WHERE papers MATCH ?
        ORDER BY score ASC, rowid ASC
        LIMIT ?
        """,
        (match_query, top_k),
    )
    retrieved = [row[0] for row in rows]
    if len(retrieved) < top_k:
        retrieved_set = set(retrieved)
        for corpusid in fallback_corpusids:
            if corpusid in retrieved_set:
                continue
            retrieved.append(corpusid)
            retrieved_set.add(corpusid)
            if len(retrieved) == top_k:
                break
    return retrieved


def _fts_match_query(query: object) -> str | None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("LitSearch retrieval query must be non-empty.")
    tokens = list(
        dict.fromkeys(_QUERY_TOKEN_PATTERN.findall(query.casefold()))
    )[:MAX_QUERY_TOKENS]
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens)


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None
