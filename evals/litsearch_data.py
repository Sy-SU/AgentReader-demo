"""Download and project the official LitSearch benchmark data.

The official ``corpus_clean`` Parquet files contain more columns than the
lightweight AgentReader baseline needs.  This module downloads a pinned
dataset revision and streams only corpus ID, title, and abstract into ignored
JSONL artifacts.  It intentionally depends on a separate benchmark Conda
environment rather than adding Hugging Face/PyArrow to the main application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from evals.litsearch import load_litsearch_queries
from evals.litsearch_baseline import load_litsearch_corpus


DATASET_NAME = "princeton-nlp/LitSearch"
DATASET_REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"
DATA_PREPARATION_VERSION = 1
EXPECTED_QUERY_COUNT = 597
EXPECTED_CORPUS_COUNT = 64_183
QUERY_PREFIX = "query/"
CORPUS_PREFIX = "corpus_clean/"
QUERY_COLUMNS = (
    "query",
    "query_set",
    "specificity",
    "quality",
    "corpusids",
)
CORPUS_COLUMNS = ("corpusid", "title", "abstract")


def prepare_official_litsearch_data(
    output_dir: str | Path,
    cache_dir: str | Path,
    *,
    revision: str = DATASET_REVISION,
    force: bool = False,
) -> dict:
    """Download pinned Parquet shards and export validated JSONL inputs."""
    try:
        from huggingface_hub import (
            hf_hub_download,
            list_repo_files,
            try_to_load_from_cache,
        )
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "LitSearch data preparation needs environment-benchmark.yml. "
            "Create it with `conda env create -f environment-benchmark.yml`."
        ) from error

    destination = Path(output_dir)
    cache = Path(cache_dir)
    query_path = destination / "queries.jsonl"
    corpus_path = destination / "corpus.jsonl"
    metadata_path = destination / "metadata.json"
    artifacts = (query_path, corpus_path, metadata_path)

    existing = [path for path in artifacts if path.exists()]
    if existing and not force:
        if len(existing) != len(artifacts):
            raise ValueError(
                "LitSearch output is incomplete; rerun with --force after "
                "checking the existing files."
            )
        return _reuse_prepared_data(
            query_path,
            corpus_path,
            metadata_path,
            revision,
        )

    destination.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    repository_files = list_repo_files(
        DATASET_NAME,
        repo_type="dataset",
        revision=revision,
    )
    query_files, corpus_files = _select_parquet_files(repository_files)
    source_cache_reused = all(
        isinstance(
            try_to_load_from_cache(
                DATASET_NAME,
                filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache,
            ),
            str,
        )
        for filename in query_files + corpus_files
    )

    def download(filename: str) -> Path:
        return Path(
            hf_hub_download(
                repo_id=DATASET_NAME,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache,
            )
        )

    local_query_files = [download(filename) for filename in query_files]
    local_corpus_files = [download(filename) for filename in corpus_files]
    source_resolution_duration_seconds = perf_counter() - started

    export_started = perf_counter()
    _write_validated_jsonl(
        _iter_parquet_rows(parquet, local_query_files, QUERY_COLUMNS),
        query_path,
        _project_query,
        load_litsearch_queries,
        EXPECTED_QUERY_COUNT,
    )
    _write_validated_jsonl(
        _iter_parquet_rows(parquet, local_corpus_files, CORPUS_COLUMNS),
        corpus_path,
        _project_corpus,
        load_litsearch_corpus,
        EXPECTED_CORPUS_COUNT,
    )
    export_duration_seconds = perf_counter() - export_started

    source_files = []
    for filename, local_path in zip(
        query_files + corpus_files,
        local_query_files + local_corpus_files,
        strict=True,
    ):
        source_files.append(
            {"path": filename, "size_bytes": local_path.stat().st_size}
        )

    report = {
        "data_preparation_version": DATA_PREPARATION_VERSION,
        "dataset": DATASET_NAME,
        "dataset_revision": revision,
        "query_count": EXPECTED_QUERY_COUNT,
        "corpus_count": EXPECTED_CORPUS_COUNT,
        "source_files": source_files,
        "source_size_bytes": sum(item["size_bytes"] for item in source_files),
        "source_cache_reused": source_cache_reused,
        "artifacts": {
            "queries": _artifact_metadata(query_path),
            "corpus": _artifact_metadata(corpus_path),
        },
        "timings_seconds": {
            "source_resolution": round(
                source_resolution_duration_seconds,
                3,
            ),
            "projection": round(export_duration_seconds, 3),
        },
        "reused": False,
    }
    _write_json_atomic(report, metadata_path)
    return report


def format_data_report(report: dict) -> str:
    """Render a compact preparation summary without cache internals."""
    action = "reused" if report["reused"] else "downloaded and projected"
    return "\n".join(
        [
            "=== LitSearch Official Data ===",
            f"Status: {action}",
            f"Dataset revision: {report['dataset_revision']}",
            f"Queries: {report['query_count']}",
            f"Corpus papers: {report['corpus_count']}",
            f"Source size: {report['source_size_bytes']} bytes",
            f"Query JSONL: {report['artifacts']['queries']['path']}",
            f"Corpus JSONL: {report['artifacts']['corpus']['path']}",
        ]
    )


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download a pinned LitSearch query/corpus_clean revision and "
            "export the columns needed by AgentReader's benchmark adapter."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/evals/litsearch"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/evals/huggingface-cache"),
    )
    parser.add_argument("--revision", default=DATASET_REVISION)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Atomically replace an existing projection after validation.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = prepare_official_litsearch_data(
            args.output_dir,
            args.cache_dir,
            revision=args.revision,
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"LitSearch data preparation failed: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_data_report(report))
    return 0


def _select_parquet_files(files: Iterable[str]) -> tuple[list[str], list[str]]:
    files = tuple(files)
    query_files = sorted(
        filename
        for filename in files
        if filename.startswith(QUERY_PREFIX) and filename.endswith(".parquet")
    )
    corpus_files = sorted(
        filename
        for filename in files
        if filename.startswith(CORPUS_PREFIX)
        and filename.endswith(".parquet")
    )
    if len(query_files) != 1 or len(corpus_files) != 6:
        raise ValueError(
            "Pinned LitSearch revision must contain one query and six "
            "corpus_clean Parquet shards."
        )
    return query_files, corpus_files


def _iter_parquet_rows(
    parquet: object,
    paths: Iterable[Path],
    columns: tuple[str, ...],
) -> Iterator[dict]:
    for path in paths:
        parquet_file = parquet.ParquetFile(path)
        missing = set(columns).difference(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(
                f"LitSearch Parquet {path.name} lacks columns: "
                f"{', '.join(sorted(missing))}."
            )
        for batch in parquet_file.iter_batches(
            batch_size=1_024,
            columns=list(columns),
        ):
            yield from batch.to_pylist()


def _project_query(record: object) -> dict:
    if not isinstance(record, dict):
        raise ValueError("LitSearch query row must be an object.")
    missing = set(QUERY_COLUMNS).difference(record)
    if missing:
        raise ValueError(
            f"LitSearch query row lacks fields: {', '.join(sorted(missing))}."
        )
    return {key: record[key] for key in QUERY_COLUMNS}


def _project_corpus(record: object) -> dict:
    if not isinstance(record, dict):
        raise ValueError("LitSearch corpus row must be an object.")
    missing = set(CORPUS_COLUMNS).difference(record)
    if missing:
        raise ValueError(
            f"LitSearch corpus row lacks fields: {', '.join(sorted(missing))}."
        )
    return {key: record[key] for key in CORPUS_COLUMNS}


def _write_validated_jsonl(
    records: Iterable[object],
    output_path: Path,
    projector: Callable[[object], dict],
    loader: Callable[[Path], list[dict]],
    expected_count: int,
) -> None:
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".jsonl",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            for record in records:
                temporary_file.write(
                    json.dumps(projector(record), ensure_ascii=False) + "\n"
                )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        normalized = loader(temporary_path)
        if len(normalized) != expected_count:
            raise ValueError(
                "LitSearch projection count mismatch: expected "
                f"{expected_count}, received {len(normalized)}."
            )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _artifact_metadata(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _reuse_prepared_data(
    query_path: Path,
    corpus_path: Path,
    metadata_path: Path,
    revision: str,
) -> dict:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Existing LitSearch metadata is invalid JSON.") from error
    if metadata.get("dataset_revision") != revision:
        raise ValueError(
            "Existing LitSearch projection uses another revision; use "
            "--force to replace it."
        )
    if len(load_litsearch_queries(query_path)) != EXPECTED_QUERY_COUNT:
        raise ValueError("Existing LitSearch query projection is incomplete.")
    if len(load_litsearch_corpus(corpus_path)) != EXPECTED_CORPUS_COUNT:
        raise ValueError("Existing LitSearch corpus projection is incomplete.")
    for key, path in (("queries", query_path), ("corpus", corpus_path)):
        actual = _artifact_metadata(path)
        expected = metadata.get("artifacts", {}).get(key, {})
        if actual["sha256"] != expected.get("sha256"):
            raise ValueError(
                f"Existing LitSearch {key} checksum does not match metadata."
            )
    return {**metadata, "reused": True}


def _write_json_atomic(payload: dict, output_path: Path) -> None:
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".json",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
