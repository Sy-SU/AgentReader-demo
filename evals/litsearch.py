"""Compatibility scorer for the official LitSearch retrieval benchmark.

The official LitSearch retrieval script writes each query together with its
gold ``corpusids`` and ranked ``retrieved`` Semantic Scholar corpus IDs.  This
module reads that JSON/JSONL shape without pulling the 2.85 GB dataset or any
of the benchmark's heavyweight model dependencies into AgentReader.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path


LITSEARCH_EVALUATION_VERSION = 1
DEFAULT_TOP_KS = (5, 20)
LITSEARCH_DATASET_URL = (
    "https://huggingface.co/datasets/princeton-nlp/LitSearch"
)
LITSEARCH_REPOSITORY_URL = "https://github.com/princeton-nlp/LitSearch"


def load_litsearch_results(path: str | Path) -> list[dict]:
    """Load official-style LitSearch retrieval results from JSON or JSONL."""
    records = _load_json_records(path)
    normalized = [
        _normalize_record(record, index, require_retrieved=True)
        for index, record in enumerate(records)
    ]
    if not normalized:
        raise ValueError("LitSearch results must contain at least one record.")
    return normalized


def load_litsearch_queries(path: str | Path) -> list[dict]:
    """Load official LitSearch queries before retrieval has been run."""
    return normalize_litsearch_queries(_load_json_records(path))


def normalize_litsearch_queries(records: Iterable[dict]) -> list[dict]:
    """Validate in-memory LitSearch query records without predictions."""
    normalized = [
        _normalize_record(record, index, require_retrieved=False)
        for index, record in enumerate(records)
    ]
    if not normalized:
        raise ValueError("LitSearch queries must contain at least one record.")
    return normalized


def _load_json_records(path: str | Path) -> list[object]:
    result_path = Path(path)
    if result_path.suffix.casefold() == ".jsonl":
        records = []
        for line_number, line in enumerate(
            result_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    "LitSearch JSONL line "
                    f"{line_number} is invalid: {error.msg}."
                ) from error
    elif result_path.suffix.casefold() == ".json":
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"LitSearch JSON is invalid: {error.msg}."
            ) from error
        records = _records_from_payload(payload)
    else:
        raise ValueError("LitSearch input must use .json or .jsonl.")
    return records


def evaluate_litsearch(
    records: Iterable[dict],
    top_ks: Iterable[int] = DEFAULT_TOP_KS,
) -> dict:
    """Evaluate ranked corpus IDs with LitSearch's macro Recall@K."""
    normalized_records = [
        _normalize_record(record, index, require_retrieved=True)
        for index, record in enumerate(records)
    ]
    if not normalized_records:
        raise ValueError("LitSearch results must contain at least one record.")
    normalized_top_ks = _normalize_top_ks(top_ks)

    cases = []
    for index, record in enumerate(normalized_records):
        recalls = {
            str(top_k): _recall_at_k(
                record["retrieved"], record["corpusids"], top_k
            )
            for top_k in normalized_top_ks
        }
        cases.append(
            {
                "index": index,
                "query": record["query"],
                "query_set": record["query_set"],
                "specificity": record["specificity"],
                "quality": record["quality"],
                "gold_corpusids": list(record["corpusids"]),
                "retrieved_corpusids": list(record["retrieved"]),
                "recall_at_k": recalls,
            }
        )

    return {
        "evaluation_version": LITSEARCH_EVALUATION_VERSION,
        "benchmark": "LitSearch",
        "source": {
            "dataset": LITSEARCH_DATASET_URL,
            "repository": LITSEARCH_REPOSITORY_URL,
        },
        "case_count": len(cases),
        "top_ks": list(normalized_top_ks),
        "official_metrics": {
            "broad_recall_at_20": _subset_recall(
                cases, specificity=0, top_k=20
            ),
            "specific_recall_at_5": _subset_recall(
                cases, specificity=1, top_k=5
            ),
            "specific_recall_at_20": _subset_recall(
                cases, specificity=1, top_k=20
            ),
        },
        "all_queries": {
            f"recall_at_{top_k}": _mean(
                case["recall_at_k"][str(top_k)] for case in cases
            )
            for top_k in normalized_top_ks
        },
        "by_query_set": {
            query_set: _group_metrics(cases, normalized_top_ks, query_set)
            for query_set in sorted(
                {case["query_set"] for case in cases}
            )
        },
        "cases": cases,
    }


def format_litsearch_report(result: dict) -> str:
    """Format the official headline metrics and local diagnostics."""
    official = result["official_metrics"]
    lines = [
        "=== LitSearch Benchmark ===",
        f"Cases: {result['case_count']}",
        "Official protocol (macro Recall@K):",
        (
            "  Broad Recall@20: "
            f"{_format_metric(official['broad_recall_at_20'])}"
        ),
        (
            "  Specific Recall@5: "
            f"{_format_metric(official['specific_recall_at_5'])}"
        ),
        (
            "  Specific Recall@20: "
            f"{_format_metric(official['specific_recall_at_20'])}"
        ),
        "All-query diagnostics:",
    ]
    for top_k in result["top_ks"]:
        value = result["all_queries"][f"recall_at_{top_k}"]
        lines.append(f"  Recall@{top_k}: {_format_metric(value)}")
    lines.extend(
        [
            "",
            "Input must contain gold `corpusids` and ranked `retrieved` IDs.",
            "This scorer does not call an LLM or download the full corpus.",
        ]
    )
    return "\n".join(lines)


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score official-style LitSearch retrieval JSON/JSONL results."
        )
    )
    parser.add_argument(
        "results",
        type=Path,
        help=(
            "LitSearch result file containing query, corpusids, specificity, "
            "and retrieved fields."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        action="append",
        dest="top_ks",
        help=(
            "Additional Recall cutoff; may be repeated. Official 5 and 20 "
            "cutoffs are always included."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete machine-readable report.",
    )
    args = parser.parse_args(argv)

    try:
        records = load_litsearch_results(args.results)
        result = evaluate_litsearch(
            records,
            top_ks=args.top_ks or DEFAULT_TOP_KS,
        )
    except (OSError, ValueError) as error:
        print(f"LitSearch evaluation failed: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_litsearch_report(result))
    return 0


def _records_from_payload(payload: object) -> list[object]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError(
        "LitSearch JSON must be a record list or a dataset-viewer object "
        "with a rows list."
    )


def _normalize_record(
    record: object,
    index: int,
    *,
    require_retrieved: bool,
) -> dict:
    if (
        isinstance(record, dict)
        and isinstance(record.get("row"), dict)
    ):
        record = record["row"]
    if not isinstance(record, dict):
        raise ValueError(f"LitSearch record {index} must be an object.")

    query = record.get("query")
    query_set = record.get("query_set")
    specificity = record.get("specificity")
    quality = record.get("quality")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"LitSearch record {index} needs a query.")
    if not isinstance(query_set, str) or not query_set.strip():
        raise ValueError(f"LitSearch record {index} needs a query_set.")
    if isinstance(specificity, bool) or specificity not in {0, 1}:
        raise ValueError(
            f"LitSearch record {index} specificity must be 0 or 1."
        )
    if isinstance(quality, bool) or quality not in {1, 2}:
        raise ValueError(
            f"LitSearch record {index} quality must be 1 or 2."
        )

    corpusids = _normalize_corpusids(
        record.get("corpusids"),
        f"LitSearch record {index} corpusids",
        allow_empty=False,
    )
    normalized = {
        "query": query.strip(),
        "query_set": query_set.strip(),
        "specificity": specificity,
        "quality": quality,
        "corpusids": corpusids,
    }
    if require_retrieved:
        normalized["retrieved"] = _normalize_corpusids(
            record.get("retrieved"),
            f"LitSearch record {index} retrieved",
            allow_empty=True,
            allow_duplicates=True,
        )
    return normalized


def _normalize_corpusids(
    values: object,
    label: str,
    *,
    allow_empty: bool,
    allow_duplicates: bool = False,
) -> list[int]:
    if not isinstance(values, list) or (not values and not allow_empty):
        suffix = "a list" if allow_empty else "a non-empty list"
        raise ValueError(f"{label} must be {suffix}.")
    normalized = []
    for value in values:
        corpusid = _candidate_corpusid(value)
        if corpusid is None:
            raise ValueError(f"{label} contains an invalid corpus ID.")
        if not allow_duplicates and corpusid in normalized:
            raise ValueError(f"{label} must not contain duplicate corpus IDs.")
        normalized.append(corpusid)
    return normalized


def _candidate_corpusid(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    if isinstance(value, (list, tuple)) and value:
        return _candidate_corpusid(value[0])
    if isinstance(value, dict):
        return _candidate_corpusid(value.get("corpusid"))
    return None


def _normalize_top_ks(top_ks: Iterable[int]) -> tuple[int, ...]:
    normalized = []
    for top_k in top_ks:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("LitSearch top_k values must be positive integers.")
        if top_k not in normalized:
            normalized.append(top_k)
    if not normalized:
        raise ValueError("LitSearch needs at least one top_k value.")
    for required in DEFAULT_TOP_KS:
        if required not in normalized:
            normalized.append(required)
    return tuple(sorted(normalized))


def _recall_at_k(retrieved: list[int], gold: list[int], top_k: int) -> float:
    matched = set(retrieved[:top_k]).intersection(gold)
    return len(matched) / len(gold)


def _subset_recall(
    cases: list[dict],
    *,
    specificity: int,
    top_k: int,
) -> float | None:
    selected = [
        case["recall_at_k"][str(top_k)]
        for case in cases
        if case["specificity"] == specificity
    ]
    return _mean(selected)


def _group_metrics(
    cases: list[dict],
    top_ks: tuple[int, ...],
    query_set: str,
) -> dict:
    selected = [case for case in cases if case["query_set"] == query_set]
    return {
        "case_count": len(selected),
        **{
            f"recall_at_{top_k}": _mean(
                case["recall_at_k"][str(top_k)] for case in selected
            )
            for top_k in top_ks
        },
    }


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"
