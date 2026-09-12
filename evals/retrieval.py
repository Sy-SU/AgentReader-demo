"""Offline evaluation for the deterministic paper retrieval pipeline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

from tools.index import (
    INDEX_FORMAT_VERSION,
    MAX_INDEX_CHARS,
    MAX_INDEX_PAGES,
)
from tools.retrieval import (
    BM25_SCORING_METHOD,
    DEFAULT_MIN_QUERY_TERM_COVERAGE,
    DEFAULT_SCORING_METHOD,
    SUPPORTED_SCORING_METHODS,
    TFIDF_SCORING_METHOD,
    chunk_paper_index,
    rank_paper_chunks,
)


DATASET_FORMAT_VERSION = 1
DEFAULT_DATASET_PATH = Path(__file__).with_name("retrieval_cases.json")
DEFAULT_TOP_K = 3
COMPARISON_CONFIGURATIONS = (
    ("tfidf_raw", TFIDF_SCORING_METHOD, 0.0),
    ("bm25_raw", BM25_SCORING_METHOD, 0.0),
    (
        "tfidf_guarded",
        TFIDF_SCORING_METHOD,
        DEFAULT_MIN_QUERY_TERM_COVERAGE,
    ),
    (
        "bm25_guarded",
        BM25_SCORING_METHOD,
        DEFAULT_MIN_QUERY_TERM_COVERAGE,
    ),
)


def load_retrieval_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> dict:
    """Read and validate a deterministic retrieval evaluation dataset."""
    dataset_path = Path(path)
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Evaluation dataset is not valid JSON: {error.msg}."
        ) from error
    validate_retrieval_dataset(dataset)
    return dataset


def validate_retrieval_dataset(dataset: object) -> None:
    """Validate the small, versioned evaluation dataset contract."""
    if not isinstance(dataset, dict):
        raise ValueError("Evaluation dataset must be a JSON object.")
    if dataset.get("version") != DATASET_FORMAT_VERSION:
        raise ValueError("Evaluation dataset version is unsupported.")

    description = dataset.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Evaluation dataset needs a description.")

    paper = dataset.get("paper")
    if not isinstance(paper, dict):
        raise ValueError("Evaluation dataset needs a paper object.")
    paper_id = paper.get("paper_id")
    title = paper.get("title")
    pages = paper.get("pages")
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("Evaluation paper needs a non-empty paper_id.")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Evaluation paper needs a non-empty title.")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Evaluation paper needs at least one page.")

    page_numbers = []
    for page in pages:
        if not isinstance(page, dict) or set(page) != {"page", "text"}:
            raise ValueError(
                "Each evaluation page must contain only page and text."
            )
        page_number = page["page"]
        page_text = page["text"]
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
        ):
            raise ValueError(
                "Evaluation page numbers must be positive integers."
            )
        if not isinstance(page_text, str) or not page_text.strip():
            raise ValueError("Evaluation page text must be non-empty.")
        page_numbers.append(page_number)
    if page_numbers != sorted(set(page_numbers)):
        raise ValueError(
            "Evaluation page numbers must be strictly increasing."
        )

    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation dataset needs at least one case.")
    case_ids = []
    valid_pages = set(page_numbers)
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each evaluation case must be an object.")
        case_id = case.get("id")
        query = case.get("query")
        expected_pages = case.get("expected_pages")
        tags = case.get("tags")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Each evaluation case needs a non-empty id.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Evaluation case {case_id!r} needs a non-empty query."
            )
        if not isinstance(expected_pages, list) or any(
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number not in valid_pages
            for page_number in expected_pages
        ):
            raise ValueError(
                f"Evaluation case {case_id!r} has invalid expected_pages."
            )
        if expected_pages != list(dict.fromkeys(expected_pages)):
            raise ValueError(
                f"Evaluation case {case_id!r} repeats an expected page."
            )
        if (
            not isinstance(tags, list)
            or not tags
            or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
            or tags != list(dict.fromkeys(tags))
        ):
            raise ValueError(
                f"Evaluation case {case_id!r} has invalid tags."
            )
        case_ids.append(case_id)
    if case_ids != list(dict.fromkeys(case_ids)):
        raise ValueError("Evaluation case ids must be unique.")


def evaluate_retrieval_dataset(
    dataset: dict,
    top_k: int = DEFAULT_TOP_K,
    scoring_method: str = DEFAULT_SCORING_METHOD,
    min_query_term_coverage: float = DEFAULT_MIN_QUERY_TERM_COVERAGE,
) -> dict:
    """Evaluate one retrieval configuration against evidence pages."""
    validate_retrieval_dataset(dataset)
    index_result = _dataset_index(dataset["paper"])
    chunk_result = chunk_paper_index(index_result)

    case_results = []
    for case in dataset["cases"]:
        ranked = rank_paper_chunks(
            chunk_result,
            query=case["query"],
            top_k=top_k,
            scoring_method=scoring_method,
            min_query_term_coverage=min_query_term_coverage,
        )
        retrieved_pages = list(
            dict.fromkeys(match["page"] for match in ranked["matches"])
        )
        expected_pages = list(case["expected_pages"])
        answerable = bool(expected_pages)

        recall_at_k = None
        hit_at_k = None
        reciprocal_rank = None
        no_answer_correct = None
        if answerable:
            matched_pages = set(expected_pages).intersection(retrieved_pages)
            recall_at_k = len(matched_pages) / len(expected_pages)
            hit_at_k = bool(matched_pages)
            reciprocal_rank = _reciprocal_rank(
                ranked["matches"], set(expected_pages)
            )
            passed = recall_at_k == 1.0
        else:
            no_answer_correct = not ranked["found"]
            passed = no_answer_correct

        case_results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "tags": list(case["tags"]),
                "answerable": answerable,
                "expected_pages": expected_pages,
                "retrieved_pages": retrieved_pages,
                "query_term_coverage": ranked["query_term_coverage"],
                "rejected_low_query_coverage": ranked[
                    "rejected_low_query_coverage"
                ],
                "retrieved": [
                    {
                        "rank": rank,
                        "page": match["page"],
                        "score": match["score"],
                        "matched_terms": list(match["matched_terms"]),
                    }
                    for rank, match in enumerate(ranked["matches"], start=1)
                ],
                "recall_at_k": _round_metric(recall_at_k),
                "hit_at_k": hit_at_k,
                "reciprocal_rank": _round_metric(reciprocal_rank),
                "no_answer_correct": no_answer_correct,
                "passed": passed,
            }
        )

    aggregate = _aggregate(case_results)
    tags = sorted(
        {tag for case_result in case_results for tag in case_result["tags"]}
    )
    return {
        "dataset_version": dataset["version"],
        "description": dataset["description"],
        "paper_id": dataset["paper"]["paper_id"],
        "paper_title": dataset["paper"]["title"],
        "scoring_method": scoring_method,
        "minimum_query_term_coverage": min_query_term_coverage,
        "top_k": top_k,
        **aggregate,
        "by_tag": {
            tag: _aggregate(
                [
                    case_result
                    for case_result in case_results
                    if tag in case_result["tags"]
                ]
            )
            for tag in tags
        },
        "cases": case_results,
    }


def compare_retrieval_methods(
    dataset: dict,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """Compare raw and guarded TF-IDF/BM25 on the same dataset."""
    configurations = []
    for configuration_id, scoring_method, minimum_coverage in (
        COMPARISON_CONFIGURATIONS
    ):
        result = evaluate_retrieval_dataset(
            dataset,
            top_k=top_k,
            scoring_method=scoring_method,
            min_query_term_coverage=minimum_coverage,
        )
        result["configuration_id"] = configuration_id
        configurations.append(result)

    selected = max(configurations, key=_quality_key)
    return {
        "comparison_version": 1,
        "dataset_version": dataset["version"],
        "description": dataset["description"],
        "top_k": top_k,
        "configurations": configurations,
        "decision": {
            "selected_configuration": selected["configuration_id"],
            "selected_scoring_method": selected["scoring_method"],
            "minimum_query_term_coverage": selected[
                "minimum_query_term_coverage"
            ],
            "selection_policy": (
                "Prefer higher Recall@K, no-answer accuracy, Hit Rate@K, "
                "then MRR; ties keep TF-IDF for lower change cost."
            ),
        },
    }


def format_evaluation_report(result: dict) -> str:
    """Format the evaluation result for a human-readable terminal report."""
    metrics = result["metrics"]
    lines = [
        "=== Retrieval Evaluation ===",
        f"Dataset: {result['description']}",
        (
            f"Ranking: {result['scoring_method']} | Top K: {result['top_k']} "
            "| Minimum query coverage: "
            f"{result['minimum_query_term_coverage']:.2f}"
        ),
        (
            "Cases: "
            f"{result['case_count']} "
            f"({result['answerable_count']} answerable, "
            f"{result['no_answer_count']} no-answer)"
        ),
        "",
        (
            f"Recall@{result['top_k']}: "
            f"{_format_metric(metrics['recall_at_k'])}"
        ),
        (
            f"Hit Rate@{result['top_k']}: "
            f"{_format_metric(metrics['hit_rate_at_k'])}"
        ),
        f"MRR: {_format_metric(metrics['mrr'])}",
        (
            "No-answer Accuracy: "
            f"{_format_metric(metrics['no_answer_accuracy'])}"
        ),
    ]

    failed_cases = [case for case in result["cases"] if not case["passed"]]
    lines.extend(["", f"Failed cases: {len(failed_cases)}"])
    for case in failed_cases:
        lines.append(
            "- "
            f"{case['id']}: expected={case['expected_pages']}, "
            f"retrieved={case['retrieved_pages']}"
        )
    if not failed_cases:
        lines.append("- none")
    return "\n".join(lines)


def format_comparison_report(comparison: dict) -> str:
    """Format a compact side-by-side retrieval comparison."""
    lines = [
        "=== Retrieval Method Comparison ===",
        f"Dataset: {comparison['description']}",
        f"Top K: {comparison['top_k']}",
        "",
        "Configuration       Recall  HitRate  MRR     NoAnswer",
    ]
    for result in comparison["configurations"]:
        metrics = result["metrics"]
        lines.append(
            f"{result['configuration_id']:<19} "
            f"{_format_metric(metrics['recall_at_k']):<7} "
            f"{_format_metric(metrics['hit_rate_at_k']):<8} "
            f"{_format_metric(metrics['mrr']):<7} "
            f"{_format_metric(metrics['no_answer_accuracy'])}"
        )
    decision = comparison["decision"]
    lines.extend(
        [
            "",
            f"Selected: {decision['selected_configuration']}",
            decision["selection_policy"],
        ]
    )
    return "\n".join(lines)


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the local paper retrieval baseline offline."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to a versioned retrieval evaluation JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve for each case (default: 3).",
    )
    parser.add_argument(
        "--method",
        choices=SUPPORTED_SCORING_METHODS,
        default=DEFAULT_SCORING_METHOD,
        help="Ranking method for a single evaluation.",
    )
    parser.add_argument(
        "--minimum-query-coverage",
        type=float,
        default=DEFAULT_MIN_QUERY_TERM_COVERAGE,
        help="Minimum fraction of unique query terms found in the corpus.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare raw and guarded TF-IDF/BM25 configurations.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete machine-readable evaluation result.",
    )
    args = parser.parse_args(argv)

    try:
        dataset = load_retrieval_dataset(args.dataset)
        if args.compare:
            result = compare_retrieval_methods(dataset, top_k=args.top_k)
        else:
            result = evaluate_retrieval_dataset(
                dataset,
                top_k=args.top_k,
                scoring_method=args.method,
                min_query_term_coverage=args.minimum_query_coverage,
            )
    except (OSError, ValueError) as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.compare:
        print(format_comparison_report(result))
    else:
        print(format_evaluation_report(result))
    return 0


def _dataset_index(paper: dict) -> dict:
    pages = [
        {"page": page["page"], "text": page["text"].strip()}
        for page in paper["pages"]
    ]
    page_numbers = [page["page"] for page in pages]
    return {
        "version": INDEX_FORMAT_VERSION,
        "paper_id": paper["paper_id"].strip(),
        "title": paper["title"].strip(),
        "pdf_sha256": "0" * 64,
        "pdf_size_bytes": 0,
        "max_pages": MAX_INDEX_PAGES,
        "max_chars": MAX_INDEX_CHARS,
        "page_count": max(page_numbers),
        "pages_scanned": max(page_numbers),
        "pages_indexed": len(pages),
        "page_numbers": page_numbers,
        "last_page_partial": False,
        "char_count": sum(len(page["text"]) for page in pages),
        "text_available": True,
        "coverage_complete": True,
        "truncated": False,
        "truncation_reasons": [],
        "pages": pages,
        "index_status": "evaluation",
    }


def _reciprocal_rank(matches: list[dict], expected_pages: set[int]) -> float:
    for rank, match in enumerate(matches, start=1):
        if match["page"] in expected_pages:
            return 1 / rank
    return 0.0


def _aggregate(case_results: Iterable[dict]) -> dict:
    cases = list(case_results)
    answerable = [case for case in cases if case["answerable"]]
    no_answer = [case for case in cases if not case["answerable"]]
    return {
        "case_count": len(cases),
        "answerable_count": len(answerable),
        "no_answer_count": len(no_answer),
        "metrics": {
            "recall_at_k": _mean(
                case["recall_at_k"] for case in answerable
            ),
            "hit_rate_at_k": _mean(
                float(case["hit_at_k"]) for case in answerable
            ),
            "mrr": _mean(
                case["reciprocal_rank"] for case in answerable
            ),
            "no_answer_accuracy": _mean(
                float(case["no_answer_correct"]) for case in no_answer
            ),
        },
    }


def _quality_key(
    result: dict,
) -> tuple[float, float, float, float, int]:
    metrics = result["metrics"]
    quality = tuple(
        -1.0 if metrics[name] is None else metrics[name]
        for name in (
            "recall_at_k",
            "no_answer_accuracy",
            "hit_rate_at_k",
            "mrr",
        )
    )
    tfidf_tie_breaker = int(
        result["scoring_method"] == TFIDF_SCORING_METHOD
    )
    return (*quality, tfidf_tie_breaker)


def _mean(values: Iterable[float]) -> float | None:
    numbers = list(values)
    if not numbers:
        return None
    return round(sum(numbers) / len(numbers), 6)


def _round_metric(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"
