"""Deterministic building blocks for minimal local text retrieval."""

import math
import re
from collections import Counter

from .extract import MAX_EXTRACT_CHARS
from .index import (
    INDEX_FORMAT_VERSION,
    load_or_build_paper_index,
    validate_paper_index,
)


DEFAULT_CHUNK_SIZE = 1_200
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 4_000
PAGE_MARKER_PATTERN = re.compile(r"(?m)^\[Page (?P<page>\d+)\]\n")
TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*|"
    r"[\u3400-\u4dbf\u4e00-\u9fff]+"
)
MAX_TOP_K = 5
MAX_RETRIEVAL_QUERY_CHARS = 500
TFIDF_SCORING_METHOD = "tfidf_cosine"
BM25_SCORING_METHOD = "bm25"
SUPPORTED_SCORING_METHODS = (
    TFIDF_SCORING_METHOD,
    BM25_SCORING_METHOD,
)
DEFAULT_SCORING_METHOD = TFIDF_SCORING_METHOD
DEFAULT_MIN_QUERY_TERM_COVERAGE = 0.5
BM25_K1 = 1.5
BM25_B = 0.75


def chunk_paper_text(
    extraction_result: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """Split bounded extracted text into page-local character windows."""
    if not isinstance(extraction_result, dict):
        raise ValueError("extraction_result must be a dictionary.")

    paper_id = extraction_result.get("paper_id")
    text = extraction_result.get("text")
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("extraction_result has no valid paper_id.")
    if not isinstance(text, str):
        raise ValueError("extraction_result text must be a string.")
    if len(text) > MAX_EXTRACT_CHARS:
        raise ValueError(
            f"extraction_result text exceeds {MAX_EXTRACT_CHARS} characters."
        )
    _validate_chunk_settings(chunk_size=chunk_size, overlap=overlap)

    normalized_paper_id = paper_id.strip()
    if not text:
        _validate_source_page_metadata(extraction_result, [])
        return _chunk_result(
            source_result=extraction_result,
            paper_id=normalized_paper_id,
            source_char_count=0,
            page_numbers=[],
            chunks=[],
            chunk_size=chunk_size,
            overlap=overlap,
        )

    pages = _parse_pages(text)
    _validate_source_page_metadata(
        extraction_result,
        [page_number for page_number, _ in pages],
    )
    chunks = []
    for page_number, page_text in pages:
        chunks.extend(
            _chunk_page(
                paper_id=normalized_paper_id,
                page_number=page_number,
                page_text=page_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return _chunk_result(
        source_result=extraction_result,
        paper_id=normalized_paper_id,
        source_char_count=len(text),
        page_numbers=[page_number for page_number, _ in pages],
        chunks=chunks,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def chunk_paper_index(
    index_result: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    """Split all indexed pages into page-local character windows."""
    validate_paper_index(index_result)
    _validate_chunk_settings(chunk_size=chunk_size, overlap=overlap)

    paper_id = index_result["paper_id"].strip()
    chunks = []
    for page in index_result["pages"]:
        chunks.extend(
            _chunk_page(
                paper_id=paper_id,
                page_number=page["page"],
                page_text=page["text"],
                chunk_size=chunk_size,
                overlap=overlap,
            )
        )

    return _chunk_result(
        source_result=index_result,
        paper_id=paper_id,
        source_char_count=index_result["char_count"],
        page_numbers=list(index_result["page_numbers"]),
        chunks=chunks,
        chunk_size=chunk_size,
        overlap=overlap,
    )


def rank_paper_chunks(
    chunk_result: dict,
    query: str,
    top_k: int = 3,
    scoring_method: str = DEFAULT_SCORING_METHOD,
    min_query_term_coverage: float = DEFAULT_MIN_QUERY_TERM_COVERAGE,
) -> dict:
    """Rank chunks deterministically and reject weak corpus evidence."""
    if not isinstance(chunk_result, dict):
        raise ValueError("chunk_result must be a dictionary.")
    normalized_query = _validate_retrieval_request(query, top_k)
    _validate_ranking_settings(
        scoring_method=scoring_method,
        min_query_term_coverage=min_query_term_coverage,
    )

    paper_id = chunk_result.get("paper_id")
    chunks = chunk_result.get("chunks")
    if not isinstance(paper_id, str) or not paper_id.strip():
        raise ValueError("chunk_result has no valid paper_id.")
    if not isinstance(chunks, list):
        raise ValueError("chunk_result chunks must be a list.")
    _validate_chunks(chunks=chunks, paper_id=paper_id.strip())

    query_terms = _tokenize(normalized_query)
    unique_query_terms = list(dict.fromkeys(query_terms))
    tokenized_chunks = [_tokenize(chunk["text"]) for chunk in chunks]
    corpus_terms = set().union(*(set(terms) for terms in tokenized_chunks))
    matched_query_terms = [
        term for term in unique_query_terms if term in corpus_terms
    ]
    query_term_coverage = len(matched_query_terms) / len(unique_query_terms)
    rejected_low_query_coverage = (
        query_term_coverage < min_query_term_coverage
    )

    if scoring_method == TFIDF_SCORING_METHOD:
        scores = _tfidf_scores(tokenized_chunks, query_terms)
    else:
        scores = _bm25_scores(tokenized_chunks, query_terms)

    scored_chunks = []
    for chunk, terms, score in zip(
        chunks,
        tokenized_chunks,
        scores,
        strict=True,
    ):
        if score <= 0:
            continue

        chunk_terms = set(terms)
        matched_terms = [
            term for term in unique_query_terms if term in chunk_terms
        ]
        scored_chunks.append(
            {
                **chunk,
                "score": round(score, 6),
                "matched_terms": matched_terms,
                "_sort_score": score,
            }
        )

    scored_chunks.sort(
        key=lambda chunk: (
            -chunk["_sort_score"],
            chunk["page"],
            chunk["char_start"],
            chunk["chunk_id"],
        )
    )
    selected_chunks = (
        [] if rejected_low_query_coverage else scored_chunks[:top_k]
    )
    for chunk in selected_chunks:
        chunk.pop("_sort_score")

    return {
        "found": bool(selected_chunks),
        "paper_id": paper_id.strip(),
        "title": chunk_result.get("title"),
        "query": normalized_query,
        "query_terms": unique_query_terms,
        "matched_query_terms": matched_query_terms,
        "query_term_coverage": round(query_term_coverage, 6),
        "minimum_query_term_coverage": min_query_term_coverage,
        "rejected_low_query_coverage": rejected_low_query_coverage,
        "scoring_method": scoring_method,
        "count": len(selected_chunks),
        "total_matches": len(scored_chunks),
        "total_chunks": len(chunks),
        "top_k": top_k,
        "truncated": (
            not rejected_low_query_coverage and len(scored_chunks) > top_k
        ),
        "source_truncated": bool(chunk_result.get("source_truncated")),
        "source_truncation_reasons": list(
            chunk_result.get("source_truncation_reasons", [])
        ),
        "matches": selected_chunks,
    }


def retrieve_paper_chunks(
    download_result: dict,
    query: str,
    top_k: int = 3,
) -> dict:
    """Load or build a full index, then rank a few matching chunks."""
    _validate_retrieval_request(query, top_k)
    index_result = load_or_build_paper_index(download_result)
    chunk_result = chunk_paper_index(index_result)
    ranked_result = rank_paper_chunks(
        chunk_result,
        query=query,
        top_k=top_k,
        scoring_method=DEFAULT_SCORING_METHOD,
        min_query_term_coverage=DEFAULT_MIN_QUERY_TERM_COVERAGE,
    )
    ranked_result["page_count"] = index_result["page_count"]
    ranked_result["pages_scanned"] = index_result["pages_scanned"]
    ranked_result["pages_read"] = index_result["pages_indexed"]
    ranked_result["page_numbers"] = list(index_result["page_numbers"])
    ranked_result["last_page_partial"] = index_result[
        "last_page_partial"
    ]
    ranked_result["coverage_complete"] = index_result["coverage_complete"]
    ranked_result["index_status"] = index_result["index_status"]
    ranked_result["index_version"] = INDEX_FORMAT_VERSION
    if "index_rebuild_reason" in index_result:
        ranked_result["index_rebuild_reason"] = index_result[
            "index_rebuild_reason"
        ]
    if "warning" in index_result:
        ranked_result["warning"] = index_result["warning"]
    return ranked_result


def _validate_retrieval_request(query: str, top_k: int) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    normalized_query = query.strip()
    if len(normalized_query) > MAX_RETRIEVAL_QUERY_CHARS:
        raise ValueError(
            f"query exceeds {MAX_RETRIEVAL_QUERY_CHARS} characters."
        )
    if not _tokenize(normalized_query):
        raise ValueError("query must contain searchable text or numbers.")
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or not 1 <= top_k <= MAX_TOP_K
    ):
        raise ValueError(f"top_k must be an integer from 1 to {MAX_TOP_K}.")
    return normalized_query


def _parse_pages(text: str) -> list[tuple[int, str]]:
    markers = list(PAGE_MARKER_PATTERN.finditer(text))
    if not markers or markers[0].start() != 0:
        raise ValueError(
            "extraction_result text must start with a [Page N] marker."
        )

    pages = []
    previous_page_number = 0
    for index, marker in enumerate(markers):
        page_number = int(marker.group("page"))
        if page_number <= previous_page_number:
            raise ValueError("Page markers must be strictly increasing.")
        previous_page_number = page_number

        content_start = marker.end()
        content_end = (
            markers[index + 1].start()
            if index + 1 < len(markers)
            else len(text)
        )
        page_text = text[content_start:content_end].strip()
        pages.append((page_number, page_text))
    return pages


def _validate_source_page_metadata(
    extraction_result: dict,
    parsed_page_numbers: list[int],
) -> None:
    """Ensure text markers agree with the extraction result contract."""
    expected_page_numbers = extraction_result.get("page_numbers")
    if expected_page_numbers is None:
        return
    if (
        not isinstance(expected_page_numbers, list)
        or any(
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
            for page_number in expected_page_numbers
        )
        or expected_page_numbers != sorted(set(expected_page_numbers))
    ):
        raise ValueError(
            "extraction_result page_numbers must be strictly increasing "
            "positive integers."
        )
    if extraction_result.get("pages_read") != len(expected_page_numbers):
        raise ValueError(
            "extraction_result pages_read must equal len(page_numbers)."
        )
    if parsed_page_numbers != expected_page_numbers:
        raise ValueError(
            "Text page markers do not match extraction_result page_numbers."
        )


def _tokenize(value: str) -> list[str]:
    terms = []
    for match in TOKEN_PATTERN.finditer(value):
        token = match.group(0).casefold()
        if _is_cjk(token):
            if len(token) == 1:
                terms.append(token)
            else:
                terms.extend(
                    token[index : index + 2]
                    for index in range(len(token) - 1)
                )
        else:
            terms.append(token)
    return terms


def _is_cjk(value: str) -> bool:
    return bool(value) and (
        "\u3400" <= value[0] <= "\u4dbf"
        or "\u4e00" <= value[0] <= "\u9fff"
    )


def _validate_ranking_settings(
    scoring_method: str,
    min_query_term_coverage: float,
) -> None:
    if scoring_method not in SUPPORTED_SCORING_METHODS:
        supported = ", ".join(SUPPORTED_SCORING_METHODS)
        raise ValueError(f"scoring_method must be one of: {supported}.")
    if (
        isinstance(min_query_term_coverage, bool)
        or not isinstance(min_query_term_coverage, (int, float))
        or not 0 <= min_query_term_coverage <= 1
    ):
        raise ValueError(
            "min_query_term_coverage must be a number from 0 to 1."
        )


def _tfidf_scores(
    tokenized_chunks: list[list[str]],
    query_terms: list[str],
) -> list[float]:
    document_frequency = Counter()
    for terms in tokenized_chunks:
        document_frequency.update(set(terms))

    document_count = len(tokenized_chunks)
    idf = {
        term: math.log(
            (document_count + 1) / (document_frequency[term] + 1)
        )
        + 1
        for term in set(query_terms).union(document_frequency)
    }
    query_vector = _tfidf_vector(query_terms, idf)
    query_norm = _vector_norm(query_vector)

    scores = []
    for terms in tokenized_chunks:
        chunk_vector = _tfidf_vector(terms, idf)
        chunk_norm = _vector_norm(chunk_vector)
        if not chunk_norm:
            scores.append(0.0)
            continue
        dot_product = sum(
            query_vector.get(term, 0.0) * chunk_vector.get(term, 0.0)
            for term in query_vector
        )
        scores.append(dot_product / (query_norm * chunk_norm))
    return scores


def _bm25_scores(
    tokenized_chunks: list[list[str]],
    query_terms: list[str],
) -> list[float]:
    document_count = len(tokenized_chunks)
    if not document_count:
        return []

    document_frequency = Counter()
    for terms in tokenized_chunks:
        document_frequency.update(set(terms))
    average_document_length = sum(
        len(terms) for terms in tokenized_chunks
    ) / document_count
    if not average_document_length:
        return [0.0] * document_count

    unique_query_terms = list(dict.fromkeys(query_terms))
    scores = []
    for terms in tokenized_chunks:
        term_frequency = Counter(terms)
        length_normalization = BM25_K1 * (
            1
            - BM25_B
            + BM25_B * len(terms) / average_document_length
        )
        score = 0.0
        for term in unique_query_terms:
            frequency = term_frequency[term]
            if not frequency:
                continue
            inverse_document_frequency = math.log(
                1
                + (
                    document_count
                    - document_frequency[term]
                    + 0.5
                )
                / (document_frequency[term] + 0.5)
            )
            score += inverse_document_frequency * (
                frequency * (BM25_K1 + 1)
                / (frequency + length_normalization)
            )
        scores.append(score)
    return scores


def _tfidf_vector(
    terms: list[str],
    idf: dict[str, float],
) -> dict[str, float]:
    if not terms:
        return {}
    counts = Counter(terms)
    term_count = len(terms)
    return {
        term: count / term_count * idf[term]
        for term, count in counts.items()
    }


def _vector_norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(weight * weight for weight in vector.values()))


def _validate_chunks(chunks: list[dict], paper_id: str) -> None:
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Each chunk must be a dictionary.")
        if chunk.get("paper_id") != paper_id:
            raise ValueError("Each chunk must match chunk_result paper_id.")
        if not isinstance(chunk.get("chunk_id"), str):
            raise ValueError("Each chunk must have a string chunk_id.")
        if not isinstance(chunk.get("text"), str):
            raise ValueError("Each chunk must have string text.")

        page = chunk.get("page")
        char_start = chunk.get("char_start")
        char_end = chunk.get("char_end")
        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page < 1
        ):
            raise ValueError("Each chunk must have a positive integer page.")
        if (
            isinstance(char_start, bool)
            or not isinstance(char_start, int)
            or char_start < 0
            or isinstance(char_end, bool)
            or not isinstance(char_end, int)
            or char_end < char_start
        ):
            raise ValueError("Each chunk must have valid character offsets.")


def _chunk_page(
    paper_id: str,
    page_number: int,
    page_text: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    chunks = []
    start = 0
    while start < len(page_text):
        end = min(start + chunk_size, len(page_text))
        chunk_text = page_text[start:end]
        if chunk_text.strip():
            chunks.append(
                {
                    "chunk_id": (
                        f"{paper_id}:p{page_number}:{start}-{end}"
                    ),
                    "paper_id": paper_id,
                    "page": page_number,
                    "char_start": start,
                    "char_end": end,
                    "text": chunk_text,
                }
            )

        if end == len(page_text):
            break
        start = end - overlap
    return chunks


def _validate_chunk_settings(chunk_size: int, overlap: int) -> None:
    if (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE
    ):
        raise ValueError(
            f"chunk_size must be an integer from {MIN_CHUNK_SIZE} "
            f"to {MAX_CHUNK_SIZE}."
        )
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, int)
        or not 0 <= overlap <= chunk_size // 2
    ):
        raise ValueError(
            "overlap must be an integer from 0 to half of chunk_size."
        )


def _chunk_result(
    source_result: dict,
    paper_id: str,
    source_char_count: int,
    page_numbers: list[int],
    chunks: list[dict],
    chunk_size: int,
    overlap: int,
) -> dict:
    truncation_reasons = source_result.get("truncation_reasons", [])
    if not isinstance(truncation_reasons, list):
        truncation_reasons = []
    return {
        "paper_id": paper_id,
        "title": source_result.get("title"),
        "source_char_count": source_char_count,
        "source_truncated": bool(source_result.get("truncated")),
        "source_truncation_reasons": list(truncation_reasons),
        "page_numbers": page_numbers,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
