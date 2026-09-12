"""arXiv-first paper search with a Crossref fallback."""

import json
import os
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .library import paper_id


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_TIMEOUT_SECONDS = 8
ARXIV_RETRY_DELAY_SECONDS = 3
MAX_ARXIV_RETRY_DELAY_SECONDS = 10
CROSSREF_API_URL = "https://api.crossref.org/works"
CROSSREF_TIMEOUT_SECONDS = 10
MAX_SEARCH_RESULTS = 3
MAX_SEARCH_CANDIDATES = 10
ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_ID_PATTERN = re.compile(
    r"(?<!\d)(\d{4}\.\d{4,5}(?:v\d+)?)(?!\d)",
    re.IGNORECASE,
)
SEARCH_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+"
)
SEARCH_RANKING_METHOD = "weighted_lexical_overlap_v2"


def search_paper(query: str) -> dict:
    """Search arXiv first, then Crossref, and return candidate papers."""
    normalized_query = " ".join(query.split())
    if not normalized_query:
        raise ValueError("Search query must not be empty.")

    try:
        arxiv_result = _parse_arxiv_results(
            _fetch_arxiv_feed(normalized_query),
            query=normalized_query,
        )
    except RuntimeError as error:
        arxiv_status = str(error)
        fallback_reason = "arxiv_unavailable"
    else:
        if arxiv_result["found"]:
            return arxiv_result
        arxiv_status = "no results"
        fallback_reason = "arxiv_no_results"

    try:
        crossref_result = _parse_crossref_results(
            _fetch_crossref_data(normalized_query),
            query=normalized_query,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "Paper search failed. "
            f"arXiv: {arxiv_status}; Crossref: {error}"
        ) from error

    crossref_result["fallback_reason"] = fallback_reason
    if fallback_reason == "arxiv_unavailable":
        crossref_result["arxiv_error"] = arxiv_status
    return crossref_result


def _fetch_arxiv_feed(query: str) -> bytes:
    """Fetch relevance-ranked candidates from the arXiv Atom API."""
    arxiv_id = _extract_arxiv_id(query)
    if arxiv_id:
        query_parameters = {
            "id_list": arxiv_id,
            "max_results": 1,
        }
    else:
        safe_query = query.replace("\\", " ").replace('"', " ")
        query_parameters = {
            "search_query": f'all:"{safe_query}"',
            "start": 0,
            "max_results": MAX_SEARCH_CANDIDATES,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    parameters = urlencode(query_parameters)
    request = Request(
        f"{ARXIV_API_URL}?{parameters}",
        headers={"User-Agent": "AgentReaderDemo/1.0"},
    )

    for attempt in range(2):
        try:
            with urlopen(request, timeout=ARXIV_TIMEOUT_SECONDS) as response:
                return response.read()
        except HTTPError as error:
            if error.code == 429 and attempt == 0:
                sleep(_arxiv_retry_delay(error))
                continue
            raise RuntimeError(
                f"arXiv returned HTTP {error.code}."
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Could not reach arXiv: {error}") from error

    raise RuntimeError("arXiv request failed after retry.")


def _extract_arxiv_id(query: str) -> str | None:
    match = ARXIV_ID_PATTERN.search(query)
    return match.group(1) if match else None


def _arxiv_retry_delay(error: HTTPError) -> float:
    retry_after = error.headers.get("Retry-After") if error.headers else None
    try:
        delay = float(retry_after)
    except (TypeError, ValueError):
        delay = ARXIV_RETRY_DELAY_SECONDS
    return min(max(delay, 0), MAX_ARXIV_RETRY_DELAY_SECONDS)


def _fetch_crossref_data(query: str) -> bytes:
    """Fetch bibliographic candidates from the Crossref REST API."""
    parameters = {
        "query.bibliographic": query,
        "rows": MAX_SEARCH_CANDIDATES,
        "select": "DOI,title,author,published,URL,abstract,link",
    }
    mailto = os.getenv("CROSSREF_MAILTO")
    if mailto:
        parameters["mailto"] = mailto

    request = Request(
        f"{CROSSREF_API_URL}?{urlencode(parameters)}",
        headers={
            "Accept": "application/json",
            "User-Agent": "AgentReaderDemo/1.0",
        },
    )

    try:
        with urlopen(request, timeout=CROSSREF_TIMEOUT_SECONDS) as response:
            return response.read()
    except HTTPError as error:
        raise RuntimeError(
            f"Crossref returned HTTP {error.code}."
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"Could not reach Crossref: {error}") from error


def _parse_arxiv_results(feed: bytes, query: str) -> dict:
    """Parse candidate papers from an arXiv Atom response."""
    try:
        root = ET.fromstring(feed)
    except ET.ParseError as error:
        raise RuntimeError("arXiv returned invalid XML.") from error

    entries = root.findall("atom:entry", ATOM_NAMESPACE)
    if not entries:
        return _search_result(source="arxiv", papers=[], query=query)

    first_entry_id = _element_text(entries[0], "atom:id")
    if "/api/errors#" in first_entry_id:
        message = (
            _element_text(entries[0], "atom:summary")
            or "Unknown API error"
        )
        raise RuntimeError(f"arXiv API error: {message}")

    papers = [
        _parse_arxiv_entry(entry)
        for entry in entries[:MAX_SEARCH_CANDIDATES]
    ]
    return _search_result(source="arxiv", papers=papers, query=query)


def _parse_arxiv_entry(entry: ET.Element) -> dict:
    entry_id = _element_text(entry, "atom:id")
    abstract_url = _https_url(entry_id)
    pdf_url = None
    for link in entry.findall("atom:link", ATOM_NAMESPACE):
        if link.get("title") == "pdf":
            pdf_url = _https_url(link.get("href", ""))
            break

    authors = [
        _collapse_whitespace(
            author.findtext(
                "atom:name", default="", namespaces=ATOM_NAMESPACE
            )
        )
        for author in entry.findall("atom:author", ATOM_NAMESPACE)
    ]

    return {
        "source": "arxiv",
        "arxiv_id": (
            abstract_url.rsplit("/abs/", 1)[-1]
            if "/abs/" in abstract_url
            else ""
        ),
        "doi": None,
        "title": _element_text(entry, "atom:title"),
        "authors": [author for author in authors if author],
        "abstract": _element_text(entry, "atom:summary"),
        "published": _element_text(entry, "atom:published"),
        "paper_url": abstract_url,
        "pdf_url": pdf_url,
    }


def _parse_crossref_results(data: bytes, query: str) -> dict:
    """Parse candidate papers from a Crossref JSON response."""
    try:
        payload = json.loads(data)
        items = payload["message"]["items"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("Crossref returned invalid JSON.") from error

    if not isinstance(items, list):
        raise RuntimeError("Crossref returned an invalid items field.")

    papers = [
        _parse_crossref_item(item)
        for item in items[:MAX_SEARCH_CANDIDATES]
        if isinstance(item, dict)
    ]
    return _search_result(source="crossref", papers=papers, query=query)


def _parse_crossref_item(item: dict) -> dict:
    titles = item.get("title") or []
    title = _collapse_whitespace(titles[0]) if titles else ""
    authors = []
    for author in item.get("author") or []:
        name = _collapse_whitespace(
            " ".join(
                part
                for part in (author.get("given"), author.get("family"))
                if part
            )
        )
        if name:
            authors.append(name)

    doi = item.get("DOI")
    paper_url = item.get("URL") or (
        f"https://doi.org/{doi}" if doi else None
    )
    pdf_url = None
    for link in item.get("link") or []:
        url = link.get("URL")
        content_type = link.get("content-type", "")
        if url and (content_type == "application/pdf" or "/pdf/" in url):
            pdf_url = _https_url(url)
            break

    return {
        "source": "crossref",
        "arxiv_id": None,
        "doi": doi,
        "title": title,
        "authors": authors,
        "abstract": _strip_markup(item.get("abstract") or "") or None,
        "published": _crossref_date(item.get("published")),
        "paper_url": _https_url(paper_url) if paper_url else None,
        "pdf_url": pdf_url,
    }


def _search_result(source: str, papers: list[dict], query: str) -> dict:
    ranked_papers = _rank_search_candidates(papers, query)
    selected_papers = ranked_papers[:MAX_SEARCH_RESULTS]
    identified_papers = [
        {**paper, "candidate_id": paper_id(paper)}
        for paper in selected_papers
    ]
    return {
        "found": bool(identified_papers),
        "source": source,
        "count": len(identified_papers),
        "candidate_count": len(ranked_papers),
        "truncated": len(ranked_papers) > len(identified_papers),
        "ranking_method": SEARCH_RANKING_METHOD,
        "relevance_assessment": _relevance_assessment(
            source,
            ranked_papers,
            query,
        ),
        "papers": identified_papers,
    }


def _rank_search_candidates(
    papers: list[dict],
    query: str,
) -> list[dict]:
    """Rank a bounded source result using transparent lexical overlap."""
    query_sequence = _search_terms(query)
    query_terms = _unique_terms(query_sequence)
    ranked = []

    for source_index, paper in enumerate(papers):
        title_terms = _search_terms(str(paper.get("title") or ""))
        abstract_terms = _search_terms(str(paper.get("abstract") or ""))
        title_term_set = set(title_terms)
        abstract_term_set = set(abstract_terms)
        title_matches = [
            term for term in query_terms if term in title_term_set
        ]
        abstract_matches = [
            term for term in query_terms if term in abstract_term_set
        ]
        title_phrase_match = _contains_term_sequence(
            title_terms,
            query_sequence,
        )
        exact_title_match = title_terms == query_sequence
        score = (
            4 * len(title_matches)
            + len(abstract_matches)
            + (2 * len(query_terms) if title_phrase_match else 0)
            + (2 * len(query_terms) if exact_title_match else 0)
        )
        ranked.append(
            {
                **paper,
                "local_relevance": {
                    "score": score,
                    "title_matches": title_matches,
                    "abstract_matches": abstract_matches,
                    "title_phrase_match": title_phrase_match,
                    "exact_title_match": exact_title_match,
                    "source_rank": source_index + 1,
                },
                "_source_index": source_index,
            }
        )

    ranked.sort(
        key=lambda paper: (
            -paper["local_relevance"]["score"],
            paper["_source_index"],
        )
    )
    for paper in ranked:
        paper.pop("_source_index")
    return ranked


def _relevance_assessment(
    source: str,
    ranked_papers: list[dict],
    query: str,
) -> dict:
    best_score = (
        ranked_papers[0]["local_relevance"]["score"]
        if ranked_papers
        else None
    )
    queried_arxiv_id = _extract_arxiv_id(query)
    identifier_match = bool(
        source == "arxiv"
        and queried_arxiv_id
        and any(
            _base_arxiv_id(paper.get("arxiv_id"))
            == _base_arxiv_id(queried_arxiv_id)
            for paper in ranked_papers
        )
    )

    if identifier_match:
        status = "identifier_match"
    elif not ranked_papers:
        status = "no_candidates"
    elif best_score is not None and best_score > 0:
        status = "lexical_match"
    else:
        status = "no_lexical_match"

    return {
        "status": status,
        "best_score": best_score,
    }


def _base_arxiv_id(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return re.sub(
        r"v\d+$",
        "",
        value.strip(),
        flags=re.IGNORECASE,
    ).casefold()


def _search_terms(value: str) -> list[str]:
    terms = []
    for match in SEARCH_TOKEN_PATTERN.finditer(value):
        token = match.group(0).casefold()
        if _is_cjk(token) and len(token) > 1:
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


def _unique_terms(terms: list[str]) -> list[str]:
    return list(dict.fromkeys(terms))


def _contains_term_sequence(
    terms: list[str],
    sequence: list[str],
) -> bool:
    if not sequence or len(sequence) > len(terms):
        return False
    sequence_length = len(sequence)
    return any(
        terms[index : index + sequence_length] == sequence
        for index in range(len(terms) - sequence_length + 1)
    )


def _element_text(element: ET.Element, path: str) -> str:
    return _collapse_whitespace(
        element.findtext(path, default="", namespaces=ATOM_NAMESPACE)
    )


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _https_url(value: str) -> str:
    if value.startswith("http://"):
        return f"https://{value.removeprefix('http://')}"
    return value


def _crossref_date(published: object) -> str | None:
    if not isinstance(published, dict):
        return None
    date_parts = published.get("date-parts")
    if not date_parts or not isinstance(date_parts[0], list):
        return None

    parts = date_parts[0][:3]
    if not parts:
        return None
    return "-".join(
        str(part) if index == 0 else f"{part:02d}"
        for index, part in enumerate(parts)
    )


class _MarkupTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _strip_markup(value: str) -> str:
    parser = _MarkupTextExtractor()
    parser.feed(value)
    return _collapse_whitespace(" ".join(parser.parts))
