PAPER_CATALOG = {
    "tasa": {
        "title": "TASA (demo record)",
        "pdf_url": "https://example.com/tasa.pdf",
    }
}


SEARCH_PAPER_SCHEMA = {
    "name": "search_paper",
    "description": "Search the local demo paper catalog by title or keyword.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The paper title or keyword to search for.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def search_paper(query: str) -> dict:
    """Search the local demo catalog and return a structured result."""
    paper = PAPER_CATALOG.get(query.strip().lower())

    if paper is None:
        return {
            "found": False,
            "title": None,
            "pdf_url": None,
        }

    return {
        "found": True,
        "title": paper["title"],
        "pdf_url": paper["pdf_url"],
    }
