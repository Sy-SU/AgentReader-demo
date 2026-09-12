"""LLM-visible JSON schemas for the available Tools."""


SEARCH_PAPER_SCHEMA = {
    "name": "search_paper",
    "description": (
        "Search arXiv first and Crossref as fallback. Returns up to three "
        "candidate papers after deterministic local relevance ranking."
    ),
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


SAVE_PAPER_SCHEMA = {
    "name": "save_paper",
    "description": (
        "Save one search candidate to the local JSON library. Only call this "
        "when the user explicitly asks to save a paper, and pass the exact "
        "candidate_id returned by search_paper."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_id": {
                "type": "string",
                "description": (
                    "The exact candidate_id from a previous search_paper "
                    "result in the current task."
                ),
            }
        },
        "required": ["candidate_id"],
        "additionalProperties": False,
    },
}


LIST_LIBRARY_SCHEMA = {
    "name": "list_library",
    "description": (
        "List recently saved papers from the local JSON library without "
        "modifying it. Abstracts are omitted from the result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": (
                    "Maximum number of recent papers to return. Defaults "
                    "to 20."
                ),
            }
        },
        "additionalProperties": False,
    },
}


DOWNLOAD_PAPER_SCHEMA = {
    "name": "download_paper",
    "description": (
        "Download and cache the PDF for one paper previously returned by "
        "search_paper or list_library. Pass only its exact stable ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": (
                    "The exact candidate_id from search_paper or id from "
                    "list_library."
                ),
            }
        },
        "required": ["paper_id"],
        "additionalProperties": False,
    },
}


EXTRACT_PAPER_TEXT_SCHEMA = {
    "name": "extract_paper_text",
    "description": (
        "Extract bounded text from a PDF previously returned by "
        "download_paper. Pass only its paper_id and optional limits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": (
                    "The exact paper_id from a previous download_paper "
                    "result in the current task."
                ),
            },
            "max_pages": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": (
                    "Maximum pages to parse from the beginning. Defaults "
                    "to 5."
                ),
            },
            "max_chars": {
                "type": "integer",
                "minimum": 500,
                "maximum": 20000,
                "description": (
                    "Maximum extracted characters returned to the model. "
                    "Defaults to 12000."
                ),
            },
        },
        "required": ["paper_id"],
        "additionalProperties": False,
    },
}


RETRIEVE_PAPER_CHUNKS_SCHEMA = {
    "name": "retrieve_paper_chunks",
    "description": (
        "Find a few relevant text chunks through a bounded full-document "
        "local index for a PDF previously returned by download_paper. Use "
        "this for a focused question or topic, and pass only its paper_id, "
        "a lexical search query, and optional top_k."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "paper_id": {
                "type": "string",
                "description": (
                    "The exact paper_id from a previous download_paper "
                    "result in the current task."
                ),
            },
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
                "description": (
                    "Focused keywords for lexical retrieval. For an English "
                    "paper, include likely English technical terms."
                ),
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
                "description": (
                    "Maximum relevant chunks to return. Defaults to 3."
                ),
            },
        },
        "required": ["paper_id", "query"],
        "additionalProperties": False,
    },
}
