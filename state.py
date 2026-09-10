def create_state(user_query: str) -> dict:
    """Create the state for one agent run."""
    return {
        "user_query": user_query,
        "messages": [
            {"role": "user", "content": user_query},
        ],
        "step": 0,
    }
