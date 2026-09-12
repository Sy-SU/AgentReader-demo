def create_state(user_query: str) -> dict:
    """Create the state for one agent run."""
    return {
        "user_query": user_query,
        "messages": [
            {"role": "user", "content": user_query},
        ],
        "step": 0,
    }


def append_user_message(state: dict, content: str) -> None:
    """Append one follow-up message to the current conversation state."""
    state["messages"].append({"role": "user", "content": content})
