from llm import call_llm
from tools import SEARCH_PAPER_SCHEMA


INSTRUCTIONS = """
You are a literature assistant.
Use the available tools when information must be retrieved.
Return a final answer when the available information is sufficient.
""".strip()


ALLOWED_TOOLS = [SEARCH_PAPER_SCHEMA]


def decide_next_action(state: dict) -> dict:
    """Ask the LLM for one decision based on the current state."""
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        *state["messages"],
    ]
    return call_llm(messages=messages, tools=ALLOWED_TOOLS)
