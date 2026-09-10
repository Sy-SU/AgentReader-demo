import inspect

from agent import decide_next_action
from tools import search_paper


TOOL_REGISTRY = {
    "search_paper": search_paper,
}


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Validate and execute one tool call through the registry."""
    tool = TOOL_REGISTRY.get(tool_name)

    if tool is None:
        return {
            "error": {
                "type": "unknown_tool",
                "message": f"Tool '{tool_name}' is not registered.",
            }
        }

    if not isinstance(arguments, dict):
        return {
            "error": {
                "type": "invalid_arguments",
                "message": "Tool arguments must be a dictionary.",
            }
        }

    try:
        inspect.signature(tool).bind(**arguments)
    except TypeError as error:
        return {
            "error": {
                "type": "invalid_arguments",
                "message": str(error),
            }
        }

    try:
        return tool(**arguments)
    except Exception as error:
        return {
            "error": {
                "type": "tool_execution_error",
                "message": str(error),
            }
        }


def run_agent(state: dict, max_steps: int = 5) -> str:
    """Run the Agent Loop until a final answer or the step limit."""
    while state["step"] < max_steps:
        response = decide_next_action(state)
        state["step"] += 1

        if response["type"] == "final":
            final_answer = response["content"]
            state["messages"].append(
                {"role": "assistant", "content": final_answer}
            )
            return final_answer

        if response["type"] != "tool_call":
            raise ValueError(f"Unknown LLM response type: {response['type']}")

        tool_name = response["tool_name"]
        tool_arguments = response["tool_arguments"]
        tool_call_id = response["tool_call_id"]

        state["messages"].append(
            {
                "role": "assistant",
                "tool_call": {
                    "id": tool_call_id,
                    "name": tool_name,
                    "arguments": tool_arguments,
                },
            }
        )

        observation = execute_tool(tool_name, tool_arguments)
        state["messages"].append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": observation,
            }
        )

    stopped_answer = f"Agent stopped after reaching max_steps={max_steps}."
    state["messages"].append(
        {"role": "assistant", "content": stopped_answer}
    )
    return stopped_answer
