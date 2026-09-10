import argparse
import json

from runtime import run_agent
from state import create_state


def print_debug_trace(state: dict) -> None:
    """Print the internal Agent execution trace for learning and debugging."""
    print("\n=== Agent Debug Trace ===")
    print(f"[Input] {state['user_query']}")

    decision_step = 0
    for message in state["messages"][1:]:
        if "tool_call" in message:
            decision_step += 1
            tool_call = message["tool_call"]
            arguments = json.dumps(
                tool_call["arguments"], ensure_ascii=False
            )
            print(f"[Step {decision_step} | LLM] tool_call")
            print(f"  name: {tool_call['name']}")
            print(f"  arguments: {arguments}")
            print(f"  call_id: {tool_call['id']}")
            continue

        if message["role"] == "tool":
            result = json.dumps(message["content"], ensure_ascii=False)
            print(f"[Step {decision_step} | Tool] result")
            print(f"  name: {message['name']}")
            print(f"  content: {result}")
            continue

        if message["role"] == "assistant":
            decision_step += 1
            print(f"[Step {decision_step} | LLM] final")

    print(f"[Summary] LLM steps: {state['step']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal literature Agent demo."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show the Agent's Tool Calls, Tool Results, and step count",
    )
    return parser.parse_args()


def main(debug: bool = False) -> None:
    user_query = input("请输入文献任务：")
    state = create_state(user_query)

    try:
        final_answer = run_agent(state)
    except RuntimeError as error:
        if debug:
            print_debug_trace(state)
        print(f"\n运行失败：{error}")
        raise SystemExit(1) from error

    if debug:
        print_debug_trace(state)

    print("\nFinal Answer:")
    print(final_answer)


if __name__ == "__main__":
    arguments = parse_args()
    main(debug=arguments.debug)
