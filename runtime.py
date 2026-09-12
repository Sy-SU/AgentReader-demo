import inspect
from collections.abc import Callable
from copy import deepcopy
from time import perf_counter

from agent import decide_next_action
from events import AgentEvent, EventHandler
from tools import (
    download_paper,
    extract_paper_text,
    list_library,
    retrieve_paper_chunks,
    save_paper,
    search_paper,
)


TOOL_REGISTRY = {
    "search_paper": search_paper,
    "save_paper": save_paper,
    "list_library": list_library,
    "download_paper": download_paper,
    "extract_paper_text": extract_paper_text,
    "retrieve_paper_chunks": retrieve_paper_chunks,
}
MAX_SEARCH_ATTEMPTS_PER_TURN = 2


def execute_tool(
    tool_name: str,
    arguments: dict,
    state: dict | None = None,
    confirm_save: Callable[[dict], bool] | None = None,
) -> dict:
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

    if tool_name == "save_paper":
        resolved_arguments = _resolve_save_arguments(arguments, state)
        if "error" in resolved_arguments:
            return resolved_arguments

        paper = resolved_arguments["paper"]
        if confirm_save is None:
            return {
                "error": {
                    "type": "approval_required",
                    "message": (
                        "save_paper requires explicit user confirmation."
                    ),
                }
            }

        try:
            approved = confirm_save(paper)
        except Exception as error:
            return {
                "error": {
                    "type": "approval_error",
                    "message": f"Could not confirm save_paper: {error}",
                }
            }

        if not approved:
            return {
                "saved": False,
                "reason": "user_declined",
                "candidate_id": paper.get("candidate_id"),
                "title": paper.get("title"),
            }

        arguments = resolved_arguments

    if tool_name == "download_paper":
        resolved_arguments = _resolve_download_arguments(arguments, state)
        if "error" in resolved_arguments:
            return resolved_arguments
        arguments = resolved_arguments

    if tool_name == "extract_paper_text":
        resolved_arguments = _resolve_extract_arguments(arguments, state)
        if "error" in resolved_arguments:
            return resolved_arguments
        arguments = resolved_arguments

    if tool_name == "retrieve_paper_chunks":
        resolved_arguments = _resolve_retrieval_arguments(arguments, state)
        if "error" in resolved_arguments:
            return resolved_arguments
        arguments = resolved_arguments

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


def _resolve_save_arguments(arguments: dict, state: dict | None) -> dict:
    """Resolve a model-selected candidate ID to trusted search metadata."""
    if set(arguments) != {"candidate_id"}:
        return {
            "error": {
                "type": "invalid_arguments",
                "message": "save_paper requires only candidate_id.",
            }
        }

    candidate_id = arguments["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return {
            "error": {
                "type": "invalid_arguments",
                "message": "candidate_id must be a non-empty string.",
            }
        }

    messages = state.get("messages", []) if isinstance(state, dict) else []
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if message.get("name") != "search_paper":
            continue

        result = message.get("content")
        papers = result.get("papers", []) if isinstance(result, dict) else []
        for paper in papers:
            if (
                isinstance(paper, dict)
                and paper.get("candidate_id") == candidate_id
            ):
                return {"paper": paper}

    return {
        "error": {
            "type": "unknown_candidate",
            "message": (
                f"Candidate '{candidate_id}' was not returned by "
                "search_paper in the current task."
            ),
        }
    }


def _resolve_download_arguments(arguments: dict, state: dict | None) -> dict:
    """Resolve a stable paper ID from trusted search or library results."""
    if set(arguments) != {"paper_id"}:
        return {
            "error": {
                "type": "invalid_arguments",
                "message": "download_paper requires only paper_id.",
            }
        }

    requested_id = arguments["paper_id"]
    if not isinstance(requested_id, str) or not requested_id.strip():
        return {
            "error": {
                "type": "invalid_arguments",
                "message": "paper_id must be a non-empty string.",
            }
        }

    messages = state.get("messages", []) if isinstance(state, dict) else []
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if message.get("name") not in {"search_paper", "list_library"}:
            continue

        result = message.get("content")
        papers = result.get("papers", []) if isinstance(result, dict) else []
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            trusted_id = paper.get("candidate_id") or paper.get("id")
            if trusted_id == requested_id:
                return {"paper": paper}

    return {
        "error": {
            "type": "unknown_paper",
            "message": (
                f"Paper '{requested_id}' was not returned by search_paper "
                "or list_library in the current task."
            ),
        }
    }


def _resolve_extract_arguments(arguments: dict, state: dict | None) -> dict:
    """Resolve a paper ID to a trusted prior PDF download result."""
    allowed_arguments = {"paper_id", "max_pages", "max_chars"}
    if "paper_id" not in arguments or not set(arguments) <= allowed_arguments:
        return {
            "error": {
                "type": "invalid_arguments",
                "message": (
                    "extract_paper_text requires paper_id and accepts only "
                    "optional max_pages and max_chars."
                ),
            }
        }

    requested_id = arguments["paper_id"]
    if not isinstance(requested_id, str) or not requested_id.strip():
        return {
            "error": {
                "type": "invalid_arguments",
                "message": "paper_id must be a non-empty string.",
            }
        }

    messages = state.get("messages", []) if isinstance(state, dict) else []
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if message.get("name") != "download_paper":
            continue

        result = message.get("content")
        if not isinstance(result, dict) or "error" in result:
            continue
        if result.get("paper_id") != requested_id:
            continue
        if not isinstance(result.get("local_path"), str):
            continue

        resolved = {"download_result": result}
        if "max_pages" in arguments:
            resolved["max_pages"] = arguments["max_pages"]
        if "max_chars" in arguments:
            resolved["max_chars"] = arguments["max_chars"]
        return resolved

    return {
        "error": {
            "type": "unknown_download",
            "message": (
                f"Paper '{requested_id}' was not returned by "
                "download_paper in the current task."
            ),
        }
    }


def _resolve_retrieval_arguments(arguments: dict, state: dict | None) -> dict:
    """Resolve retrieval input to a trusted prior PDF download result."""
    allowed_arguments = {"paper_id", "query", "top_k"}
    required_arguments = {"paper_id", "query"}
    if (
        not required_arguments <= set(arguments)
        or not set(arguments) <= allowed_arguments
    ):
        return {
            "error": {
                "type": "invalid_arguments",
                "message": (
                    "retrieve_paper_chunks requires paper_id and query, and "
                    "accepts only optional top_k."
                ),
            }
        }

    requested_id = arguments["paper_id"]
    if not isinstance(requested_id, str) or not requested_id.strip():
        return {
            "error": {
                "type": "invalid_arguments",
                "message": "paper_id must be a non-empty string.",
            }
        }

    messages = state.get("messages", []) if isinstance(state, dict) else []
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if message.get("name") != "download_paper":
            continue

        result = message.get("content")
        if not isinstance(result, dict) or "error" in result:
            continue
        if result.get("paper_id") != requested_id:
            continue
        if not isinstance(result.get("local_path"), str):
            continue

        resolved = {
            "download_result": result,
            "query": arguments["query"],
        }
        if "top_k" in arguments:
            resolved["top_k"] = arguments["top_k"]
        return resolved

    return {
        "error": {
            "type": "unknown_download",
            "message": (
                f"Paper '{requested_id}' was not returned by "
                "download_paper in the current task."
            ),
        }
    }


def run_agent(
    state: dict,
    max_steps: int = 5,
    confirm_save: Callable[[dict], bool] | None = None,
    on_event: EventHandler | None = None,
) -> str:
    """Run one conversational turn until a final answer or the step limit."""
    steps_this_turn = 0
    search_queries_this_turn = set()

    while steps_this_turn < max_steps:
        turn_step = steps_this_turn + 1
        next_total_step = state["step"] + 1
        _emit_event(
            on_event,
            {
                "kind": "llm_started",
                "turn_step": turn_step,
                "total_step": next_total_step,
            },
        )
        llm_started_at = perf_counter()
        try:
            response = decide_next_action(state)
        except KeyboardInterrupt:
            cancelled_answer = "当前模型调用已取消。"
            state["messages"].append(
                {"role": "assistant", "content": cancelled_answer}
            )
            _emit_event(
                on_event,
                {
                    "kind": "turn_finished",
                    "turn_step": turn_step,
                    "total_step": state["step"],
                    "status": "cancelled",
                },
            )
            return cancelled_answer
        except Exception as error:
            _emit_event(
                on_event,
                {
                    "kind": "run_failed",
                    "turn_step": turn_step,
                    "total_step": next_total_step,
                    "duration_ms": _elapsed_ms(llm_started_at),
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            raise
        steps_this_turn += 1
        state["step"] += 1
        _emit_event(
            on_event,
            {
                "kind": "llm_finished",
                "turn_step": turn_step,
                "total_step": state["step"],
                "duration_ms": _elapsed_ms(llm_started_at),
                "response_type": response.get("type", "unknown"),
            },
        )

        if response["type"] == "final":
            final_answer = response["content"]
            state["messages"].append(
                {"role": "assistant", "content": final_answer}
            )
            _emit_event(
                on_event,
                {
                    "kind": "turn_finished",
                    "turn_step": turn_step,
                    "total_step": state["step"],
                    "status": "completed",
                },
            )
            return final_answer

        if response["type"] != "tool_call":
            error = ValueError(
                f"Unknown LLM response type: {response['type']}"
            )
            _emit_event(
                on_event,
                {
                    "kind": "run_failed",
                    "turn_step": turn_step,
                    "total_step": state["step"],
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            raise error

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
        _emit_event(
            on_event,
            {
                "kind": "tool_started",
                "turn_step": turn_step,
                "total_step": state["step"],
                "tool_name": tool_name,
                "arguments": tool_arguments,
            },
        )

        tool_started_at = perf_counter()
        tool_cancelled = False
        try:
            observation = _search_guard_result(
                tool_name,
                tool_arguments,
                search_queries_this_turn,
            )
            if observation is None:
                observation = execute_tool(
                    tool_name,
                    tool_arguments,
                    state=state,
                    confirm_save=confirm_save,
                )
        except KeyboardInterrupt:
            tool_cancelled = True
            observation = {
                "error": {
                    "type": "cancelled",
                    "message": f"Tool '{tool_name}' was cancelled by the user.",
                }
            }
        state["messages"].append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": observation,
            }
        )
        _emit_event(
            on_event,
            {
                "kind": "tool_finished",
                "turn_step": turn_step,
                "total_step": state["step"],
                "duration_ms": _elapsed_ms(tool_started_at),
                "tool_name": tool_name,
                "arguments": tool_arguments,
                "result": observation,
            },
        )
        if tool_cancelled:
            cancelled_answer = "当前工具操作已取消。"
            state["messages"].append(
                {"role": "assistant", "content": cancelled_answer}
            )
            _emit_event(
                on_event,
                {
                    "kind": "turn_finished",
                    "turn_step": turn_step,
                    "total_step": state["step"],
                    "status": "cancelled",
                },
            )
            return cancelled_answer

    stopped_answer = f"Agent stopped after reaching max_steps={max_steps}."
    state["messages"].append(
        {"role": "assistant", "content": stopped_answer}
    )
    _emit_event(
        on_event,
        {
            "kind": "turn_finished",
            "turn_step": steps_this_turn,
            "total_step": state["step"],
            "status": "max_steps_reached",
        },
    )
    return stopped_answer


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1_000, 2)


def _emit_event(
    on_event: EventHandler | None,
    event: AgentEvent,
) -> None:
    """Notify an optional observer without giving it control of the loop."""
    if on_event is None:
        return
    try:
        on_event(deepcopy(event))
    except Exception:
        return


def _search_guard_result(
    tool_name: str,
    arguments: object,
    search_queries_this_turn: set[str],
) -> dict | None:
    """Reject duplicate or excessive searches within one user turn."""
    if tool_name != "search_paper" or not isinstance(arguments, dict):
        return None
    if set(arguments) != {"query"}:
        return None

    query = arguments["query"]
    if not isinstance(query, str):
        return None
    normalized_query = " ".join(query.split())
    if not normalized_query:
        return None

    if normalized_query in search_queries_this_turn:
        return {
            "error": {
                "type": "duplicate_search_query",
                "message": (
                    f"Search query '{normalized_query}' was already "
                    "attempted in the current user turn. Use a refined "
                    "query or return a final answer."
                ),
            }
        }

    if len(search_queries_this_turn) >= MAX_SEARCH_ATTEMPTS_PER_TURN:
        return {
            "error": {
                "type": "search_limit_reached",
                "message": (
                    "The current user turn already attempted "
                    f"{MAX_SEARCH_ATTEMPTS_PER_TURN} different search "
                    "queries. Ask the user for clarification before "
                    "searching again."
                ),
                "limit": MAX_SEARCH_ATTEMPTS_PER_TURN,
            }
        }

    search_queries_this_turn.add(normalized_query)
    return None
