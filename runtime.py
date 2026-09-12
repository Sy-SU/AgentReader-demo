import inspect
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from agent import decide_next_action as decide_agent_action
from checkpoint import clear_checkpoint, save_checkpoint
from executor import decide_step_action as decide_plan_step
from events import AgentEvent, EventHandler
from planner import decide_initial_action
from planning import (
    MAX_PLAN_STEPS,
    MAX_TASK_LLM_STEPS,
    MAX_TASK_REPLANS,
    MAX_TASK_TOOL_CALLS,
    block_current_step,
    cancel_plan,
    complete_current_step,
    create_plan,
    fail_current_step,
    fail_plan,
    record_current_step_evidence,
    record_plan_usage,
    resume_blocked_step,
    revise_plan,
    start_next_step,
    validate_plan,
)
from tools import (
    download_paper,
    extract_paper_text,
    list_library,
    retrieve_paper_chunks,
    save_paper,
    search_paper,
)
from tools.download import validate_cached_pdf_path


TOOL_REGISTRY = {
    "search_paper": search_paper,
    "save_paper": save_paper,
    "list_library": list_library,
    "download_paper": download_paper,
    "extract_paper_text": extract_paper_text,
    "retrieve_paper_chunks": retrieve_paper_chunks,
}
MAX_SEARCH_ATTEMPTS_PER_TURN = 2
DEFAULT_MAX_STEPS_PER_TURN = 20
SIDE_EFFECT_TOOLS = frozenset({"save_paper", "download_paper"})


def decide_next_action(state: dict, *, allow_planning: bool = False) -> dict:
    """Route the first eligible decision through Planner, then use V2 Agent."""
    if allow_planning:
        return decide_initial_action(state)
    return decide_agent_action(state)


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


def cancel_active_task(
    state: dict,
    checkpoint_file: str | Path | None = None,
    on_event: EventHandler | None = None,
) -> dict:
    """Cancel one trusted active Plan and remove its recoverable checkpoint."""
    active_plan = _active_plan(state)
    if active_plan is None:
        raise ValueError("The current State has no active task Plan.")

    cancelled_plan = cancel_plan(active_plan)
    _clear_task_checkpoint(checkpoint_file)
    state["plan"] = cancelled_plan
    _emit_plan_step_changes(on_event, state, active_plan, turn_step=0)
    _emit_event(
        on_event,
        {
            "kind": "task_cancelled",
            "turn_step": 0,
            "total_step": state["step"],
            "task_id": cancelled_plan["task_id"],
            "plan_revision": cancelled_plan["revision"],
            "status": "cancelled",
        },
    )
    return deepcopy(cancelled_plan)


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
    max_steps: int = DEFAULT_MAX_STEPS_PER_TURN,
    confirm_save: Callable[[dict], bool] | None = None,
    on_event: EventHandler | None = None,
    checkpoint_file: str | Path | None = None,
) -> str:
    """Run one conversational turn until a final answer or the step limit."""
    active_plan = _active_plan(state)
    if active_plan is not None and active_plan["status"] == "blocked":
        if _latest_message_role(state) == "user":
            previous_plan = active_plan
            state["plan"] = resume_blocked_step(active_plan)
            active_plan = state["plan"]
            _emit_plan_step_changes(
                on_event,
                state,
                previous_plan,
                turn_step=0,
            )
            _persist_active_checkpoint(
                state,
                checkpoint_file,
                on_event=on_event,
                turn_step=0,
            )
        else:
            answer = _format_plan_answer(active_plan, created=False)
            state["messages"].append(
                {"role": "assistant", "content": answer}
            )
            _persist_active_checkpoint(
                state,
                checkpoint_file,
                on_event=on_event,
                turn_step=0,
            )
            _emit_event(
                on_event,
                {
                    "kind": "turn_finished",
                    "turn_step": 0,
                    "total_step": state["step"],
                    "status": "blocked",
                },
            )
            return answer
    plan_execution = active_plan is not None

    steps_this_turn = 0
    search_queries_this_turn = set()

    while steps_this_turn < max_steps:
        if (
            plan_execution
            and state["plan"]["budget"]["llm_steps"]
            >= MAX_TASK_LLM_STEPS
        ):
            return _finish_plan_at_budget_limit(
                state,
                budget_name="LLM 决策",
                limit=MAX_TASK_LLM_STEPS,
                turn_step=steps_this_turn,
                on_event=on_event,
                checkpoint_file=checkpoint_file,
            )
        if plan_execution and state["plan"]["current_step_id"] is None:
            previous_plan = state["plan"]
            state["plan"] = start_next_step(state["plan"])
            _emit_plan_step_changes(
                on_event,
                state,
                previous_plan,
                turn_step=steps_this_turn,
            )
            _persist_active_checkpoint(
                state,
                checkpoint_file,
                on_event=on_event,
                turn_step=steps_this_turn,
            )

        turn_step = steps_this_turn + 1
        next_total_step = state["step"] + 1
        plan_issue = (
            _latest_plan_issue(state) if plan_execution else None
        )
        replan_reason = None
        if (
            plan_issue is not None
            and _plan_can_replan(state["plan"])
        ):
            replan_reason = plan_issue["reason"]
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
            if plan_execution:
                response = decide_plan_step(
                    state,
                    state["plan"],
                    replan_reason=replan_reason,
                )
            else:
                response = decide_next_action(
                    state,
                    allow_planning=steps_this_turn == 0,
                )
        except KeyboardInterrupt:
            cancelled_answer = "当前模型调用已取消。"
            state["messages"].append(
                {"role": "assistant", "content": cancelled_answer}
            )
            if plan_execution:
                _persist_active_checkpoint(
                    state,
                    checkpoint_file,
                    on_event=on_event,
                    turn_step=turn_step,
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
        if plan_execution:
            state["plan"] = record_plan_usage(
                state["plan"],
                llm_steps=1,
            )
            _persist_active_checkpoint(
                state,
                checkpoint_file,
                on_event=on_event,
                turn_step=turn_step,
            )
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

        if response["type"] == "plan":
            try:
                plan = _create_runtime_plan(state, response)
            except Exception as error:
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
                raise
            answer = _format_plan_answer(plan, created=True)
            state["messages"].append(
                _assistant_message(response, content=answer)
            )
            _emit_event(
                on_event,
                {
                    "kind": "plan_created",
                    "turn_step": turn_step,
                    "total_step": state["step"],
                    "task_id": plan["task_id"],
                    "plan_revision": plan["revision"],
                    "status": plan["status"],
                    "step_count": len(plan["steps"]),
                },
            )
            _persist_active_checkpoint(
                state,
                checkpoint_file,
                on_event=on_event,
                turn_step=turn_step,
            )
            _emit_event(
                on_event,
                {
                    "kind": "turn_finished",
                    "turn_step": turn_step,
                    "total_step": state["step"],
                    "status": "planned",
                },
            )
            return answer

        if response["type"] == "replan":
            if not plan_execution or replan_reason is None:
                raise ValueError(
                    "Replan is not authorized for the latest observation."
                )
            previous_plan = state["plan"]
            failed = fail_current_step(previous_plan)
            state["plan"] = revise_plan(
                failed,
                response["step_descriptions"],
            )
            state["messages"].append(
                _assistant_message(
                    response,
                    content=_format_replan_answer(state["plan"]),
                )
            )
            _emit_plan_step_changes(
                on_event,
                state,
                previous_plan,
                turn_step=turn_step,
            )
            _emit_event(
                on_event,
                {
                    "kind": "plan_replanned",
                    "turn_step": turn_step,
                    "total_step": state["step"],
                    "task_id": state["plan"]["task_id"],
                    "plan_revision": state["plan"]["revision"],
                    "status": state["plan"]["status"],
                    "step_count": len(state["plan"]["steps"]),
                    "replan_count": state["plan"]["budget"]["replans"],
                    "reason": replan_reason,
                },
            )
            _persist_active_checkpoint(
                state,
                checkpoint_file,
                on_event=on_event,
                turn_step=turn_step,
            )
            continue

        if response["type"] == "blocked":
            if not plan_execution:
                raise ValueError(
                    "Only an active Plan can request clarification."
                )
            answer = response["content"]
            previous_plan = state["plan"]
            state["plan"] = block_current_step(state["plan"])
            state["messages"].append(
                _assistant_message(response, content=answer)
            )
            _emit_plan_step_changes(
                on_event,
                state,
                previous_plan,
                turn_step=turn_step,
            )
            _emit_task_blocked(
                on_event,
                state,
                turn_step=turn_step,
                reason="user clarification is required",
            )
            _persist_active_checkpoint(
                state,
                checkpoint_file,
                on_event=on_event,
                turn_step=turn_step,
            )
            _emit_event(
                on_event,
                {
                    "kind": "turn_finished",
                    "turn_step": turn_step,
                    "total_step": state["step"],
                    "status": "blocked",
                },
            )
            return answer

        if response["type"] == "final":
            final_answer = response["content"]
            state["messages"].append(
                _assistant_message(response, content=final_answer)
            )
            if plan_execution:
                if (
                    plan_issue is not None
                    and plan_issue["block_final"]
                ):
                    previous_plan = state["plan"]
                    state["plan"] = block_current_step(state["plan"])
                    _emit_plan_step_changes(
                        on_event,
                        state,
                        previous_plan,
                        turn_step=turn_step,
                    )
                    _emit_task_blocked(
                        on_event,
                        state,
                        turn_step=turn_step,
                        reason=plan_issue["reason"],
                    )
                    _persist_active_checkpoint(
                        state,
                        checkpoint_file,
                        on_event=on_event,
                        turn_step=turn_step,
                    )
                    _emit_event(
                        on_event,
                        {
                            "kind": "turn_finished",
                            "turn_step": turn_step,
                            "total_step": state["step"],
                            "status": "blocked",
                        },
                    )
                    return final_answer
                previous_plan = state["plan"]
                state["plan"] = complete_current_step(state["plan"])
                _emit_plan_step_changes(
                    on_event,
                    state,
                    previous_plan,
                    turn_step=turn_step,
                )
                if state["plan"]["status"] != "completed":
                    _persist_active_checkpoint(
                        state,
                        checkpoint_file,
                        on_event=on_event,
                        turn_step=turn_step,
                    )
                    continue
                _clear_task_checkpoint(checkpoint_file)
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

        if (
            plan_execution
            and state["plan"]["budget"]["tool_calls"]
            >= MAX_TASK_TOOL_CALLS
        ):
            return _finish_plan_at_budget_limit(
                state,
                budget_name="Tool 执行",
                limit=MAX_TASK_TOOL_CALLS,
                turn_step=turn_step,
                on_event=on_event,
                checkpoint_file=checkpoint_file,
            )

        state["messages"].append(
            _assistant_message(
                response,
                content=response.get("content"),
                tool_call={
                    "id": tool_call_id,
                    "name": tool_name,
                    "arguments": tool_arguments,
                },
            )
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
            observation = None
            if plan_execution:
                observation = _reuse_completed_side_effect(
                    tool_name,
                    tool_arguments,
                    state,
                )
            if observation is None:
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
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": observation,
        }
        if plan_execution:
            evidence_ref = (
                "tool-result-"
                f"{state['plan']['budget']['tool_calls'] + 1:03d}"
            )
            updated_plan = record_plan_usage(
                state["plan"],
                tool_calls=1,
            )
            updated_plan = record_current_step_evidence(
                updated_plan,
                [evidence_ref],
            )
            tool_message["evidence_ref"] = evidence_ref
            state["plan"] = updated_plan
        state["messages"].append(tool_message)
        if plan_execution:
            _persist_active_checkpoint(
                state,
                checkpoint_file,
                on_event=on_event,
                turn_step=turn_step,
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
            if plan_execution:
                _persist_active_checkpoint(
                    state,
                    checkpoint_file,
                    on_event=on_event,
                    turn_step=turn_step,
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

    if plan_execution:
        stopped_answer = (
            "本轮已达到模型决策上限"
            f"（max_steps={max_steps}），当前计划仍在进行中。"
            "请输入“继续”从当前步骤恢复执行。"
        )
    else:
        stopped_answer = (
            "本轮已达到模型决策上限"
            f"（max_steps={max_steps}），Agent 已停止本轮执行。"
        )
    state["messages"].append(
        {
            "role": "assistant",
            "content": stopped_answer,
            "runtime_control": "max_steps_reached",
        }
    )
    if plan_execution:
        _persist_active_checkpoint(
            state,
            checkpoint_file,
            on_event=on_event,
            turn_step=steps_this_turn,
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


def _create_runtime_plan(state: dict, action: dict) -> dict:
    """Create trusted Plan fields from one normalized internal plan action."""
    required_fields = {
        "type",
        "content",
        "tool_call_id",
        "step_descriptions",
    }
    allowed_fields = {
        *required_fields,
        "reasoning_content",
        "reasoning_details",
    }
    if (
        not required_fields <= set(action)
        or not set(action) <= allowed_fields
        or action.get("type") != "plan"
    ):
        raise ValueError("Invalid internal plan action contract.")
    _validate_reasoning_fields(action)
    if action.get("content") is not None:
        raise ValueError("An internal plan action cannot contain text.")
    if not isinstance(action.get("tool_call_id"), str) or not action[
        "tool_call_id"
    ]:
        raise ValueError("An internal plan action requires a call ID.")
    if state.get("plan") is not None or state.get("task_id") is not None:
        raise ValueError("The current State already contains a task Plan.")

    task_id = f"task-{uuid4().hex}"
    plan = create_plan(
        task_id=task_id,
        goal=_latest_user_goal(state),
        step_descriptions=action["step_descriptions"],
    )
    plan = record_plan_usage(plan, llm_steps=1)

    state["task_id"] = task_id
    state["plan"] = plan
    return plan


def _latest_user_goal(state: dict) -> str:
    """Return the user message that caused the current planning decision."""
    for message in reversed(state.get("messages", [])):
        content = message.get("content") if isinstance(message, dict) else None
        if (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(content, str)
            and content.strip()
        ):
            return content.strip()
    raise ValueError("Cannot create a Plan without a user goal.")


def _active_plan(state: dict) -> dict | None:
    """Return a validated active Plan without allowing silent replacement."""
    plan = state.get("plan")
    if plan is None:
        return None
    validate_plan(plan)
    if state.get("task_id") != plan["task_id"]:
        raise ValueError("State task_id does not match its Plan.")
    if plan["status"] in {"running", "blocked"}:
        return plan
    return None


def _format_plan_answer(plan: dict, *, created: bool) -> str:
    """Render a bounded Plan preview for creation or a blocked task."""
    if created:
        heading = "已创建任务计划："
    else:
        heading = (
            "当前任务处于阻塞状态，"
            "请先补充完成当前步骤所需的信息："
        )
    lines = [heading]
    lines.extend(
        f"{index}. {step['description']}"
        for index, step in enumerate(plan["steps"], start=1)
    )
    return "\n".join(lines)


def _format_replan_answer(plan: dict) -> str:
    """Render an internal Assistant message for the revised plan history."""
    pending = [
        step["description"]
        for step in plan["steps"]
        if step["status"] == "pending"
    ]
    lines = [f"任务计划已修订（revision {plan['revision']}）："]
    lines.extend(
        f"{index}. {description}"
        for index, description in enumerate(pending, start=1)
    )
    return "\n".join(lines)


def _latest_message_role(state: dict) -> str | None:
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[-1], dict):
        return None
    role = messages[-1].get("role")
    return role if isinstance(role, str) else None


def _latest_plan_issue(state: dict) -> dict | None:
    """Classify the latest unhandled Tool Result for trusted plan control."""
    tool_message = _latest_unhandled_tool_message(state)
    if tool_message is None:
        return None

    tool_name = tool_message.get("name")
    result = tool_message.get("content")
    if not isinstance(result, dict):
        return {
            "reason": "the latest tool returned an invalid result",
            "block_final": True,
        }

    error = result.get("error")
    if isinstance(error, dict):
        error_type = error.get("type")
        label = error_type if isinstance(error_type, str) else "unknown"
        return {
            "reason": f"the latest tool failed with {label}",
            "block_final": True,
        }

    if tool_name == "search_paper":
        assessment = result.get("relevance_assessment")
        status = (
            assessment.get("status")
            if isinstance(assessment, dict)
            else None
        )
        if result.get("found") is False or status in {
            "no_candidates",
            "no_lexical_match",
        }:
            return {
                "reason": "the paper search produced no usable candidate",
                "block_final": True,
            }
        papers = result.get("papers")
        if isinstance(papers, list) and len(papers) > 1:
            has_exact_title = any(
                isinstance(paper, dict)
                and isinstance(paper.get("local_relevance"), dict)
                and paper["local_relevance"].get("exact_title_match") is True
                for paper in papers
            )
            if not has_exact_title:
                return {
                    "reason": "the paper search returned ambiguous candidates",
                    "block_final": False,
                }

    if tool_name == "retrieve_paper_chunks" and result.get("found") is False:
        reason = (
            "the retrieved passages have insufficient query coverage"
            if result.get("rejected_low_query_coverage") is True
            else "the paper retrieval produced no matching passage"
        )
        return {"reason": reason, "block_final": True}

    if tool_name == "extract_paper_text" and result.get(
        "text_available"
    ) is False:
        return {
            "reason": "the downloaded PDF contains no extractable text",
            "block_final": True,
        }

    if tool_name == "list_library" and result.get("count") == 0:
        return {
            "reason": "the local paper library is empty",
            "block_final": True,
        }
    return None


def _latest_unhandled_tool_message(state: dict) -> dict | None:
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if role == "user":
            continue
        if role == "assistant":
            if message.get("runtime_control") == "max_steps_reached":
                continue
            return None
        if role == "tool":
            return message
        return None
    return None


def _plan_can_replan(plan: dict) -> bool:
    validate_plan(plan)
    if plan["status"] != "running" or plan["current_step_id"] is None:
        return False
    if plan["budget"]["replans"] >= MAX_TASK_REPLANS:
        return False
    preserved_after_failure = sum(
        step["status"] in {"completed", "failed", "running"}
        for step in plan["steps"]
    )
    return preserved_after_failure < MAX_PLAN_STEPS


def _assistant_message(
    response: dict,
    *,
    content: str | None,
    tool_call: dict | None = None,
) -> dict:
    """Build an Assistant State message with opaque reasoning metadata."""
    _validate_reasoning_fields(response)
    message = {"role": "assistant", "content": content}
    if tool_call is not None:
        message["tool_call"] = tool_call
    if "reasoning_details" in response:
        message["reasoning_details"] = deepcopy(response["reasoning_details"])
    elif "reasoning_content" in response:
        message["reasoning_content"] = response["reasoning_content"]
    return message


def _validate_reasoning_fields(response: dict) -> None:
    present = {
        field
        for field in ("reasoning_content", "reasoning_details")
        if field in response
    }
    if len(present) > 1:
        raise ValueError(
            "A normalized response cannot contain two reasoning representations."
        )
    if "reasoning_content" in present and not isinstance(
        response["reasoning_content"], str
    ):
        raise ValueError("reasoning_content must be text.")
    if "reasoning_details" in present and not isinstance(
        response["reasoning_details"], list
    ):
        raise ValueError("reasoning_details must be a list.")


def _finish_plan_at_budget_limit(
    state: dict,
    *,
    budget_name: str,
    limit: int,
    turn_step: int,
    on_event: EventHandler | None,
    checkpoint_file: str | Path | None,
) -> str:
    """Fail an active Plan before exceeding a trusted Runtime budget."""
    previous_plan = state["plan"]
    state["plan"] = fail_plan(state["plan"])
    answer = f"任务已停止：已达到 {budget_name}上限（{limit}）。"
    state["messages"].append({"role": "assistant", "content": answer})
    _clear_task_checkpoint(checkpoint_file)
    _emit_plan_step_changes(
        on_event,
        state,
        previous_plan,
        turn_step=turn_step,
    )
    _emit_event(
        on_event,
        {
            "kind": "turn_finished",
            "turn_step": turn_step,
            "total_step": state["step"],
            "status": "failed",
        },
    )
    return answer


def _persist_active_checkpoint(
    state: dict,
    checkpoint_file: str | Path | None,
    *,
    on_event: EventHandler | None = None,
    turn_step: int = 0,
) -> None:
    """Persist one coherent active Plan boundary when configured by the UI."""
    if checkpoint_file is None:
        return
    result = save_checkpoint(state, checkpoint_file)
    plan = state["plan"]
    _emit_event(
        on_event,
        {
            "kind": "checkpoint_saved",
            "turn_step": turn_step,
            "total_step": state["step"],
            "task_id": plan["task_id"],
            "plan_revision": plan["revision"],
            "status": plan["status"],
            "checkpoint_size_bytes": result["size_bytes"],
        },
    )


def _emit_plan_step_changes(
    on_event: EventHandler | None,
    state: dict,
    previous_plan: dict,
    *,
    turn_step: int,
) -> None:
    """Emit bounded observations for Runtime-owned step transitions."""
    previous_steps = {
        step["id"]: step for step in previous_plan["steps"]
    }
    plan = state["plan"]
    for step in plan["steps"]:
        previous_step = previous_steps.get(step["id"])
        if (
            previous_step is None
            or previous_step["status"] == step["status"]
        ):
            continue
        _emit_event(
            on_event,
            {
                "kind": "plan_step_changed",
                "turn_step": turn_step,
                "total_step": state["step"],
                "task_id": plan["task_id"],
                "plan_revision": plan["revision"],
                "step_id": step["id"],
                "step_description": step["description"],
                "previous_status": previous_step["status"],
                "status": step["status"],
            },
        )


def _emit_task_blocked(
    on_event: EventHandler | None,
    state: dict,
    *,
    turn_step: int,
    reason: str,
) -> None:
    plan = state["plan"]
    _emit_event(
        on_event,
        {
            "kind": "task_blocked",
            "turn_step": turn_step,
            "total_step": state["step"],
            "task_id": plan["task_id"],
            "plan_revision": plan["revision"],
            "step_id": plan["current_step_id"],
            "status": plan["status"],
            "reason": reason,
        },
    )


def _clear_task_checkpoint(checkpoint_file: str | Path | None) -> None:
    """Clear a terminal task checkpoint when persistence is configured."""
    if checkpoint_file is None:
        return
    clear_checkpoint(checkpoint_file)


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


def _reuse_completed_side_effect(
    tool_name: str,
    arguments: object,
    state: dict,
) -> dict | None:
    """Reuse a completed write/download result without repeating its effect."""
    if tool_name not in SIDE_EFFECT_TOOLS or not isinstance(arguments, dict):
        return None

    messages = state.get("messages", [])
    for index in range(len(messages) - 2, -1, -1):
        call_message = messages[index]
        if not isinstance(call_message, dict):
            continue
        tool_call = call_message.get("tool_call")
        if not isinstance(tool_call, dict):
            continue
        if (
            tool_call.get("name") != tool_name
            or tool_call.get("arguments") != arguments
        ):
            continue
        if index + 1 >= len(messages):
            continue

        result_message = messages[index + 1]
        if not isinstance(result_message, dict):
            continue
        if (
            result_message.get("role") != "tool"
            or result_message.get("name") != tool_name
            or result_message.get("tool_call_id") != tool_call.get("id")
        ):
            continue
        result = result_message.get("content")
        if not _is_completed_side_effect_result(tool_name, result):
            continue

        reused = deepcopy(result)
        reused["runtime_reused"] = True
        reused["runtime_reuse_reason"] = "duplicate_side_effect"
        return reused
    return None


def _is_completed_side_effect_result(tool_name: str, result: object) -> bool:
    if not isinstance(result, dict) or "error" in result:
        return False
    if tool_name == "save_paper":
        return result.get("saved") is True or result.get("reason") == (
            "already_exists"
        )
    if tool_name == "download_paper":
        completed = (
            isinstance(result.get("paper_id"), str)
            and isinstance(result.get("local_path"), str)
            and (
                result.get("downloaded") is True
                or result.get("cached") is True
            )
        )
        if not completed:
            return False
        try:
            validate_cached_pdf_path(result["local_path"])
        except (OSError, RuntimeError, ValueError):
            return False
        return True
    return False
