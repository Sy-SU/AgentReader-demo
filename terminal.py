"""Interactive terminal presentation for AgentReader."""

import json
import os
import sys
from collections.abc import Callable
from typing import Any, TextIO

from events import AgentEvent


EXIT_COMMANDS = {"exit", "quit", "退出"}
SAVE_APPROVALS = {"y", "yes", "是", "确认"}
DEBUG_TEXT_PREVIEW_CHARS = 400
SLASH_COMMANDS = (
    "/help",
    "/plan",
    "/cancel",
    "/debug",
    "/clear",
    "/exit",
)


class TerminalUI:
    """Render Runtime events and collect interactive user input."""

    def __init__(
        self,
        debug: bool = False,
        plain: bool = False,
        input_func: Callable[[str], str] | None = None,
        output: TextIO | None = None,
    ) -> None:
        self.debug = debug
        self.output = output or sys.stdout
        self._input_func = input_func
        self._status = None
        self._failure_reported = False
        self.interactive = input_func is not None or sys.stdin.isatty()
        self.enhanced = (
            not plain
            and input_func is None
            and self.interactive
            and self.output.isatty()
        )
        self._console = None
        self._session = None

        if self.enhanced:
            self._initialize_enhanced_terminal()

    def _initialize_enhanced_terminal(self) -> None:
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter
            from prompt_toolkit.history import InMemoryHistory
            from prompt_toolkit.styles import Style
            from rich.console import Console
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "交互式终端依赖 prompt-toolkit 和 rich。请运行 "
                "'conda env update -f environment.yml --prune'。"
            ) from error

        style = Style.from_dict({"prompt": "bold ansicyan"})
        completer = WordCompleter(list(SLASH_COMMANDS), sentence=True)
        self._session = PromptSession(
            history=InMemoryHistory(),
            completer=completer,
            complete_while_typing=False,
            style=style,
        )
        self._console = Console(file=self.output)

    def print_banner(self) -> None:
        provider = os.getenv("LLM_PROVIDER", "fake")
        if self._console is not None:
            self._console.print("AgentReader", style="bold cyan")
            self._console.print(
                f"{provider} · 输入 /help 查看命令",
                style="dim",
                markup=False,
            )
            return
        self._write(f"AgentReader · {provider} · 输入 /help 查看命令")

    def print_thinking_mode(self, max_steps: int) -> None:
        self._write(
            "Thinking 模式已开启 · Provider reasoning=enabled · "
            f"单轮 max_steps={max_steps}"
        )

    def read_input(self) -> str:
        if self._session is not None:
            return self._session.prompt([("class:prompt", "You › ")])
        if self._input_func is not None:
            return self._input_func("You › ")
        return input("You › ")

    def begin_turn(self) -> None:
        self._failure_reported = False

    def handle_event(self, event: AgentEvent) -> None:
        kind = event["kind"]
        if kind == "llm_started":
            self._handle_llm_started(event)
        elif kind == "llm_finished":
            self._handle_llm_finished(event)
        elif kind == "tool_started":
            self._handle_tool_started(event)
        elif kind == "tool_finished":
            self._handle_tool_finished(event)
        elif kind == "turn_finished":
            self._stop_status()
        elif kind == "run_failed":
            self._stop_status()
            self._failure_reported = True
            self.print_error(event.get("message", "Agent 运行失败。"))
        elif kind == "plan_created":
            self._handle_plan_created(event)
        elif kind == "plan_step_changed":
            self._handle_plan_step_changed(event)
        elif kind == "plan_replanned":
            self._handle_plan_replanned(event)
        elif kind == "task_blocked":
            self._handle_task_blocked(event)
        elif kind == "task_cancelled":
            self._handle_task_cancelled(event)
        elif kind == "checkpoint_saved":
            self._handle_checkpoint_saved(event)

    def _handle_llm_started(self, event: AgentEvent) -> None:
        if self.debug:
            self._write(f"[Step {event['turn_step']} | LLM] 正在思考…")
            return
        if self._console is not None:
            self._stop_status()
            self._status = self._console.status(
                "Agent 正在思考…", spinner="dots"
            )
            self._status.start()
            return
        self._write("… Agent 正在思考")

    def _handle_llm_finished(self, event: AgentEvent) -> None:
        self._stop_status()
        if not self.debug:
            return
        duration = _duration_text(event.get("duration_ms"))
        self._write(
            f"[Step {event['turn_step']} | LLM] "
            f"{event.get('response_type', 'unknown')}{duration}"
        )

    def _handle_tool_started(self, event: AgentEvent) -> None:
        name = event.get("tool_name", "unknown_tool")
        arguments = event.get("arguments", {})
        if self.debug:
            self._write(f"[Step {event['turn_step']} | Tool] start")
            self._write(f"  name: {name}")
            self._write(
                "  arguments: " + json.dumps(arguments, ensure_ascii=False)
            )
            return

        detail = _tool_argument_summary(name, arguments)
        suffix = f" · {detail}" if detail else ""
        self._write(f"● {name}{suffix}")

    def _handle_tool_finished(self, event: AgentEvent) -> None:
        name = event.get("tool_name", "unknown_tool")
        result = event.get("result", {})
        duration = _duration_text(event.get("duration_ms"))
        if self.debug:
            display_result = _debug_tool_content(name, result)
            self._write(
                f"[Step {event['turn_step']} | Tool] result{duration}"
            )
            self._write(f"  name: {name}")
            self._write(
                "  content: "
                + json.dumps(display_result, ensure_ascii=False)
            )
            return

        marker = "✗" if isinstance(result, dict) and "error" in result else "✓"
        summary = _tool_result_summary(name, result)
        suffix = f" · {summary}" if summary else ""
        self._write(f"{marker} {name}{suffix}{duration}")

    def _handle_plan_created(self, event: AgentEvent) -> None:
        self._stop_status()
        if self.debug:
            self._write(
                "[Plan] created · "
                f"revision {event.get('plan_revision')} · "
                f"{event.get('step_count')} steps"
            )

    def _handle_plan_step_changed(self, event: AgentEvent) -> None:
        self._stop_status()
        step_id = event.get("step_id", "unknown-step")
        status = event.get("status", "unknown")
        description = _shorten(event.get("step_description", ""), 160)
        if self.debug:
            self._write(
                f"[Plan] {step_id} · "
                f"{event.get('previous_status', 'unknown')} → {status} · "
                f"{description}"
            )
            return
        marker = {
            "running": "○",
            "completed": "✓",
            "blocked": "!",
            "failed": "✗",
        }.get(status, "·")
        self._write(f"{marker} {step_id} · {status} · {description}")

    def _handle_plan_replanned(self, event: AgentEvent) -> None:
        self._stop_status()
        self._write(
            "↻ Plan revised · "
            f"revision {event.get('plan_revision')} · "
            f"replans {event.get('replan_count')}"
        )

    def _handle_task_blocked(self, event: AgentEvent) -> None:
        self._stop_status()
        if self.debug:
            self._write(
                "[Plan] blocked · "
                f"{_shorten(event.get('reason', ''), 200)}"
            )

    def _handle_task_cancelled(self, event: AgentEvent) -> None:
        self._stop_status()
        if self.debug:
            self._write(
                "[Plan] cancelled · "
                f"revision {event.get('plan_revision')}"
            )

    def _handle_checkpoint_saved(self, event: AgentEvent) -> None:
        self._stop_status()
        if self.debug:
            self._write(
                "[Checkpoint] saved · "
                f"{event.get('checkpoint_size_bytes', 0)} bytes"
            )

    def print_answer(self, answer: str) -> None:
        self._stop_status()
        if self._console is not None:
            from rich.markdown import Markdown

            self._console.print("\nAgent ›", style="bold green")
            self._console.print(Markdown(answer))
            return
        self._write("\nAgent ›")
        self._write(answer)

    def print_help(self) -> None:
        self._write(
            "\n可用命令：\n"
            "  /help           显示帮助\n"
            "  /plan           显示当前任务计划\n"
            "  /cancel         取消当前计划任务\n"
            "  /debug          切换 Debug 模式\n"
            "  /debug on|off   开启或关闭 Debug 模式\n"
            "  /clear          清空当前会话 State\n"
            "  /exit           退出 AgentReader\n"
            "\n普通文本 exit、quit、退出也可以退出。"
        )

    def set_debug(self, value: bool) -> None:
        self.debug = value
        status = "开启" if value else "关闭"
        self._write(f"Debug 模式已{status}。")

    def clear_conversation(self) -> None:
        self._stop_status()
        if self._console is not None:
            self._console.clear()
            self.print_banner()
        self._write("当前会话 State 已清空。")

    def print_active_task_clear_rejected(self) -> None:
        self._stop_status()
        self._write(
            "当前存在未完成的计划任务，不能使用 /clear。"
            "请先继续任务或使用 /cancel 明确取消。"
        )

    def print_plan(self, plan: dict) -> None:
        self._stop_status()
        current_step_id = plan.get("current_step_id")
        current_step = next(
            (
                step
                for step in plan.get("steps", [])
                if step.get("id") == current_step_id
            ),
            None,
        )
        next_step = next(
            (
                step
                for step in plan.get("steps", [])
                if step.get("status") == "pending"
            ),
            None,
        )
        current_summary = (
            f"{current_step_id} · "
            f"{_shorten(current_step['description'], 160)}"
            if current_step is not None
            else (
                "尚未启动"
                if next_step is not None
                else "已经结束"
            )
        )
        next_summary = (
            f"{next_step.get('id')} · "
            f"{_shorten(str(next_step.get('description', '')), 160)}"
            if next_step is not None
            else "无"
        )
        lines = [
            "\n当前任务计划：",
            f"  目标：{_shorten(str(plan.get('goal', '')), 200)}",
            f"  状态：{plan.get('status', 'unknown')}",
            f"  Revision：{plan.get('revision', 'unknown')}",
            f"  当前步骤：{current_summary}",
            f"  下一步骤：{next_summary}",
            "  步骤：",
        ]
        for index, step in enumerate(plan.get("steps", [])[:8], start=1):
            marker = "→" if step.get("id") == current_step_id else " "
            lines.append(
                f"  {marker} {index}. [{step.get('status', 'unknown')}] "
                f"{_shorten(str(step.get('description', '')), 200)}"
            )
        self._write("\n".join(lines))

    def print_no_plan(self) -> None:
        self._stop_status()
        self._write("当前会话还没有任务计划。")

    def print_no_active_task(self) -> None:
        self._stop_status()
        self._write("当前没有可取消的活动计划任务。")

    def print_task_cancelled(self, plan: dict) -> None:
        self._stop_status()
        self._write(
            "已取消当前计划任务并清除 Checkpoint："
            f"{_shorten(str(plan.get('goal', '')), 200)}"
        )

    def print_cancel_error(self, error: object) -> None:
        self._stop_status()
        self._write(
            "取消任务失败："
            f"{_shorten(str(error), 400)}。"
            "任务与 Checkpoint 已保留。"
        )

    def print_unknown_command(self, command: str) -> None:
        self._write(f"未知命令：{command}。输入 /help 查看可用命令。")

    def print_command_usage(self, usage: str) -> None:
        self._write(f"命令格式：{usage}")

    def print_empty_input(self) -> None:
        self._write("请输入任务，或输入 /help 查看命令。")

    def print_input_cancelled(self) -> None:
        self._stop_status()
        self._write("已取消当前输入。输入 /exit 可以退出。")

    def print_exit(self) -> None:
        self._stop_status()
        self._write("已退出。")

    def print_checkpoint_available(self, path: object) -> None:
        self._stop_status()
        self._write(
            "检测到尚未完成的计划任务，已拒绝启动新任务。\n"
            f"Checkpoint：{path}\n"
            "请运行 `python main.py --resume` 恢复该任务。"
        )

    def print_checkpoint_restored(self, state: dict, path: object) -> None:
        self._stop_status()
        plan = state["plan"]
        current_step = next(
            (
                step
                for step in plan["steps"]
                if step["id"] == plan["current_step_id"]
            ),
            None,
        )
        next_step = next(
            (
                step
                for step in plan["steps"]
                if step["status"] == "pending"
            ),
            None,
        )
        current_summary = (
            _shorten(current_step["description"], 160)
            if current_step is not None
            else "尚未启动"
        )
        next_summary = (
            _shorten(next_step["description"], 160)
            if next_step is not None
            else "无"
        )
        self._write(
            "已恢复计划任务：\n"
            f"  目标：{_shorten(plan['goal'], 200)}\n"
            f"  状态：{plan['status']}\n"
            f"  Revision：{plan['revision']}\n"
            f"  当前步骤：{current_summary}\n"
            f"  下一步骤：{next_summary}\n"
            f"  Checkpoint：{path}\n"
            "请补充阻塞信息，或输入“继续”执行。"
        )

    def print_checkpoint_error(self, error: object) -> None:
        self._stop_status()
        self._write(
            "Checkpoint 恢复失败："
            f"{_shorten(str(error), 400)}\n"
            "原文件未被覆盖，请检查后重试。"
        )

    def print_error(self, error: object) -> None:
        self._stop_status()
        self._write(f"运行失败：{_shorten(str(error), 400)}")

    def report_unhandled_error(self, error: object) -> None:
        if not self._failure_reported:
            self.print_error(error)

    def confirm_save(self, paper: dict) -> bool:
        self._stop_status()
        self._write("\n=== 保存确认 ===")
        self._write(f"标题：{paper.get('title') or '未知'}")

        authors = paper.get("authors") or []
        if authors:
            self._write(f"作者：{', '.join(authors)}")

        self._write(f"来源：{paper.get('source') or '未知'}")
        self._write(f"链接：{paper.get('paper_url') or '未知'}")
        self._write(f"候选 ID：{paper.get('candidate_id') or '未知'}")

        try:
            if self._input_func is not None:
                answer = self._input_func(
                    "确认保存到本地文献库？[y/N]："
                )
            else:
                answer = input("确认保存到本地文献库？[y/N]：")
        except (EOFError, KeyboardInterrupt):
            self._write("未确认，已取消保存。")
            return False

        return answer.strip().lower() in SAVE_APPROVALS

    def _stop_status(self) -> None:
        if self._status is None:
            return
        self._status.stop()
        self._status = None

    def _write(self, message: str) -> None:
        if self._console is not None:
            self._console.print(message, markup=False)
            return
        print(message, file=self.output)


def parse_slash_command(value: str) -> tuple[str, str] | None:
    """Return a normalized slash command and its optional argument."""
    stripped = value.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split(maxsplit=1)
    command = parts[0].casefold()
    argument = parts[1].casefold() if len(parts) == 2 else ""
    return command, argument


def confirm_save(paper: dict) -> bool:
    """Backward-compatible plain confirmation helper."""
    return TerminalUI(plain=True).confirm_save(paper)


def print_debug_trace(
    state: dict,
    start_message_index: int = 1,
    input_text: str | None = None,
) -> None:
    """Print a completed trace for tests and non-event integrations."""
    print("\n=== Agent Debug Trace ===")
    print(f"[Input] {input_text or state['user_query']}")

    decision_step = 0
    for message in state["messages"][start_message_index:]:
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
            display_content = _debug_tool_content(
                message["name"], message["content"]
            )
            result = json.dumps(display_content, ensure_ascii=False)
            print(f"[Step {decision_step} | Tool] result")
            print(f"  name: {message['name']}")
            print(f"  content: {result}")
            continue

        if message["role"] == "assistant":
            decision_step += 1
            print(f"[Step {decision_step} | LLM] final")

    print(f"[Summary] LLM steps: {decision_step}")


def _debug_tool_content(tool_name: str, content: object) -> object:
    """Keep paper text results readable without flooding the terminal."""
    if not isinstance(content, dict):
        return content

    if tool_name == "search_paper":
        display_content = dict(content)
        papers = display_content.get("papers")
        if isinstance(papers, list):
            display_content["papers"] = [
                _debug_paper_preview(paper) for paper in papers
            ]
        return display_content

    if tool_name == "retrieve_paper_chunks":
        display_content = dict(content)
        matches = display_content.get("matches")
        if isinstance(matches, list):
            display_content["matches"] = [
                _debug_chunk_preview(match) for match in matches
            ]
        return display_content

    if tool_name != "extract_paper_text":
        return content

    display_content = dict(content)
    text = display_content.pop("text", None)
    if isinstance(text, str):
        display_content["text_preview"] = _debug_text_preview(text)
    return display_content


def _debug_paper_preview(paper: object) -> object:
    if not isinstance(paper, dict):
        return paper
    display_paper = dict(paper)
    abstract = display_paper.pop("abstract", None)
    if isinstance(abstract, str):
        display_paper["abstract_preview"] = _debug_text_preview(abstract)
    return display_paper


def _debug_chunk_preview(chunk: object) -> object:
    if not isinstance(chunk, dict):
        return chunk
    display_chunk = dict(chunk)
    text = display_chunk.pop("text", None)
    if isinstance(text, str):
        display_chunk["text_preview"] = _debug_text_preview(text)
    return display_chunk


def _debug_text_preview(text: str) -> str:
    preview = text[:DEBUG_TEXT_PREVIEW_CHARS]
    if len(text) > DEBUG_TEXT_PREVIEW_CHARS:
        preview += "…"
    return preview


def _tool_argument_summary(tool_name: str, arguments: object) -> str:
    if not isinstance(arguments, dict):
        return ""
    if tool_name in {"search_paper", "retrieve_paper_chunks"}:
        return _shorten(str(arguments.get("query", "")), 100)
    paper_id = arguments.get("paper_id") or arguments.get("candidate_id")
    return _shorten(str(paper_id), 100) if paper_id else ""


def _tool_result_summary(tool_name: str, result: object) -> str:
    if not isinstance(result, dict):
        return _shorten(str(result), 160)
    if "error" in result:
        error = result["error"]
        if isinstance(error, dict):
            error_type = error.get("type", "error")
            message = _shorten(str(error.get("message", "")), 160)
            return f"{error_type}: {message}" if message else str(error_type)
        return _shorten(str(error), 160)

    if tool_name == "search_paper":
        source = result.get("source", "unknown")
        count = result.get("count", 0)
        assessment = result.get("relevance_assessment")
        status = assessment.get("status") if isinstance(assessment, dict) else None
        parts = [str(source), f"{count} 篇候选"]
        if status:
            parts.append(str(status))
        return " · ".join(parts)
    if tool_name == "list_library":
        return f"返回 {result.get('count', 0)} / 共 {result.get('total', 0)} 篇"
    if tool_name == "save_paper":
        if result.get("saved"):
            return f"已保存 {_shorten(str(result.get('title', '')), 100)}"
        return str(result.get("reason", "未保存"))
    if tool_name == "download_paper":
        action = "命中缓存" if result.get("cached") else "下载完成"
        return f"{action} · {_format_bytes(result.get('size_bytes'))}"
    if tool_name == "extract_paper_text":
        coverage = _page_coverage_text(result.get("page_numbers", []))
        partial = "，末页部分" if result.get("last_page_partial") else ""
        return (
            f"覆盖 {coverage}{partial} · "
            f"{result.get('char_count', 0)} 字符"
        )
    if tool_name == "retrieve_paper_chunks":
        if result.get("rejected_low_query_coverage"):
            coverage = result.get("query_term_coverage")
            minimum = result.get("minimum_query_term_coverage")
            if isinstance(coverage, (int, float)) and isinstance(
                minimum, (int, float)
            ):
                return (
                    "查询证据不足 · "
                    f"查询词覆盖 {coverage:.0%} < {minimum:.0%}"
                )
            return "查询证据不足"
        pages = sorted(
            {
                match.get("page")
                for match in result.get("matches", [])
                if isinstance(match, dict) and isinstance(match.get("page"), int)
            }
        )
        summary = (
            f"{result.get('count', 0)} 个片段 · "
            f"命中 {_page_coverage_text(pages)}"
        )
        index_status = {
            "built": "新建索引",
            "cached": "复用索引",
            "rebuilt": "重建索引",
        }.get(result.get("index_status"))
        if index_status:
            coverage = (
                "全文覆盖"
                if result.get("coverage_complete")
                else "部分覆盖"
            )
            summary += f" · {index_status}/{coverage}"
        return summary
    return "完成"


def _page_coverage_text(page_numbers: object) -> str:
    if not isinstance(page_numbers, list) or not page_numbers:
        return "无可用页"
    numbers = [number for number in page_numbers if isinstance(number, int)]
    if not numbers:
        return "无可用页"
    if numbers == list(range(numbers[0], numbers[-1] + 1)):
        if len(numbers) == 1:
            return f"第 {numbers[0]} 页"
        return f"第 {numbers[0]}–{numbers[-1]} 页"
    return "第 " + "、".join(str(number) for number in numbers) + " 页"


def _format_bytes(value: Any) -> str:
    if not isinstance(value, int) or value < 0:
        return "大小未知"
    if value < 1_024:
        return f"{value} B"
    if value < 1_024 * 1_024:
        return f"{value / 1_024:.1f} KiB"
    return f"{value / (1_024 * 1_024):.1f} MiB"


def _duration_text(value: object) -> str:
    if not isinstance(value, (int, float)):
        return ""
    if value < 1_000:
        return f" · {value:.0f} ms"
    return f" · {value / 1_000:.2f} s"


def _shorten(value: str, limit: int) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "…"
