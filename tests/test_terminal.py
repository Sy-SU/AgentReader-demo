import os
import subprocess
import sys
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from checkpoint import save_checkpoint
from main import (
    THINKING_MAX_STEPS_PER_TURN,
    THINKING_MAX_TASK_LLM_STEPS,
    main as run_cli,
    parse_args,
)
from planning import create_plan, start_next_step
from state import create_state
from terminal import TerminalUI, parse_slash_command


class TerminalUITests(unittest.TestCase):
    def test_thinking_argument_is_explicit(self):
        with patch("sys.argv", ["main.py", "--thinking"]):
            arguments = parse_args()

        self.assertTrue(arguments.thinking)

    def test_thinking_mode_forces_reasoning_and_raises_turn_limit(self):
        inputs = iter(["复杂任务", "/exit"])
        output = StringIO()
        terminal = TerminalUI(
            plain=True,
            input_func=lambda prompt: next(inputs),
            output=output,
        )
        observed = {}

        def fake_run_agent(
            state,
            max_steps=None,
            max_task_llm_steps=None,
            confirm_save=None,
            on_event=None,
            checkpoint_file=None,
        ):
            observed["max_steps"] = max_steps
            observed["max_task_llm_steps"] = max_task_llm_steps
            observed["thinking"] = os.getenv("LLM_THINKING")
            return "完成。"

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "active.json"
            with (
                patch.dict(
                    os.environ,
                    {"LLM_THINKING": "disabled"},
                    clear=False,
                ),
                patch("main.run_agent", side_effect=fake_run_agent),
            ):
                run_cli(
                    ui=terminal,
                    thinking=True,
                    checkpoint_file=checkpoint,
                )

        self.assertEqual(
            observed,
            {
                "max_steps": THINKING_MAX_STEPS_PER_TURN,
                "max_task_llm_steps": THINKING_MAX_TASK_LLM_STEPS,
                "thinking": "enabled",
            },
        )
        self.assertEqual(THINKING_MAX_STEPS_PER_TURN, 100)
        self.assertEqual(THINKING_MAX_TASK_LLM_STEPS, 100)
        self.assertIn("Thinking 模式已开启", output.getvalue())
        self.assertIn("max_steps=100", output.getvalue())

    def test_checkpoint_summary_distinguishes_current_and_next_steps(self):
        output = StringIO()
        terminal = TerminalUI(plain=True, output=output)
        state = create_state("比较两篇论文")
        state["task_id"] = "task-summary"
        state["plan"] = create_plan(
            state["task_id"],
            "比较两篇论文",
            ["搜索第一篇论文", "搜索第二篇论文"],
        )

        terminal.print_checkpoint_restored(state, "active.json")
        not_started = output.getvalue()
        self.assertIn("当前步骤：尚未启动", not_started)
        self.assertIn("下一步骤：搜索第一篇论文", not_started)

        output.seek(0)
        output.truncate(0)
        state["plan"] = start_next_step(state["plan"])
        terminal.print_checkpoint_restored(state, "active.json")
        started = output.getvalue()
        self.assertIn("当前步骤：搜索第一篇论文", started)
        self.assertIn("下一步骤：搜索第二篇论文", started)

    def test_parse_slash_command_keeps_normal_messages_separate(self):
        self.assertIsNone(parse_slash_command("请搜索一篇论文"))
        self.assertEqual(parse_slash_command(" /HELP "), ("/help", ""))
        self.assertEqual(
            parse_slash_command("/debug ON"),
            ("/debug", "on"),
        )
        self.assertEqual(
            parse_slash_command("/debug\toff"),
            ("/debug", "off"),
        )

    def test_plain_terminal_renders_live_bounded_tool_events(self):
        output = StringIO()
        terminal = TerminalUI(debug=True, plain=True, output=output)
        terminal.handle_event(
            {
                "kind": "llm_started",
                "turn_step": 1,
                "total_step": 1,
            }
        )
        terminal.handle_event(
            {
                "kind": "llm_finished",
                "turn_step": 1,
                "total_step": 1,
                "response_type": "tool_call",
                "duration_ms": 12.0,
            }
        )
        terminal.handle_event(
            {
                "kind": "tool_started",
                "turn_step": 1,
                "total_step": 1,
                "tool_name": "search_paper",
                "arguments": {"query": "Attention Is All You Need"},
            }
        )
        terminal.handle_event(
            {
                "kind": "tool_finished",
                "turn_step": 1,
                "total_step": 1,
                "tool_name": "search_paper",
                "arguments": {"query": "Attention Is All You Need"},
                "duration_ms": 25.0,
                "result": {
                    "found": True,
                    "source": "arxiv",
                    "count": 1,
                    "papers": [
                        {
                            "title": "Attention Is All You Need",
                            "abstract": "A" * 1_000,
                        }
                    ],
                },
            }
        )

        rendered = output.getvalue()
        self.assertIn("[Step 1 | LLM] 正在思考", rendered)
        self.assertIn("search_paper", rendered)
        self.assertIn('"abstract_preview"', rendered)
        self.assertNotIn("A" * 401, rendered)
        self.assertNotIn("\x1b[", rendered)

    def test_debug_cli_prints_execution_health_after_answer(self):
        inputs = iter(["请搜索论文", "/exit"])
        output = StringIO()
        terminal = TerminalUI(
            debug=True,
            plain=True,
            input_func=lambda prompt: next(inputs),
            output=output,
        )

        def fake_run_agent(
            state,
            confirm_save=None,
            on_event=None,
            checkpoint_file=None,
        ):
            on_event({"kind": "llm_started", "turn_step": 1})
            on_event(
                {
                    "kind": "llm_finished",
                    "turn_step": 1,
                    "response_type": "final",
                    "duration_ms": 10,
                }
            )
            on_event(
                {
                    "kind": "turn_finished",
                    "turn_step": 1,
                    "status": "completed",
                }
            )
            return "搜索完成。"

        with patch("main.run_agent", side_effect=fake_run_agent):
            run_cli(ui=terminal)

        rendered = output.getvalue()
        self.assertLess(rendered.index("搜索完成。"), rendered.index("运行健康分"))
        self.assertIn("运行健康分 100/100", rendered)
        self.assertIn("Tool N/A", rendered)
        self.assertIn("不代表答案正确率", rendered)

    def test_normal_cli_does_not_print_execution_health(self):
        inputs = iter(["普通任务", "/exit"])
        output = StringIO()
        terminal = TerminalUI(
            plain=True,
            input_func=lambda prompt: next(inputs),
            output=output,
        )

        def fake_run_agent(
            state,
            confirm_save=None,
            on_event=None,
            checkpoint_file=None,
        ):
            on_event({"kind": "turn_finished", "status": "completed"})
            return "完成。"

        with patch("main.run_agent", side_effect=fake_run_agent):
            run_cli(ui=terminal)

        self.assertNotIn("运行健康分", output.getvalue())

    def test_terminal_renders_bounded_plan_and_checkpoint_events(self):
        output = StringIO()
        terminal = TerminalUI(debug=True, plain=True, output=output)
        terminal.handle_event(
            {
                "kind": "plan_step_changed",
                "turn_step": 1,
                "total_step": 2,
                "task_id": "task-test",
                "plan_revision": 1,
                "step_id": "step-001",
                "step_description": "A" * 1_000,
                "previous_status": "pending",
                "status": "running",
            }
        )
        terminal.handle_event(
            {
                "kind": "plan_replanned",
                "turn_step": 2,
                "total_step": 3,
                "plan_revision": 2,
                "replan_count": 1,
            }
        )
        terminal.handle_event(
            {
                "kind": "checkpoint_saved",
                "turn_step": 2,
                "total_step": 3,
                "checkpoint_size_bytes": 2048,
            }
        )

        rendered = output.getvalue()
        self.assertIn("pending → running", rendered)
        self.assertIn("revision 2", rendered)
        self.assertIn("2048 bytes", rendered)
        self.assertNotIn("A" * 161, rendered)

    def test_normal_terminal_summarizes_tool_result(self):
        output = StringIO()
        terminal = TerminalUI(plain=True, output=output)
        terminal.handle_event(
            {
                "kind": "tool_started",
                "turn_step": 1,
                "total_step": 1,
                "tool_name": "retrieve_paper_chunks",
                "arguments": {
                    "paper_id": "arxiv:1706.03762",
                    "query": "encoder architecture",
                },
            }
        )
        terminal.handle_event(
            {
                "kind": "tool_finished",
                "turn_step": 1,
                "total_step": 1,
                "tool_name": "retrieve_paper_chunks",
                "duration_ms": 100.0,
                "result": {
                    "count": 2,
                    "matches": [{"page": 3}, {"page": 5}],
                    "index_status": "cached",
                    "coverage_complete": True,
                },
            }
        )

        rendered = output.getvalue()
        self.assertIn("● retrieve_paper_chunks", rendered)
        self.assertIn("✓ retrieve_paper_chunks", rendered)
        self.assertIn("第 3、5 页", rendered)
        self.assertIn("复用索引/全文覆盖", rendered)

    def test_normal_terminal_explains_low_query_coverage_rejection(self):
        output = StringIO()
        terminal = TerminalUI(plain=True, output=output)
        terminal.handle_event(
            {
                "kind": "tool_finished",
                "turn_step": 1,
                "total_step": 1,
                "tool_name": "retrieve_paper_chunks",
                "duration_ms": 10.0,
                "result": {
                    "found": False,
                    "count": 0,
                    "matches": [],
                    "query_term_coverage": 0.333333,
                    "minimum_query_term_coverage": 0.5,
                    "rejected_low_query_coverage": True,
                },
            }
        )

        rendered = output.getvalue()
        self.assertIn("查询证据不足", rendered)
        self.assertIn("查询词覆盖 33% < 50%", rendered)

    def test_cli_commands_do_not_enter_state_and_clear_starts_new_state(self):
        inputs = iter(
            [
                "",
                "/help",
                "/debug on",
                "第一个任务",
                "/clear",
                "第二个任务",
                "/debug off",
                "/unknown",
                "/exit",
            ]
        )
        output = StringIO()
        terminal = TerminalUI(
            plain=True,
            input_func=lambda prompt: next(inputs),
            output=output,
        )
        observed_states = []
        observed_debug = []

        def fake_run_agent(
            state,
            confirm_save=None,
            on_event=None,
            checkpoint_file=None,
        ):
            observed_states.append(state)
            observed_debug.append(on_event.__self__.debug)
            answer = f"完成：{state['messages'][-1]['content']}"
            state["messages"].append(
                {"role": "assistant", "content": answer}
            )
            return answer

        with patch("main.run_agent", side_effect=fake_run_agent):
            run_cli(ui=terminal)

        self.assertEqual(len(observed_states), 2)
        self.assertIsNot(observed_states[0], observed_states[1])
        self.assertEqual(observed_debug, [True, True])
        self.assertEqual(
            [
                message["content"]
                for state in observed_states
                for message in state["messages"]
                if message["role"] == "user"
            ],
            ["第一个任务", "第二个任务"],
        )
        rendered = output.getvalue()
        self.assertIn("请输入任务", rendered)
        self.assertIn("可用命令", rendered)
        self.assertIn("Debug 模式已开启", rendered)
        self.assertIn("当前会话 State 已清空", rendered)
        self.assertIn("未知命令", rendered)
        self.assertIn("已退出", rendered)
        self.assertNotIn("\x1b[", rendered)

    def test_cli_recovers_after_runtime_error(self):
        inputs = iter(["失败任务", "重试任务", "/exit"])
        output = StringIO()
        terminal = TerminalUI(
            plain=True,
            input_func=lambda prompt: next(inputs),
            output=output,
        )

        call_count = 0

        def run_with_one_failure(
            state,
            confirm_save=None,
            on_event=None,
            checkpoint_file=None,
        ):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("provider unavailable")
            answer = "重试成功。"
            state["messages"].append(
                {"role": "assistant", "content": answer}
            )
            return answer

        with patch("main.run_agent", side_effect=run_with_one_failure):
            run_cli(ui=terminal)

        rendered = output.getvalue()
        self.assertIn("运行失败：provider unavailable", rendered)
        self.assertIn("重试成功", rendered)
        self.assertIn("已退出", rendered)

    def test_cli_refuses_to_overwrite_an_active_checkpoint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            state = create_state("比较两篇论文")
            state["task_id"] = "task-existing"
            state["plan"] = create_plan(
                state["task_id"],
                "比较两篇论文",
                ["搜索第一篇论文", "搜索第二篇论文"],
            )
            save_checkpoint(state, path)

            output = StringIO()
            terminal = TerminalUI(
                plain=True,
                input_func=lambda prompt: self.fail(
                    "CLI must reject before reading a new task"
                ),
                output=output,
            )
            with self.assertRaises(SystemExit) as raised:
                run_cli(ui=terminal, checkpoint_file=path)

            self.assertEqual(raised.exception.code, 1)
            rendered = output.getvalue()
            self.assertIn("检测到尚未完成", rendered)
            self.assertIn("python main.py --resume", rendered)

    def test_cli_resume_loads_checkpoint_and_keeps_same_task(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            state = create_state("比较两篇论文")
            state["task_id"] = "task-resume"
            state["plan"] = create_plan(
                state["task_id"],
                "比较两篇论文",
                ["搜索论文", "比较论文"],
            )
            save_checkpoint(state, path)

            inputs = iter(["继续", "/exit"])
            output = StringIO()
            terminal = TerminalUI(
                plain=True,
                input_func=lambda prompt: next(inputs),
                output=output,
            )
            observed = []

            def fake_run_agent(
                resumed_state,
                confirm_save=None,
                on_event=None,
                checkpoint_file=None,
            ):
                observed.append((resumed_state, checkpoint_file))
                return "任务继续执行。"

            with patch("main.run_agent", side_effect=fake_run_agent):
                run_cli(
                    ui=terminal,
                    resume=True,
                    checkpoint_file=path,
                )

            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0][0]["task_id"], "task-resume")
            self.assertEqual(observed[0][1], path.resolve())
            self.assertEqual(
                observed[0][0]["messages"][-1],
                {"role": "user", "content": "继续"},
            )
            rendered = output.getvalue()
            self.assertIn("已恢复计划任务", rendered)
            self.assertIn("当前步骤：尚未启动", rendered)
            self.assertIn("下一步骤：搜索论文", rendered)
            self.assertIn("任务继续执行", rendered)

    def test_clear_cannot_discard_a_resumed_active_task(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            state = create_state("比较两篇论文")
            state["task_id"] = "task-resume"
            state["plan"] = create_plan(
                state["task_id"],
                "比较两篇论文",
                ["搜索论文", "比较论文"],
            )
            save_checkpoint(state, path)

            inputs = iter(["/clear", "/exit"])
            output = StringIO()
            terminal = TerminalUI(
                plain=True,
                input_func=lambda prompt: next(inputs),
                output=output,
            )

            run_cli(
                ui=terminal,
                resume=True,
                checkpoint_file=path,
            )

            self.assertTrue(path.exists())
            rendered = output.getvalue()
            self.assertIn("不能使用 /clear", rendered)
            self.assertNotIn("当前会话 State 已清空", rendered)

    def test_plan_command_renders_bounded_active_plan(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            state = create_state("比较两篇论文")
            state["task_id"] = "task-resume"
            state["plan"] = create_plan(
                state["task_id"],
                "G" * 1_000,
                ["搜索第一篇论文", "比较核心架构"],
            )
            save_checkpoint(state, path)

            inputs = iter(["/plan", "/exit"])
            output = StringIO()
            terminal = TerminalUI(
                plain=True,
                input_func=lambda prompt: next(inputs),
                output=output,
            )
            run_cli(
                ui=terminal,
                resume=True,
                checkpoint_file=path,
            )

            rendered = output.getvalue()
            self.assertIn("当前任务计划", rendered)
            self.assertIn("状态：running", rendered)
            self.assertIn("Revision：1", rendered)
            self.assertIn("当前步骤：尚未启动", rendered)
            self.assertIn("下一步骤：step-001 · 搜索第一篇论文", rendered)
            self.assertIn("[pending] 搜索第一篇论文", rendered)
            self.assertNotIn("G" * 201, rendered)
            self.assertTrue(path.exists())

    def test_cancel_removes_checkpoint_and_allows_a_new_task(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            state = create_state("比较两篇论文")
            state["task_id"] = "task-resume"
            state["plan"] = create_plan(
                state["task_id"],
                "比较两篇论文",
                ["搜索论文", "比较论文"],
            )
            save_checkpoint(state, path)

            inputs = iter(["/cancel", "开始新任务", "/exit"])
            output = StringIO()
            terminal = TerminalUI(
                plain=True,
                input_func=lambda prompt: next(inputs),
                output=output,
            )
            observed_states = []

            def fake_run_agent(
                new_state,
                confirm_save=None,
                on_event=None,
                checkpoint_file=None,
            ):
                observed_states.append(new_state)
                return "新任务已接收。"

            with patch("main.run_agent", side_effect=fake_run_agent):
                run_cli(
                    ui=terminal,
                    resume=True,
                    checkpoint_file=path,
                )

            self.assertFalse(path.exists())
            self.assertEqual(len(observed_states), 1)
            self.assertIsNone(observed_states[0]["task_id"])
            self.assertEqual(
                observed_states[0]["messages"],
                [{"role": "user", "content": "开始新任务"}],
            )
            rendered = output.getvalue()
            self.assertIn("已取消当前计划任务", rendered)
            self.assertIn("新任务已接收", rendered)

    def test_plan_and_cancel_without_task_are_normal_commands(self):
        inputs = iter(["/plan", "/cancel", "/exit"])
        output = StringIO()
        terminal = TerminalUI(
            plain=True,
            input_func=lambda prompt: next(inputs),
            output=output,
        )

        run_cli(ui=terminal)

        rendered = output.getvalue()
        self.assertIn("还没有任务计划", rendered)
        self.assertIn("没有可取消", rendered)

    def test_cli_resume_reports_corrupt_checkpoint_without_overwriting(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            original = b"{not-json"
            path.write_bytes(original)
            output = StringIO()
            terminal = TerminalUI(
                plain=True,
                input_func=lambda prompt: self.fail(
                    "CLI must reject before reading input"
                ),
                output=output,
            )

            with self.assertRaises(SystemExit) as raised:
                run_cli(
                    ui=terminal,
                    resume=True,
                    checkpoint_file=path,
                )

            self.assertEqual(raised.exception.code, 1)
            self.assertEqual(path.read_bytes(), original)
            self.assertIn("Checkpoint 恢复失败", output.getvalue())

    def test_non_interactive_runtime_error_returns_nonzero_exit_code(self):
        environment = {**os.environ, "LLM_PROVIDER": "unsupported-provider"}
        result = subprocess.run(
            [sys.executable, "main.py", "--plain"],
            cwd=os.path.dirname(os.path.dirname(__file__)),
            input="测试失败路径\n",
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("运行失败", result.stdout)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
