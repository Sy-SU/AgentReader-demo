import os
import subprocess
import sys
import unittest
from io import StringIO
from unittest.mock import patch

from main import main as run_cli
from terminal import TerminalUI, parse_slash_command


class TerminalUITests(unittest.TestCase):
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

        def run_with_one_failure(state, confirm_save=None, on_event=None):
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
