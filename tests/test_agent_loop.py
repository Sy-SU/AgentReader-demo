import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from main import print_debug_trace
from runtime import execute_tool, run_agent
from state import create_state
from tools import search_paper


class AgentV1Tests(unittest.TestCase):
    def test_create_state_has_initial_user_message(self):
        state = create_state("请搜索 TASA")

        self.assertEqual(state["user_query"], "请搜索 TASA")
        self.assertEqual(
            state["messages"],
            [{"role": "user", "content": "请搜索 TASA"}],
        )
        self.assertEqual(state["step"], 0)

    def test_search_paper_found_and_not_found(self):
        found = search_paper("  TASA  ")
        missing = search_paper("CAST")

        self.assertTrue(found["found"])
        self.assertEqual(found["title"], "TASA (demo record)")
        self.assertFalse(missing["found"])
        self.assertIsNone(missing["title"])
        self.assertIsNone(missing["pdf_url"])

    def test_execute_tool_rejects_unknown_tool_and_bad_arguments(self):
        unknown = execute_tool("delete_everything", {})
        bad_arguments = execute_tool("search_paper", {"wrong_name": "TASA"})

        self.assertEqual(unknown["error"]["type"], "unknown_tool")
        self.assertIn("not registered", unknown["error"]["message"])
        self.assertEqual(
            bad_arguments["error"]["type"], "invalid_arguments"
        )
        self.assertTrue(bad_arguments["error"]["message"])

    @patch.dict(os.environ, {"LLM_PROVIDER": "fake"}, clear=False)
    def test_agent_loop_records_full_two_step_trace(self):
        state = create_state("请搜索 TASA")

        answer = run_agent(state)

        self.assertIn("找到论文：TASA (demo record)", answer)
        self.assertEqual(state["step"], 2)
        self.assertEqual(
            [message["role"] for message in state["messages"]],
            ["user", "assistant", "tool", "assistant"],
        )

        tool_call = state["messages"][1]["tool_call"]
        self.assertEqual(tool_call["id"], "fake-call-1")
        self.assertEqual(tool_call["name"], "search_paper")
        self.assertEqual(tool_call["arguments"], {"query": "TASA"})

        tool_result = state["messages"][2]
        self.assertEqual(tool_result["tool_call_id"], tool_call["id"])
        self.assertEqual(tool_result["name"], "search_paper")
        self.assertTrue(tool_result["content"]["found"])
        self.assertEqual(state["messages"][-1]["content"], answer)

    @patch.dict(os.environ, {"LLM_PROVIDER": "fake"}, clear=False)
    def test_max_steps_stops_the_loop(self):
        state = create_state("请搜索 TASA")

        answer = run_agent(state, max_steps=1)

        self.assertEqual(state["step"], 1)
        self.assertIn("max_steps=1", answer)
        self.assertEqual(
            [message["role"] for message in state["messages"]],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(state["messages"][-1]["content"], answer)

    @patch.dict(os.environ, {"LLM_PROVIDER": "fake"}, clear=False)
    def test_debug_trace_shows_tool_details(self):
        state = create_state("请搜索 TASA")
        run_agent(state)
        output = StringIO()

        with redirect_stdout(output):
            print_debug_trace(state)

        trace = output.getvalue()
        self.assertIn("Agent Debug Trace", trace)
        self.assertIn("search_paper", trace)
        self.assertIn("fake-call-1", trace)
        self.assertIn("[Step 1 | Tool] result", trace)
        self.assertIn("[Summary] LLM steps: 2", trace)


if __name__ == "__main__":
    unittest.main()
