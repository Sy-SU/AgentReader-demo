import os
import unittest
from unittest.mock import patch

from planner import SUBMIT_PLAN_SCHEMA
from runtime import run_agent
from state import append_user_message, create_state


DEMO_TOOL_RESULT = {
    "found": True,
    "source": "test",
    "count": 1,
    "papers": [
        {
            "candidate_id": "arxiv:1234.56789",
            "source": "arxiv",
            "arxiv_id": "1234.56789v1",
            "doi": None,
            "title": "TASA (test record)",
            "authors": ["Test Author"],
            "abstract": "Test abstract.",
            "published": "2025-01-01T00:00:00Z",
            "paper_url": "https://arxiv.org/abs/1234.56789v1",
            "pdf_url": "https://example.com/tasa.pdf",
        }
    ],
}


RUN_LIVE_TESTS = os.getenv("RUN_LIVE_LLM_TESTS") == "1"


@unittest.skipUnless(
    RUN_LIVE_TESTS,
    "set RUN_LIVE_LLM_TESTS=1 to call the real DeepSeek API",
)
class DeepSeekIntegrationTests(unittest.TestCase):
    @patch.dict(os.environ, {"LLM_PROVIDER": "deepseek"}, clear=False)
    @patch.dict(
        "runtime.TOOL_REGISTRY",
        {"search_paper": lambda query: DEMO_TOOL_RESULT},
    )
    def test_deepseek_completes_the_tool_loop(self):
        state = create_state(
            "请务必调用 search_paper 工具搜索 TASA，"
            "再根据工具返回结果回答。"
        )

        answer = run_agent(state, max_steps=5)

        self.assertIsInstance(answer, str)
        self.assertTrue(answer.strip())
        self.assertNotIn("reaching max_steps", answer)

        tool_messages = [
            message
            for message in state["messages"]
            if message["role"] == "tool"
        ]
        self.assertGreaterEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["name"], "search_paper")
        self.assertTrue(tool_messages[0]["content"]["found"])

    @patch.dict(os.environ, {"LLM_PROVIDER": "deepseek"}, clear=False)
    @patch.dict(
        "runtime.TOOL_REGISTRY",
        {"search_paper": lambda query: DEMO_TOOL_RESULT},
    )
    def test_deepseek_thinking_replays_reasoning_across_plan_steps(self):
        state = create_state("执行两步 TASA 文献检索与总结协议测试。")
        planning_instructions = """
This is a deterministic protocol integration test. Your only valid response
is one submit_plan tool call with exactly these two step descriptions:
1. Identify TASA using the available literature catalog and retain evidence.
2. Summarize the verified TASA result in one sentence using retained evidence.
Do not answer with text and do not add, remove, or merge steps.
""".strip()

        with (
            patch("planner.PLANNING_INSTRUCTIONS", planning_instructions),
            patch(
                "planner.INITIAL_ACTION_TOOLS",
                [SUBMIT_PLAN_SCHEMA],
            ),
        ):
            plan_answer = run_agent(state, max_steps=5)

        self.assertIn("已创建任务计划", plan_answer)
        self.assertIsNotNone(state["plan"])
        self.assertEqual(len(state["plan"]["steps"]), 2)

        append_user_message(state, "继续执行这个计划。")
        final_answer = run_agent(state, max_steps=6)

        self.assertIsInstance(final_answer, str)
        self.assertTrue(final_answer.strip())
        self.assertEqual(state["plan"]["status"], "completed")
        tool_messages = [
            message
            for message in state["messages"]
            if message["role"] == "tool"
        ]
        self.assertGreaterEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["name"], "search_paper")
        self.assertTrue(
            any(
                "reasoning_content" in message
                or "reasoning_details" in message
                for message in state["messages"]
                if message["role"] == "assistant"
            )
        )


if __name__ == "__main__":
    unittest.main()
