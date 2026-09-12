import os
import unittest
from unittest.mock import patch

from runtime import run_agent
from state import create_state


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


if __name__ == "__main__":
    unittest.main()
