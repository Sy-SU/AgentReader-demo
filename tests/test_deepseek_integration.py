import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
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
RUN_LIVE_SYSTEM_TESTS = (
    RUN_LIVE_TESTS
    and os.getenv("RUN_LIVE_ARXIV_TESTS") == "1"
)

LIVE_MULTI_PAPER_PLAN = {
    "type": "plan",
    "content": None,
    "tool_call_id": "live-system-plan",
    "step_descriptions": [
        (
            "Search separately for the exact arXiv papers 1706.03762 and "
            "1810.04805, and retain one verified candidate for each paper."
        ),
        (
            "Download the PDFs for both verified arXiv candidates, using "
            "their exact candidate IDs."
        ),
        (
            "Call retrieve_paper_chunks once for each downloaded paper to "
            "find page-numbered evidence about its core architecture and "
            "pre-training or training objectives."
        ),
        (
            "Compare Attention Is All You Need with BERT using only the "
            "retrieved evidence, and cite at least one page number for each "
            "paper in the final answer."
        ),
    ],
}


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


@unittest.skipUnless(
    RUN_LIVE_SYSTEM_TESTS,
    (
        "set RUN_LIVE_LLM_TESTS=1 and RUN_LIVE_ARXIV_TESTS=1 "
        "to run the real multi-paper system test"
    ),
)
class LiveMultiPaperSystemTests(unittest.TestCase):
    @patch.dict(os.environ, {"LLM_PROVIDER": "deepseek"}, clear=False)
    def test_deepseek_compares_two_real_pdfs_with_page_evidence(self):
        goal = (
            "Complete this exact multi-paper acceptance task. Find arXiv "
            "papers 1706.03762 and 1810.04805, download both PDFs, retrieve "
            "passages about their core architectures and training objectives, "
            "then compare them in Chinese. The final comparison must name "
            "both papers and cite at least one PDF page number for each."
        )

        with TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            checkpoint_file = temporary_root / "checkpoint.json"
            state = create_state(goal)

            with (
                patch.dict(
                    os.environ,
                    {
                        "PAPER_CACHE_DIR": str(temporary_root / "pdfs"),
                        "PAPER_INDEX_DIR": str(temporary_root / "indexes"),
                    },
                    clear=False,
                ),
                patch(
                    "runtime.decide_next_action",
                    return_value=LIVE_MULTI_PAPER_PLAN,
                ),
            ):
                plan_answer = run_agent(
                    state,
                    max_steps=1,
                    checkpoint_file=checkpoint_file,
                )

            self.assertIn("已创建任务计划", plan_answer)
            self.assertEqual(state["plan"]["status"], "running")

            append_user_message(state, "继续执行计划，不要跳过任何步骤。")
            with patch.dict(
                os.environ,
                {
                    "PAPER_CACHE_DIR": str(temporary_root / "pdfs"),
                    "PAPER_INDEX_DIR": str(temporary_root / "indexes"),
                },
                clear=False,
            ):
                final_answer = self._run_to_completion(
                    state,
                    checkpoint_file,
                )

            self.assertEqual(state["plan"]["status"], "completed")
            self.assertFalse(checkpoint_file.exists())
            self._assert_real_tool_evidence(state)
            self.assertIn("Attention Is All You Need", final_answer)
            self.assertIn("BERT", final_answer)
            page_citations = re.findall(
                r"(?:第\s*\d+\s*页|(?:page|p\.?)\s*\d+)",
                final_answer,
                flags=re.IGNORECASE,
            )
            self.assertGreaterEqual(len(page_citations), 2)

    def _run_to_completion(self, state, checkpoint_file):
        answer = ""
        for _ in range(3):
            answer = run_agent(
                state,
                max_steps=20,
                checkpoint_file=checkpoint_file,
            )
            if state["plan"]["status"] == "completed":
                return answer
            if state["plan"]["status"] == "blocked":
                append_user_message(
                    state,
                    (
                        "任务所需信息已经完整提供。请恢复当前步骤；如果刚才是"
                        "临时网络或检索失败，请使用同一可信论文 ID 重试，不要"
                        "虚构结果。"
                    ),
                )
            elif state["plan"]["status"] == "running":
                append_user_message(state, "继续完成当前计划。")
            else:
                self.fail(self._blocked_diagnostic(state, answer))
        self.fail(
            "The live multi-paper task did not complete within three turns; "
            + self._blocked_diagnostic(state, answer)
        )

    def _blocked_diagnostic(self, state, answer):
        plan = state["plan"]
        current_step = next(
            (
                step
                for step in plan["steps"]
                if step["id"] == plan["current_step_id"]
            ),
            None,
        )
        tool_summary = []
        for message in state["messages"]:
            if message["role"] != "tool":
                continue
            content = message.get("content", {})
            summary = {
                "name": message.get("name"),
                "paper_id": content.get("paper_id"),
                "found": content.get("found"),
                "error": content.get("error"),
            }
            tool_summary.append(summary)
        return (
            "live task became blocked; "
            f"current_step={current_step}; "
            f"answer={answer[:500]!r}; "
            f"tools={tool_summary!r}"
        )

    def _assert_real_tool_evidence(self, state):
        tool_messages = [
            message
            for message in state["messages"]
            if message["role"] == "tool"
        ]
        searches = [
            message["content"]
            for message in tool_messages
            if message["name"] == "search_paper"
        ]
        downloads = [
            message["content"]
            for message in tool_messages
            if message["name"] == "download_paper"
        ]
        retrievals = [
            message["content"]
            for message in tool_messages
            if message["name"] == "retrieve_paper_chunks"
        ]

        for arxiv_id in ("1706.03762", "1810.04805"):
            self.assertTrue(
                any(
                    any(
                        paper.get("arxiv_id", "").startswith(arxiv_id)
                        for paper in result.get("papers", [])
                    )
                    for result in searches
                ),
                f"missing real search candidate {arxiv_id}",
            )
            paper_id = f"arxiv:{arxiv_id}"
            self.assertTrue(
                any(
                    result.get("paper_id") == paper_id
                    and "error" not in result
                    for result in downloads
                ),
                f"missing real PDF download {paper_id}",
            )
            matching_retrievals = [
                result
                for result in retrievals
                if result.get("paper_id") == paper_id
            ]
            self.assertTrue(
                any(
                    result.get("found")
                    and all(
                        isinstance(match.get("page"), int)
                        and match.get("text")
                        for match in result.get("matches", [])
                    )
                    for result in matching_retrievals
                ),
                f"missing page-numbered retrieval evidence for {paper_id}",
            )


if __name__ == "__main__":
    unittest.main()
