import inspect
import json
import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from urllib.error import HTTPError

import tools
from main import confirm_save
from main import main as run_cli
from main import print_debug_trace
from llm import _normalize_api_response
from runtime import DEFAULT_MAX_STEPS_PER_TURN, execute_tool, run_agent
from state import append_user_message, create_state
from tools import list_library, save_paper, search_paper
from tools.search import (
    _fetch_arxiv_feed,
    _fetch_crossref_data,
    _search_result,
)


ARXIV_RESULT = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>  Attention Is All You Need  </title>
    <summary>  A transformer architecture.  </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7"
          rel="related" type="application/pdf" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2501.00001v1</id>
    <published>2025-01-01T00:00:00Z</published>
    <title>  A Second Candidate  </title>
    <summary>  Another abstract.  </summary>
    <author><name>Grace Hopper</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2501.00001v1"
          rel="related" type="application/pdf" />
  </entry>
</feed>
"""

EMPTY_ARXIV_RESULT = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" />
"""

CROSSREF_RESULT = b"""\
{
  "message": {
    "items": [
      {
        "DOI": "10.1000/example",
        "title": ["Fallback Paper"],
        "author": [
          {"given": "Ada", "family": "Lovelace"},
          {"given": "Alan", "family": "Turing"}
        ],
        "published": {"date-parts": [[2025, 7, 3]]},
        "URL": "http://doi.org/10.1000/example",
        "abstract": "<jats:p>A fallback abstract.</jats:p>",
        "link": [
          {
            "URL": "http://example.com/paper.pdf",
            "content-type": "application/pdf"
          }
        ]
      }
    ]
  }
}
"""

EMPTY_CROSSREF_RESULT = b'{"message": {"items": []}}'

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


class AgentV1Tests(unittest.TestCase):
    def test_tools_package_keeps_stable_public_interface(self):
        self.assertEqual(
            set(tools.__all__),
            {
                "DOWNLOAD_PAPER_SCHEMA",
                "EXTRACT_PAPER_TEXT_SCHEMA",
                "LIST_LIBRARY_SCHEMA",
                "RETRIEVE_PAPER_CHUNKS_SCHEMA",
                "SAVE_PAPER_SCHEMA",
                "SEARCH_PAPER_SCHEMA",
                "download_paper",
                "extract_paper_text",
                "list_library",
                "retrieve_paper_chunks",
                "save_paper",
                "search_paper",
            },
        )

    def test_multiple_provider_tool_calls_are_serialized_to_the_first(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "search-call-1",
                                "function": {
                                    "name": "search_paper",
                                    "arguments": json.dumps(
                                        {"query": "3D scene reconstruction"}
                                    ),
                                },
                            },
                            {
                                "id": "search-call-2",
                                "function": {
                                    "name": "search_paper",
                                    "arguments": json.dumps(
                                        {"query": "indoor scene reconstruction"}
                                    ),
                                },
                            },
                        ],
                    }
                }
            ]
        }

        normalized = _normalize_api_response(response)

        self.assertEqual(normalized["type"], "tool_call")
        self.assertEqual(normalized["tool_call_id"], "search-call-1")
        self.assertEqual(normalized["tool_name"], "search_paper")
        self.assertEqual(
            normalized["tool_arguments"],
            {"query": "3D scene reconstruction"},
        )

    def test_create_state_has_initial_user_message(self):
        state = create_state("请搜索 TASA")

        self.assertEqual(state["user_query"], "请搜索 TASA")
        self.assertEqual(
            state["messages"],
            [{"role": "user", "content": "请搜索 TASA"}],
        )
        self.assertEqual(state["step"], 0)

        append_user_message(state, "第 1 篇")
        self.assertEqual(
            state["messages"][-1],
            {"role": "user", "content": "第 1 篇"},
        )

    @patch("tools.search._fetch_arxiv_feed")
    def test_search_paper_parses_found_result(self, fetch_feed):
        fetch_feed.return_value = ARXIV_RESULT

        with patch("tools.search._fetch_crossref_data") as fetch_crossref:
            found = search_paper("  Attention Is All You Need  ")

        self.assertTrue(found["found"])
        self.assertEqual(found["source"], "arxiv")
        self.assertEqual(found["count"], 2)
        self.assertEqual(
            found["relevance_assessment"]["status"],
            "lexical_match",
        )
        self.assertGreater(
            found["relevance_assessment"]["best_score"],
            0,
        )
        self.assertEqual(len(found["papers"]), 2)

        first_paper = found["papers"][0]
        self.assertEqual(first_paper["candidate_id"], "arxiv:1706.03762")
        self.assertEqual(first_paper["arxiv_id"], "1706.03762v7")
        self.assertEqual(first_paper["title"], "Attention Is All You Need")
        self.assertEqual(
            first_paper["authors"], ["Ashish Vaswani", "Noam Shazeer"]
        )
        self.assertEqual(
            first_paper["abstract"], "A transformer architecture."
        )
        self.assertEqual(
            first_paper["published"], "2017-06-12T17:57:34Z"
        )
        self.assertEqual(
            first_paper["pdf_url"],
            "https://arxiv.org/pdf/1706.03762v7",
        )
        self.assertEqual(found["papers"][1]["title"], "A Second Candidate")
        self.assertEqual(first_paper["source"], "arxiv")
        fetch_feed.assert_called_once_with("Attention Is All You Need")
        fetch_crossref.assert_not_called()

    @patch("tools.search.urlopen")
    def test_arxiv_id_query_uses_direct_id_list(self, urlopen):
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = ARXIV_RESULT

        result = _fetch_arxiv_feed(
            "Vaswani Attention Is All You Need 1706.03762"
        )

        request = urlopen.call_args.args[0]
        self.assertEqual(result, ARXIV_RESULT)
        self.assertIn("id_list=1706.03762", request.full_url)
        self.assertNotIn("search_query", request.full_url)

    @patch("tools.search.urlopen")
    def test_arxiv_keyword_query_requests_bounded_candidate_pool(
        self, urlopen
    ):
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = EMPTY_ARXIV_RESULT

        _fetch_arxiv_feed("scene reconstruction")

        request = urlopen.call_args.args[0]
        self.assertIn("max_results=10", request.full_url)

    @patch("tools.search.urlopen")
    def test_crossref_query_requests_bounded_candidate_pool(self, urlopen):
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = EMPTY_CROSSREF_RESULT

        _fetch_crossref_data("scene reconstruction")

        request = urlopen.call_args.args[0]
        self.assertIn("rows=10", request.full_url)

    @patch(
        "tools.search._fetch_arxiv_feed",
        return_value=ARXIV_RESULT,
    )
    def test_search_locally_reranks_title_and_abstract_matches(
        self, fetch_arxiv
    ):
        result = search_paper("Second Candidate")

        self.assertEqual(result["candidate_count"], 2)
        self.assertFalse(result["truncated"])
        self.assertEqual(
            result["ranking_method"],
            "weighted_lexical_overlap_v2",
        )
        self.assertEqual(result["papers"][0]["title"], "A Second Candidate")
        relevance = result["papers"][0]["local_relevance"]
        self.assertEqual(relevance["title_matches"], ["second", "candidate"])
        self.assertTrue(relevance["title_phrase_match"])
        self.assertEqual(relevance["source_rank"], 2)
        self.assertGreater(
            relevance["score"],
            result["papers"][1]["local_relevance"]["score"],
        )
        fetch_arxiv.assert_called_once_with("Second Candidate")

    def test_local_ranking_returns_three_and_preserves_source_ties(self):
        papers = [
            {
                "source": "arxiv",
                "arxiv_id": f"2501.0000{index}",
                "title": title,
                "authors": [],
                "abstract": None,
                "published": None,
                "paper_url": f"https://arxiv.org/abs/2501.0000{index}",
                "pdf_url": None,
            }
            for index, title in enumerate(
                [
                    "Unrelated Paper",
                    "Target Alpha",
                    "Target Beta",
                    "Target Gamma",
                ],
                start=1,
            )
        ]

        result = _search_result("arxiv", papers, query="target")

        self.assertEqual(result["candidate_count"], 4)
        self.assertEqual(result["count"], 3)
        self.assertTrue(result["truncated"])
        self.assertEqual(
            [paper["title"] for paper in result["papers"]],
            ["Target Alpha", "Target Beta", "Target Gamma"],
        )
        self.assertEqual(
            [
                paper["local_relevance"]["source_rank"]
                for paper in result["papers"]
            ],
            [2, 3, 4],
        )

    def test_local_ranking_uses_abstract_matches(self):
        papers = [
            {
                "source": "arxiv",
                "arxiv_id": "2501.00001",
                "title": "First Generic Paper",
                "authors": [],
                "abstract": "No relevant content.",
                "published": None,
                "paper_url": "https://arxiv.org/abs/2501.00001",
                "pdf_url": None,
            },
            {
                "source": "arxiv",
                "arxiv_id": "2501.00002",
                "title": "Second Generic Paper",
                "authors": [],
                "abstract": "A target method for reconstruction.",
                "published": None,
                "paper_url": "https://arxiv.org/abs/2501.00002",
                "pdf_url": None,
            },
        ]

        result = _search_result("arxiv", papers, query="target method")

        first = result["papers"][0]
        self.assertEqual(first["title"], "Second Generic Paper")
        self.assertEqual(
            first["local_relevance"]["abstract_matches"],
            ["target", "method"],
        )
        self.assertEqual(first["local_relevance"]["score"], 2)

    def test_exact_title_match_outranks_longer_phrase_match(self):
        papers = [
            {
                "source": "arxiv",
                "arxiv_id": "2604.21816v1",
                "title": "Tool Attention Is All You Need: A Longer Title",
                "authors": [],
                "abstract": "Attention is all you need for these tools.",
                "published": None,
                "paper_url": "https://arxiv.org/abs/2604.21816v1",
                "pdf_url": None,
            },
            {
                "source": "arxiv",
                "arxiv_id": "1706.03762v7",
                "title": "Attention Is All You Need",
                "authors": [],
                "abstract": "A transformer architecture using attention.",
                "published": None,
                "paper_url": "https://arxiv.org/abs/1706.03762v7",
                "pdf_url": None,
            },
        ]

        result = _search_result(
            "arxiv",
            papers,
            query="Attention Is All You Need",
        )

        self.assertEqual(
            result["papers"][0]["arxiv_id"],
            "1706.03762v7",
        )
        self.assertTrue(
            result["papers"][0]["local_relevance"]["exact_title_match"]
        )
        self.assertFalse(
            result["papers"][1]["local_relevance"]["exact_title_match"]
        )

    def test_search_relevance_distinguishes_no_match_and_identifier(self):
        paper = {
            "source": "arxiv",
            "arxiv_id": "1706.03762v7",
            "title": "Attention Is All You Need",
            "authors": [],
            "abstract": "A transformer architecture.",
            "published": None,
            "paper_url": "https://arxiv.org/abs/1706.03762v7",
            "pdf_url": "https://arxiv.org/pdf/1706.03762v7",
        }

        no_match = _search_result(
            "arxiv",
            [paper],
            query="database transaction",
        )
        identifier = _search_result(
            "arxiv",
            [paper],
            query="arXiv:1706.03762",
        )

        self.assertEqual(
            no_match["relevance_assessment"],
            {"status": "no_lexical_match", "best_score": 0},
        )
        self.assertEqual(
            identifier["relevance_assessment"],
            {"status": "identifier_match", "best_score": 0},
        )

    @patch("tools.search.sleep")
    @patch("tools.search.urlopen")
    def test_arxiv_429_is_retried_once(self, urlopen, sleep):
        rate_limit_error = HTTPError(
            url="https://export.arxiv.org/api/query",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "0"},
            fp=None,
        )
        response = urlopen.return_value.__enter__.return_value
        response.read.return_value = ARXIV_RESULT
        urlopen.side_effect = [rate_limit_error, urlopen.return_value]

        result = _fetch_arxiv_feed("Attention Is All You Need")

        self.assertEqual(result, ARXIV_RESULT)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.0)

    @patch(
        "tools.search._fetch_arxiv_feed",
        return_value=EMPTY_ARXIV_RESULT,
    )
    @patch(
        "tools.search._fetch_crossref_data",
        return_value=EMPTY_CROSSREF_RESULT,
    )
    def test_search_paper_returns_normal_not_found_result(
        self, fetch_crossref, fetch_arxiv
    ):
        missing = search_paper("a paper that does not exist")

        self.assertFalse(missing["found"])
        self.assertEqual(missing["source"], "crossref")
        self.assertEqual(missing["count"], 0)
        self.assertEqual(missing["papers"], [])
        self.assertEqual(
            missing["relevance_assessment"],
            {"status": "no_candidates", "best_score": None},
        )
        self.assertEqual(missing["fallback_reason"], "arxiv_no_results")
        self.assertNotIn("arxiv_error", missing)
        fetch_arxiv.assert_called_once()
        fetch_crossref.assert_called_once()

    @patch(
        "tools.search._fetch_arxiv_feed",
        side_effect=RuntimeError("arXiv timed out"),
    )
    @patch(
        "tools.search._fetch_crossref_data",
        return_value=CROSSREF_RESULT,
    )
    def test_search_paper_falls_back_to_crossref(
        self, fetch_crossref, fetch_arxiv
    ):
        found = search_paper("Fallback Paper")

        self.assertTrue(found["found"])
        self.assertEqual(found["source"], "crossref")
        self.assertEqual(found["count"], 1)
        self.assertEqual(found["fallback_reason"], "arxiv_unavailable")
        self.assertEqual(found["arxiv_error"], "arXiv timed out")
        paper = found["papers"][0]
        self.assertEqual(paper["source"], "crossref")
        self.assertEqual(paper["candidate_id"], "doi:10.1000/example")
        self.assertEqual(paper["doi"], "10.1000/example")
        self.assertEqual(paper["title"], "Fallback Paper")
        self.assertEqual(paper["authors"], ["Ada Lovelace", "Alan Turing"])
        self.assertEqual(paper["abstract"], "A fallback abstract.")
        self.assertEqual(paper["published"], "2025-07-03")
        self.assertEqual(
            paper["paper_url"], "https://doi.org/10.1000/example"
        )
        self.assertEqual(
            paper["pdf_url"], "https://example.com/paper.pdf"
        )
        fetch_arxiv.assert_called_once_with("Fallback Paper")
        fetch_crossref.assert_called_once_with("Fallback Paper")

    def test_save_paper_writes_once_and_rejects_duplicate(self):
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "library.json"
            with patch.dict(
                os.environ,
                {"PAPER_LIBRARY_PATH": str(library_path)},
                clear=False,
            ):
                first_result = save_paper(DEMO_TOOL_RESULT["papers"][0])
                second_result = save_paper(DEMO_TOOL_RESULT["papers"][0])

            library = json.loads(library_path.read_text(encoding="utf-8"))

        self.assertTrue(first_result["saved"])
        self.assertFalse(second_result["saved"])
        self.assertEqual(second_result["reason"], "already_exists")
        self.assertEqual(first_result["paper_id"], "arxiv:1234.56789")
        self.assertEqual(first_result["total"], 1)
        self.assertEqual(second_result["total"], 1)
        self.assertEqual(len(library["papers"]), 1)
        self.assertEqual(library["papers"][0]["title"], "TASA (test record)")

    def test_list_library_returns_empty_without_creating_a_file(self):
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "library.json"
            with patch.dict(
                os.environ,
                {"PAPER_LIBRARY_PATH": str(library_path)},
                clear=False,
            ):
                result = list_library()

        self.assertEqual(result["count"], 0)
        self.assertEqual(result["total"], 0)
        self.assertFalse(result["truncated"])
        self.assertEqual(result["papers"], [])
        self.assertFalse(library_path.exists())

    def test_list_library_is_bounded_newest_first_and_omits_abstract(self):
        second_paper = {
            **DEMO_TOOL_RESULT["papers"][0],
            "candidate_id": "arxiv:9999.00001",
            "arxiv_id": "9999.00001v1",
            "title": "Second saved paper",
            "paper_url": "https://arxiv.org/abs/9999.00001v1",
        }

        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "library.json"
            with patch.dict(
                os.environ,
                {"PAPER_LIBRARY_PATH": str(library_path)},
                clear=False,
            ):
                save_paper(DEMO_TOOL_RESULT["papers"][0])
                save_paper(second_paper)
                result = list_library(limit=1)

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["total"], 2)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["papers"][0]["title"], "Second saved paper")
        self.assertNotIn("abstract", result["papers"][0])

    def test_list_library_damage_becomes_a_structured_tool_error(self):
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "library.json"
            library_path.write_text("not valid JSON", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"PAPER_LIBRARY_PATH": str(library_path)},
                clear=False,
            ):
                result = execute_tool("list_library", {})

        self.assertEqual(result["error"]["type"], "tool_execution_error")
        self.assertIn("Could not read library", result["error"]["message"])

    @patch.dict(os.environ, {"LLM_PROVIDER": "fake"}, clear=False)
    def test_agent_loop_can_list_an_empty_library(self):
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "library.json"
            with patch.dict(
                os.environ,
                {"PAPER_LIBRARY_PATH": str(library_path)},
                clear=False,
            ):
                state = create_state("请列出我的文献库")
                answer = run_agent(state)

        self.assertEqual(state["step"], 2)
        self.assertEqual(
            state["messages"][1]["tool_call"]["name"],
            "list_library",
        )
        self.assertEqual(state["messages"][1]["tool_call"]["arguments"], {})
        self.assertEqual(state["messages"][2]["content"]["count"], 0)
        self.assertIn("文献库为空", answer)
        self.assertFalse(library_path.exists())

    def test_agent_loop_can_search_then_save(self):
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "library.json"
            with (
                patch.dict(
                    os.environ,
                    {
                        "LLM_PROVIDER": "fake",
                        "PAPER_LIBRARY_PATH": str(library_path),
                    },
                    clear=False,
                ),
                patch.dict(
                    "runtime.TOOL_REGISTRY",
                    {
                        "search_paper": lambda query: DEMO_TOOL_RESULT,
                        "save_paper": save_paper,
                    },
                    clear=True,
                ),
            ):
                state = create_state("请搜索并保存 TASA")
                confirmed_papers = []

                def approve_save(paper):
                    confirmed_papers.append(paper)
                    return True

                answer = run_agent(state, confirm_save=approve_save)

            library = json.loads(library_path.read_text(encoding="utf-8"))

        self.assertEqual(state["step"], 3)
        self.assertEqual(
            [message["role"] for message in state["messages"]],
            ["user", "assistant", "tool", "assistant", "tool", "assistant"],
        )
        self.assertEqual(
            state["messages"][3]["tool_call"]["name"], "save_paper"
        )
        self.assertEqual(
            state["messages"][3]["tool_call"]["arguments"],
            {"candidate_id": "arxiv:1234.56789"},
        )
        self.assertTrue(state["messages"][4]["content"]["saved"])
        self.assertIn("论文已保存", answer)
        self.assertEqual(len(library["papers"]), 1)
        self.assertEqual(len(confirmed_papers), 1)
        self.assertEqual(
            confirmed_papers[0]["candidate_id"],
            "arxiv:1234.56789",
        )

    def test_save_tool_requires_explicit_confirmation(self):
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "library.json"
            state = create_state("保存 TASA")
            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": "test-search-call",
                    "name": "search_paper",
                    "content": DEMO_TOOL_RESULT,
                }
            )

            with patch.dict(
                os.environ,
                {"PAPER_LIBRARY_PATH": str(library_path)},
                clear=False,
            ):
                result = execute_tool(
                    "save_paper",
                    {"candidate_id": "arxiv:1234.56789"},
                    state=state,
                )

        self.assertEqual(result["error"]["type"], "approval_required")
        self.assertFalse(library_path.exists())

    def test_agent_loop_can_decline_save(self):
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "library.json"
            with (
                patch.dict(
                    os.environ,
                    {
                        "LLM_PROVIDER": "fake",
                        "PAPER_LIBRARY_PATH": str(library_path),
                    },
                    clear=False,
                ),
                patch.dict(
                    "runtime.TOOL_REGISTRY",
                    {
                        "search_paper": lambda query: DEMO_TOOL_RESULT,
                        "save_paper": save_paper,
                    },
                    clear=True,
                ),
            ):
                state = create_state("请搜索并保存 TASA")
                answer = run_agent(
                    state,
                    confirm_save=lambda paper: False,
                )

        save_result = state["messages"][4]["content"]
        self.assertEqual(state["step"], 3)
        self.assertFalse(save_result["saved"])
        self.assertEqual(save_result["reason"], "user_declined")
        self.assertIn("已取消保存", answer)
        self.assertFalse(library_path.exists())

    def test_terminal_save_confirmation_defaults_to_no(self):
        output = StringIO()

        with (
            patch("builtins.input", return_value=""),
            redirect_stdout(output),
        ):
            approved = confirm_save(DEMO_TOOL_RESULT["papers"][0])

        self.assertFalse(approved)
        self.assertIn("TASA (test record)", output.getvalue())

        with (
            patch("builtins.input", return_value="y"),
            redirect_stdout(StringIO()),
        ):
            self.assertTrue(confirm_save(DEMO_TOOL_RESULT["papers"][0]))

    def test_save_tool_rejects_candidate_outside_search_history(self):
        with TemporaryDirectory() as directory:
            library_path = Path(directory) / "library.json"
            state = create_state("请保存一篇不存在的论文")
            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": "test-search-call",
                    "name": "search_paper",
                    "content": DEMO_TOOL_RESULT,
                }
            )

            with patch.dict(
                os.environ,
                {"PAPER_LIBRARY_PATH": str(library_path)},
                clear=False,
            ):
                unknown = execute_tool(
                    "save_paper",
                    {"candidate_id": "arxiv:not-in-search-results"},
                    state=state,
                )
                copied_metadata = execute_tool(
                    "save_paper",
                    {"paper": DEMO_TOOL_RESULT["papers"][0]},
                    state=state,
                )

        self.assertEqual(unknown["error"]["type"], "unknown_candidate")
        self.assertEqual(
            copied_metadata["error"]["type"], "invalid_arguments"
        )
        self.assertFalse(library_path.exists())

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
    @patch.dict(
        "runtime.TOOL_REGISTRY",
        {"search_paper": lambda query: DEMO_TOOL_RESULT},
    )
    def test_agent_loop_records_full_two_step_trace(self):
        state = create_state("请搜索 TASA")

        answer = run_agent(state)

        self.assertIn("找到论文：TASA (test record)", answer)
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
    @patch.dict(
        "runtime.TOOL_REGISTRY",
        {"search_paper": lambda query: DEMO_TOOL_RESULT},
    )
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

    def test_default_turn_limit_is_twenty_llm_decisions(self):
        default = inspect.signature(run_agent).parameters["max_steps"].default

        self.assertEqual(DEFAULT_MAX_STEPS_PER_TURN, 20)
        self.assertEqual(default, DEFAULT_MAX_STEPS_PER_TURN)

    def test_max_steps_applies_to_each_conversational_turn(self):
        first_response = {
            "type": "final",
            "content": "请选择一篇论文。",
            "tool_call_id": None,
            "tool_name": None,
            "tool_arguments": None,
        }
        second_response = {
            "type": "final",
            "content": "已选择第一篇。",
            "tool_call_id": None,
            "tool_name": None,
            "tool_arguments": None,
        }
        state = create_state("请搜索并保存 TASA")

        with patch(
            "runtime.decide_next_action",
            side_effect=[first_response, second_response],
        ):
            first_answer = run_agent(state, max_steps=1)
            append_user_message(state, "第 1 篇")
            second_answer = run_agent(state, max_steps=1)

        self.assertEqual(first_answer, "请选择一篇论文。")
        self.assertEqual(second_answer, "已选择第一篇。")
        self.assertEqual(state["step"], 2)

    def test_duplicate_search_query_is_rejected_within_one_turn(self):
        first_search = {
            "type": "tool_call",
            "content": None,
            "tool_call_id": "search-1",
            "tool_name": "search_paper",
            "tool_arguments": {"query": "TASA   scene"},
        }
        repeated_search = {
            **first_search,
            "tool_call_id": "search-2",
            "tool_arguments": {"query": "TASA scene"},
        }
        final = {
            "type": "final",
            "content": "请提供更多线索。",
            "tool_call_id": None,
            "tool_name": None,
            "tool_arguments": None,
        }
        search = Mock(return_value=DEMO_TOOL_RESULT)
        state = create_state("请搜索 TASA")

        with (
            patch(
                "runtime.decide_next_action",
                side_effect=[first_search, repeated_search, final],
            ),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            answer = run_agent(state)

        search.assert_called_once_with(query="TASA   scene")
        duplicate_result = state["messages"][4]["content"]
        self.assertEqual(
            duplicate_result["error"]["type"],
            "duplicate_search_query",
        )
        self.assertIn(
            "current user turn",
            duplicate_result["error"]["message"],
        )
        self.assertEqual(answer, "请提供更多线索。")

    def test_same_search_query_is_allowed_in_a_later_user_turn(self):
        search_call = {
            "type": "tool_call",
            "content": None,
            "tool_call_id": "search-1",
            "tool_name": "search_paper",
            "tool_arguments": {"query": "TASA"},
        }
        final = {
            "type": "final",
            "content": "搜索完成。",
            "tool_call_id": None,
            "tool_name": None,
            "tool_arguments": None,
        }
        search = Mock(return_value=DEMO_TOOL_RESULT)
        state = create_state("请搜索 TASA")

        with (
            patch(
                "runtime.decide_next_action",
                side_effect=[
                    search_call,
                    final,
                    {**search_call, "tool_call_id": "search-2"},
                    final,
                ],
            ),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            run_agent(state)
            append_user_message(state, "请重新搜索 TASA")
            run_agent(state)

        self.assertEqual(search.call_count, 2)
        self.assertNotIn("error", state["messages"][6]["content"])

    def test_search_attempts_are_limited_per_user_turn(self):
        def search_call(call_id, query):
            return {
                "type": "tool_call",
                "content": None,
                "tool_call_id": call_id,
                "tool_name": "search_paper",
                "tool_arguments": {"query": query},
            }

        final = {
            "type": "final",
            "content": "请提供更多线索。",
            "tool_call_id": None,
            "tool_name": None,
            "tool_arguments": None,
        }
        search = Mock(return_value=DEMO_TOOL_RESULT)
        state = create_state("请搜索一篇论文")

        with (
            patch(
                "runtime.decide_next_action",
                side_effect=[
                    search_call("search-1", "first query"),
                    search_call("search-2", "refined query"),
                    search_call("search-3", "third query"),
                    final,
                ],
            ),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            answer = run_agent(state)

        self.assertEqual(search.call_count, 2)
        limited_result = state["messages"][6]["content"]
        self.assertEqual(
            limited_result["error"]["type"],
            "search_limit_reached",
        )
        self.assertEqual(limited_result["error"]["limit"], 2)
        self.assertEqual(answer, "请提供更多线索。")

    def test_cli_keeps_state_for_follow_up_messages(self):
        observed_states = []

        def fake_run_agent(
            state,
            confirm_save=None,
            on_event=None,
            checkpoint_file=None,
        ):
            self.assertTrue(callable(confirm_save))
            self.assertTrue(callable(on_event))
            observed_states.append(state)
            answer = (
                "请选择一篇论文。"
                if len(observed_states) == 1
                else "已选择第一篇。"
            )
            state["messages"].append(
                {"role": "assistant", "content": answer}
            )
            return answer

        output = StringIO()
        with (
            patch(
                "builtins.input",
                side_effect=["请搜索并保存 TASA", "第 1 篇", "exit"],
            ),
            patch("main.run_agent", side_effect=fake_run_agent),
            redirect_stdout(output),
        ):
            run_cli()

        self.assertEqual(len(observed_states), 2)
        self.assertIs(observed_states[0], observed_states[1])
        self.assertEqual(
            [
                message["content"]
                for message in observed_states[1]["messages"]
                if message["role"] == "user"
            ],
            ["请搜索并保存 TASA", "第 1 篇"],
        )
        self.assertIn("请选择一篇论文。", output.getvalue())
        self.assertIn("已选择第一篇。", output.getvalue())
        self.assertIn("已退出。", output.getvalue())

    def test_debug_cli_continues_after_long_search_trace(self):
        prompts = []
        answers = iter(
            ["Attention Is All You Need", "继续", "exit"]
        )

        def fake_input(prompt):
            prompts.append(prompt)
            return next(answers)

        def fake_run_agent(
            state,
            confirm_save=None,
            on_event=None,
            checkpoint_file=None,
        ):
            self.assertTrue(callable(on_event))
            if len(prompts) == 1:
                on_event(
                    {
                        "kind": "llm_started",
                        "turn_step": 1,
                        "total_step": 1,
                    }
                )
                on_event(
                    {
                        "kind": "llm_finished",
                        "turn_step": 1,
                        "total_step": 1,
                        "duration_ms": 1.0,
                        "response_type": "tool_call",
                    }
                )
                state["messages"].extend(
                    [
                        {
                            "role": "assistant",
                            "tool_call": {
                                "id": "search-call",
                                "name": "search_paper",
                                "arguments": {
                                    "query": "Attention Is All You Need"
                                },
                            },
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "search-call",
                            "name": "search_paper",
                            "content": {
                                "found": True,
                                "papers": [
                                    {
                                        "title": "Attention Is All You Need",
                                        "abstract": "A" * 5_000,
                                    }
                                ],
                            },
                        },
                    ]
                )
                on_event(
                    {
                        "kind": "tool_started",
                        "turn_step": 1,
                        "total_step": 1,
                        "tool_name": "search_paper",
                        "arguments": {
                            "query": "Attention Is All You Need"
                        },
                    }
                )
                on_event(
                    {
                        "kind": "tool_finished",
                        "turn_step": 1,
                        "total_step": 1,
                        "duration_ms": 1.0,
                        "tool_name": "search_paper",
                        "arguments": {
                            "query": "Attention Is All You Need"
                        },
                        "result": state["messages"][-1]["content"],
                    }
                )
                answer = "找到精确标题。"
            else:
                answer = "可以继续交互。"
            state["messages"].append(
                {"role": "assistant", "content": answer}
            )
            return answer

        output = StringIO()
        with (
            patch("builtins.input", side_effect=fake_input),
            patch("main.run_agent", side_effect=fake_run_agent),
            redirect_stdout(output),
        ):
            run_cli(debug=True)

        self.assertEqual(
            prompts,
            [
                "You › ",
                "You › ",
                "You › ",
            ],
        )
        trace = output.getvalue()
        self.assertIn('"abstract_preview"', trace)
        self.assertNotIn("A" * 401, trace)
        self.assertIn("找到精确标题。", trace)
        self.assertIn("可以继续交互。", trace)

    @patch.dict(os.environ, {"LLM_PROVIDER": "fake"}, clear=False)
    @patch.dict(
        "runtime.TOOL_REGISTRY",
        {"search_paper": lambda query: DEMO_TOOL_RESULT},
    )
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

    def test_search_debug_trace_uses_bounded_abstract_previews(self):
        state = create_state("Attention Is All You Need")
        state["messages"].extend(
            [
                {
                    "role": "assistant",
                    "tool_call": {
                        "id": "search-call",
                        "name": "search_paper",
                        "arguments": {
                            "query": "Attention Is All You Need"
                        },
                    },
                },
                {
                    "role": "tool",
                    "tool_call_id": "search-call",
                    "name": "search_paper",
                    "content": {
                        "found": True,
                        "papers": [
                            {
                                "title": f"Paper {index}",
                                "abstract": character * 5_000,
                            }
                            for index, character in enumerate("ABC", start=1)
                        ],
                    },
                },
                {"role": "assistant", "content": "搜索完成。"},
            ]
        )
        output = StringIO()

        with redirect_stdout(output):
            print_debug_trace(state)

        trace = output.getvalue()
        self.assertIn('"abstract_preview"', trace)
        self.assertNotIn('"abstract":', trace)
        self.assertNotIn("A" * 401, trace)
        self.assertLess(len(trace), 3_000)
        self.assertIn("[Step 2 | LLM] final", trace)
        self.assertTrue(
            trace.rstrip().endswith("[Summary] LLM steps: 2")
        )


if __name__ == "__main__":
    unittest.main()
