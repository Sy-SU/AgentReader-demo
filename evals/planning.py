"""Deterministic offline evaluation for the V3 planning Runtime."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from checkpoint import checkpoint_exists
from runtime import run_agent
from state import append_user_message, create_state


PLANNING_EVALUATION_VERSION = 1
SCENARIO_ID = "two-paper-comparison-success"
GOAL = "Compare Attention Is All You Need and BERT."
PLAN_ACTION = {
    "type": "plan",
    "content": None,
    "tool_call_id": "plan-eval-call",
    "step_descriptions": [
        "Find Attention Is All You Need",
        "Find BERT",
        "Compare their architectures and training objectives",
    ],
}
EXECUTOR_ACTIONS = [
    {
        "type": "tool_call",
        "content": None,
        "tool_call_id": "search-transformer",
        "tool_name": "search_paper",
        "tool_arguments": {"query": "Attention Is All You Need"},
    },
    {
        "type": "final",
        "content": "The Transformer paper has been identified.",
        "tool_call_id": None,
        "tool_name": None,
        "tool_arguments": None,
    },
    {
        "type": "tool_call",
        "content": None,
        "tool_call_id": "search-bert",
        "tool_name": "search_paper",
        "tool_arguments": {"query": "BERT"},
    },
    {
        "type": "final",
        "content": "The BERT paper has been identified.",
        "tool_call_id": None,
        "tool_name": None,
        "tool_arguments": None,
    },
    {
        "type": "final",
        "content": (
            "Comparison complete: Transformer uses autoregressive "
            "sequence transduction; BERT uses bidirectional pretraining."
        ),
        "tool_call_id": None,
        "tool_name": None,
        "tool_arguments": None,
    },
]


def evaluate_planning_runtime() -> dict:
    """Run a two-paper Plan through the real Runtime with fixed adapters."""
    state = create_state(GOAL)
    events = []
    search = Mock(side_effect=_search_fixture)

    with TemporaryDirectory() as directory:
        checkpoint_file = Path(directory) / "active.json"
        with (
            patch(
                "runtime.uuid4",
                return_value=SimpleNamespace(hex="planning-eval"),
            ),
            patch("runtime.decide_next_action", return_value=PLAN_ACTION),
        ):
            run_agent(
                state,
                checkpoint_file=checkpoint_file,
                on_event=events.append,
            )

        append_user_message(state, "Continue the plan.")
        with (
            patch(
                "runtime.decide_plan_step",
                side_effect=EXECUTOR_ACTIONS,
            ),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            final_answer = run_agent(
                state,
                checkpoint_file=checkpoint_file,
                on_event=events.append,
            )

        checkpoint_cleared = not checkpoint_exists(checkpoint_file)

    plan = state["plan"]
    checkpoint_sizes = [
        event["checkpoint_size_bytes"]
        for event in events
        if event["kind"] == "checkpoint_saved"
    ]
    search_queries = [
        call.kwargs["query"] for call in search.call_args_list
    ]
    expected_queries = ["Attention Is All You Need", "BERT"]
    evidence_refs = [
        evidence_ref
        for step in plan["steps"]
        for evidence_ref in step["evidence_refs"]
    ]
    passed = all(
        [
            plan["status"] == "completed",
            all(step["status"] == "completed" for step in plan["steps"]),
            plan["budget"]
            == {"llm_steps": 6, "tool_calls": 2, "replans": 0},
            search_queries == expected_queries,
            evidence_refs == ["tool-result-001", "tool-result-002"],
            checkpoint_cleared,
            bool(checkpoint_sizes),
        ]
    )

    return {
        "evaluation_version": PLANNING_EVALUATION_VERSION,
        "scenario_id": SCENARIO_ID,
        "description": (
            "Offline two-paper success path through a fixed initial planning "
            "action, Runtime, Tools, step transitions, evidence, and "
            "Checkpoint lifecycle."
        ),
        "passed": passed,
        "task_status": plan["status"],
        "plan_revision": plan["revision"],
        "step_statuses": [
            {"id": step["id"], "status": step["status"]}
            for step in plan["steps"]
        ],
        "metrics": {
            "task_completed": plan["status"] == "completed",
            "replan_count": plan["budget"]["replans"],
            "llm_decisions": plan["budget"]["llm_steps"],
            "tool_executions": plan["budget"]["tool_calls"],
            "checkpoint_save_count": len(checkpoint_sizes),
            "max_checkpoint_bytes": max(checkpoint_sizes, default=0),
        },
        "observations": {
            "search_queries": search_queries,
            "evidence_refs": evidence_refs,
            "checkpoint_cleared": checkpoint_cleared,
            "final_answer": final_answer,
        },
    }


def format_planning_report(result: dict) -> str:
    """Format a compact human-readable planning evaluation report."""
    metrics = result["metrics"]
    return "\n".join(
        [
            "=== Planning Runtime Evaluation ===",
            f"Scenario: {result['scenario_id']}",
            f"Passed: {result['passed']}",
            f"Task status: {result['task_status']}",
            f"Plan revision: {result['plan_revision']}",
            f"Replans: {metrics['replan_count']}",
            f"LLM decisions: {metrics['llm_decisions']}",
            f"Tool executions: {metrics['tool_executions']}",
            f"Checkpoint saves: {metrics['checkpoint_save_count']}",
            f"Max checkpoint: {metrics['max_checkpoint_bytes']} bytes",
        ]
    )


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the deterministic V3 planning Runtime."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete machine-readable evaluation.",
    )
    args = parser.parse_args(argv)
    result = evaluate_planning_runtime()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_planning_report(result))
    return 0 if result["passed"] else 1


def _search_fixture(query: str) -> dict:
    papers = {
        "Attention Is All You Need": {
            "source": "arxiv",
            "candidate_id": "arxiv:1706.03762",
            "arxiv_id": "1706.03762v7",
            "doi": None,
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani"],
            "abstract": "The Transformer is based on attention.",
            "published": "2017-06-12T17:57:34Z",
            "paper_url": "https://arxiv.org/abs/1706.03762v7",
            "pdf_url": "https://arxiv.org/pdf/1706.03762v7",
        },
        "BERT": {
            "source": "arxiv",
            "candidate_id": "arxiv:1810.04805",
            "arxiv_id": "1810.04805v2",
            "doi": None,
            "title": (
                "BERT: Pre-training of Deep Bidirectional Transformers "
                "for Language Understanding"
            ),
            "authors": ["Jacob Devlin"],
            "abstract": "BERT pretrains bidirectional representations.",
            "published": "2018-10-11T00:00:00Z",
            "paper_url": "https://arxiv.org/abs/1810.04805v2",
            "pdf_url": "https://arxiv.org/pdf/1810.04805v2",
        },
    }
    paper = papers[query]
    return {
        "found": True,
        "source": "arxiv",
        "count": 1,
        "candidate_count": 1,
        "truncated": False,
        "ranking_method": "planning_eval_fixture_v1",
        "relevance_assessment": {
            "status": "lexical_match",
            "best_score": 1,
        },
        "papers": [paper],
    }
