"""Deterministic Runtime runner for structured Agent task-success cases."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from checkpoint import checkpoint_exists
from evals.task_success import (
    evaluate_task_success,
    format_task_success_report,
    normalize_task_contract,
)
from runtime import run_agent
from state import append_user_message, create_state


AGENT_TASK_DATASET_VERSION = 1
AGENT_TASK_SUITE_VERSION = 1
DEFAULT_CASES_PATH = Path(__file__).with_name("agent_task_cases.json")
MAX_CASE_FILE_BYTES = 1024 * 1024
MAX_CASES = 20
MAX_ACTIONS = 20
MAX_TEXT_CHARS = 1_000
_CASE_KEYS = frozenset(
    {
        "id",
        "description",
        "goal",
        "mode",
        "plan_steps",
        "actions",
        "search_fixtures",
        "expectations",
    }
)
_ACTION_KEYS = frozenset(
    {
        "type",
        "content",
        "tool_call_id",
        "tool_name",
        "tool_arguments",
    }
)
_SUPPORTED_TOOLS = frozenset(
    {"search_paper", "download_paper", "retrieve_paper_chunks"}
)


def load_agent_task_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[dict]:
    """Load and strictly validate the versioned deterministic task suite."""
    case_path = Path(path)
    if case_path.stat().st_size > MAX_CASE_FILE_BYTES:
        raise ValueError("Agent task case file exceeds the 1 MiB limit.")
    try:
        payload = json.loads(case_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Agent task cases are invalid JSON: {error.msg}.") from error
    if not isinstance(payload, dict) or set(payload) != {
        "dataset_version",
        "cases",
    }:
        raise ValueError("Agent task dataset needs dataset_version and cases.")
    if payload["dataset_version"] != AGENT_TASK_DATASET_VERSION:
        raise ValueError("Agent task dataset version is not supported.")
    cases = payload["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError("Agent task dataset needs 1..20 cases.")

    normalized = [_normalize_case(case, index) for index, case in enumerate(cases)]
    identifiers = [case["id"] for case in normalized]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Agent task case IDs must be unique.")
    return normalized


def evaluate_agent_task_suite(
    cases: list[dict] | None = None,
    *,
    case_ids: list[str] | None = None,
) -> dict:
    """Run selected cases through the production Runtime and score them."""
    selected = list(cases) if cases is not None else load_agent_task_cases()
    if case_ids:
        requested = list(dict.fromkeys(case_ids))
        available = {case["id"]: case for case in selected}
        missing = [case_id for case_id in requested if case_id not in available]
        if missing:
            raise ValueError(f"Unknown Agent task cases: {missing}.")
        selected = [available[case_id] for case_id in requested]
    if not selected:
        raise ValueError("Agent task evaluation selected no cases.")

    reports = [run_agent_task_case(case) for case in selected]
    passed_count = sum(report["passed"] for report in reports)
    return {
        "suite_version": AGENT_TASK_SUITE_VERSION,
        "dataset_version": AGENT_TASK_DATASET_VERSION,
        "case_count": len(reports),
        "passed_count": passed_count,
        "pass_rate": round(passed_count / len(reports), 6),
        "mean_task_success_score": round(
            mean(report["score"] for report in reports), 1
        ),
        "passed": passed_count == len(reports),
        "cases": reports,
        "notice": (
            "Provider decisions and external Tool data are fixed. The suite "
            "tests Runtime-integrated task contracts, not live model quality."
        ),
    }


def run_agent_task_case(case: dict) -> dict:
    """Execute one validated case with real Runtime control flow."""
    normalized = _normalize_case(case, 0)
    state = create_state(normalized["goal"])
    events = []

    with TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint_file = root / "active.json"
        pdf_cache = root / "pdfs"
        tools = _FixtureTools(normalized, pdf_cache)
        registry = {
            "search_paper": tools.search_paper,
            "download_paper": tools.download_paper,
            "retrieve_paper_chunks": tools.retrieve_paper_chunks,
        }

        with (
            patch.dict(os.environ, {"PAPER_CACHE_DIR": str(pdf_cache)}),
            patch.dict("runtime.TOOL_REGISTRY", registry, clear=True),
        ):
            if normalized["mode"] == "planned":
                plan_action = {
                    "type": "plan",
                    "content": None,
                    "tool_call_id": f"plan-{normalized['id']}",
                    "step_descriptions": normalized["plan_steps"],
                }
                with (
                    patch(
                        "runtime.uuid4",
                        return_value=SimpleNamespace(
                            hex=f"eval-{normalized['id']}"
                        ),
                    ),
                    patch("runtime.decide_next_action", return_value=plan_action),
                ):
                    run_agent(
                        state,
                        checkpoint_file=checkpoint_file,
                        on_event=events.append,
                    )
                append_user_message(state, "Continue the benchmark plan.")
                with patch(
                    "runtime.decide_plan_step",
                    side_effect=deepcopy(normalized["actions"]),
                ):
                    final_answer = run_agent(
                        state,
                        checkpoint_file=checkpoint_file,
                        on_event=events.append,
                    )
            else:
                with patch(
                    "runtime.decide_next_action",
                    side_effect=deepcopy(normalized["actions"]),
                ):
                    final_answer = run_agent(
                        state,
                        checkpoint_file=checkpoint_file,
                        on_event=events.append,
                    )

        report = evaluate_task_success(normalized, state)
        report["runtime"] = {
            "event_count": len(events),
            "checkpoint_cleared": not checkpoint_exists(checkpoint_file),
            "plan_revision": (
                state["plan"]["revision"]
                if isinstance(state.get("plan"), dict)
                else None
            ),
            "final_answer_present": bool(final_answer.strip()),
        }
        report["passed"] = report["passed"] and all(
            (
                report["runtime"]["checkpoint_cleared"],
                report["runtime"]["final_answer_present"],
            )
        )
        return report


def format_agent_task_suite(report: dict, *, debug: bool = False) -> str:
    """Format suite summary and optional bounded per-case observations."""
    lines = [
        "=== Agent Task Success Evaluation ===",
        f"Cases: {report['case_count']}",
        f"Passed: {report['passed_count']}/{report['case_count']}",
        f"Pass rate: {report['pass_rate']:.3f}",
        f"Mean task success: {report['mean_task_success_score']:.1f}/100",
        "",
    ]
    for case in report["cases"]:
        lines.append(format_task_success_report(case, debug=debug))
    lines.extend(["", f"Note: {report['notice']}"])
    return "\n".join(lines)


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic AgentReader task cases through the production "
            "Runtime and score structured gold outcomes."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Run one case ID; may be repeated.",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show bounded per-dimension observations for gold cases.",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate_agent_task_suite(
            load_agent_task_cases(args.cases),
            case_ids=args.case_ids,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Agent task evaluation failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_agent_task_suite(report, debug=args.debug))
    return 0 if report["passed"] else 1


class _FixtureTools:
    def __init__(self, case: dict, pdf_cache: Path) -> None:
        self._search_results = {
            fixture["query"]: fixture["result"]
            for fixture in case["search_fixtures"]
        }
        self._pdf_cache = pdf_cache
        self._evidence_pages = {
            paper_id: index + 1
            for index, paper_id in enumerate(
                case["expectations"]["evidence_ids"]
            )
        }

    def search_paper(self, query: str) -> dict:
        if query not in self._search_results:
            raise ValueError(f"No search fixture for query {query!r}.")
        return deepcopy(self._search_results[query])

    def download_paper(self, paper: dict) -> dict:
        paper_id = paper.get("candidate_id")
        if not isinstance(paper_id, str) or not paper_id:
            raise ValueError("Download fixture needs a candidate_id.")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id)
        local_path = self._pdf_cache / f"{safe_name}.pdf"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        return {
            "downloaded": True,
            "cached": False,
            "paper_id": paper_id,
            "title": paper.get("title"),
            "pdf_url": paper.get("pdf_url"),
            "local_path": str(local_path),
            "size_bytes": local_path.stat().st_size,
        }

    def retrieve_paper_chunks(
        self,
        download_result: dict,
        query: str,
        top_k: int = 3,
    ) -> dict:
        paper_id = download_result.get("paper_id")
        if paper_id not in self._evidence_pages:
            raise ValueError(f"No evidence fixture for paper {paper_id!r}.")
        page = self._evidence_pages[paper_id]
        return {
            "found": True,
            "paper_id": paper_id,
            "title": download_result.get("title"),
            "query": query,
            "count": 1,
            "top_k": top_k,
            "matches": [
                {
                    "chunk_id": f"{paper_id}:p{page}:0-80",
                    "page": page,
                    "text": "Bounded deterministic benchmark evidence.",
                }
            ],
        }


def _normalize_case(case: object, index: int) -> dict:
    if not isinstance(case, dict) or set(case) != _CASE_KEYS:
        raise ValueError(f"Agent task case {index} has invalid fields.")
    case_id, expectations = normalize_task_contract(case)
    description = _case_text(case["description"], "description")
    goal = _case_text(case["goal"], "goal")
    mode = case["mode"]
    if mode not in {"simple", "planned"}:
        raise ValueError(f"Agent task case {case_id} has invalid mode.")
    plan_steps = case["plan_steps"]
    if not isinstance(plan_steps, list):
        raise ValueError(f"Agent task case {case_id} plan_steps must be a list.")
    normalized_steps = [_case_text(step, "plan step") for step in plan_steps]
    if mode == "simple" and normalized_steps:
        raise ValueError(f"Simple Agent task case {case_id} cannot have a plan.")
    if mode == "planned" and not 1 <= len(normalized_steps) <= 8:
        raise ValueError(f"Planned Agent task case {case_id} needs 1..8 steps.")

    actions = case["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_ACTIONS:
        raise ValueError(f"Agent task case {case_id} needs 1..20 actions.")
    normalized_actions = [
        _normalize_action(action, case_id, action_index)
        for action_index, action in enumerate(actions)
    ]
    call_ids = [
        action["tool_call_id"]
        for action in normalized_actions
        if action["type"] == "tool_call"
    ]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError(f"Agent task case {case_id} repeats Tool Call IDs.")

    fixtures = case["search_fixtures"]
    if not isinstance(fixtures, list) or len(fixtures) > MAX_ACTIONS:
        raise ValueError(f"Agent task case {case_id} has invalid search fixtures.")
    normalized_fixtures = [
        _normalize_search_fixture(fixture, case_id) for fixture in fixtures
    ]
    fixture_queries = [fixture["query"] for fixture in normalized_fixtures]
    if len(fixture_queries) != len(set(fixture_queries)):
        raise ValueError(f"Agent task case {case_id} repeats search fixtures.")
    called_queries = [
        action["tool_arguments"].get("query")
        for action in normalized_actions
        if action["type"] == "tool_call"
        and action["tool_name"] == "search_paper"
    ]
    if set(called_queries) - set(fixture_queries):
        raise ValueError(f"Agent task case {case_id} lacks a search fixture.")

    return {
        "id": case_id,
        "description": description,
        "goal": goal,
        "mode": mode,
        "plan_steps": normalized_steps,
        "actions": normalized_actions,
        "search_fixtures": normalized_fixtures,
        "expectations": expectations,
    }


def _normalize_action(action: object, case_id: str, index: int) -> dict:
    if not isinstance(action, dict) or set(action) != _ACTION_KEYS:
        raise ValueError(f"Agent task {case_id} action {index} has invalid fields.")
    action_type = action["type"]
    if action_type == "final":
        if (
            not isinstance(action["content"], str)
            or not action["content"].strip()
            or action["tool_call_id"] is not None
            or action["tool_name"] is not None
            or action["tool_arguments"] is not None
        ):
            raise ValueError(f"Agent task {case_id} final action {index} is invalid.")
        return deepcopy(action)
    if action_type != "tool_call":
        raise ValueError(f"Agent task {case_id} action {index} has invalid type.")
    if (
        action["content"] is not None
        or not isinstance(action["tool_call_id"], str)
        or not action["tool_call_id"]
        or action["tool_name"] not in _SUPPORTED_TOOLS
        or not isinstance(action["tool_arguments"], dict)
    ):
        raise ValueError(f"Agent task {case_id} Tool action {index} is invalid.")
    return deepcopy(action)


def _normalize_search_fixture(fixture: object, case_id: str) -> dict:
    if not isinstance(fixture, dict) or set(fixture) != {"query", "result"}:
        raise ValueError(f"Agent task case {case_id} has invalid search fixture.")
    query = _case_text(fixture["query"], "search fixture query")
    result = fixture["result"]
    if not isinstance(result, dict) or not isinstance(result.get("papers"), list):
        raise ValueError(f"Agent task case {case_id} search result is invalid.")
    return {"query": query, "result": deepcopy(result)}


def _case_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Agent task {label} must be non-empty text.")
    normalized = value.strip()
    if len(normalized) > MAX_TEXT_CHARS:
        raise ValueError(f"Agent task {label} exceeds {MAX_TEXT_CHARS} characters.")
    return normalized
