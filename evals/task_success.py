"""Structured task-success scoring for AgentReader benchmark cases.

Unlike execution-health scoring, this module requires an explicit gold
contract.  It never judges arbitrary answer prose and never calls an LLM.
Only bounded identifiers, tool counts, and evidence page numbers are retained
in the report.
"""

from __future__ import annotations

from collections import Counter


TASK_SUCCESS_EVALUATION_VERSION = 1
MAX_EXPECTED_ITEMS = 20
MAX_IDENTIFIER_CHARS = 160
MAX_TOOL_NAME_CHARS = 80
MAX_OBSERVED_TOOL_CALLS = 100
MAX_OBSERVED_IDENTIFIERS = 100
MAX_EVIDENCE_PAGE = 1_000_000
_EXPECTED_KEYS = frozenset(
    {
        "terminal_status",
        "required_tools",
        "allowed_tools",
        "candidate_ids",
        "download_ids",
        "evidence_ids",
        "expect_no_results",
    }
)


def evaluate_task_success(case: dict, state: dict) -> dict:
    """Score one final State against a case's explicit gold expectations."""
    case_id, expected = normalize_task_contract(case)
    observations = _collect_observations(state)

    dimensions = {
        "completion": _completion_dimension(expected, observations),
        "tool_use": _tool_use_dimension(expected, observations),
        "candidate_discovery": _coverage_dimension(
            expected["candidate_ids"],
            observations["candidate_ids"],
        ),
        "paper_selection": _coverage_dimension(
            expected["download_ids"],
            observations["download_ids"],
        ),
        "page_evidence": _coverage_dimension(
            expected["evidence_ids"],
            observations["evidence_pages"],
        ),
        "no_result_safety": _no_result_dimension(expected, observations),
    }
    applicable = [
        dimension["score"]
        for dimension in dimensions.values()
        if dimension is not None
    ]
    score = round(sum(applicable) / len(applicable), 1)
    passed = score == 100.0
    return {
        "evaluation_version": TASK_SUCCESS_EVALUATION_VERSION,
        "case_id": case_id,
        "score": score,
        "passed": passed,
        "dimensions": dimensions,
        "metrics": {
            "tool_call_count": sum(observations["tool_counts"].values()),
            "tool_error_count": observations["tool_error_count"],
            "candidate_count": len(observations["candidate_ids"]),
            "download_count": len(observations["download_ids"]),
            "evidence_paper_count": len(observations["evidence_pages"]),
            "normal_no_result_count": observations["normal_no_result_count"],
        },
        "observations": {
            "terminal_status": observations["terminal_status"],
            "tool_counts": dict(sorted(observations["tool_counts"].items())),
            "tool_error_types": observations["tool_error_types"],
            "candidate_ids": sorted(observations["candidate_ids"]),
            "download_ids": sorted(observations["download_ids"]),
            "evidence_pages": {
                paper_id: observations["evidence_pages"][paper_id]
                for paper_id in sorted(observations["evidence_pages"])
            },
        },
        "notice": (
            "This score uses an explicit structured gold contract. It does "
            "not judge final-answer prose or general model quality."
        ),
    }


def normalize_task_contract(case: object) -> tuple[str, dict]:
    """Validate and normalize the scorer-visible part of one task case."""
    if not isinstance(case, dict):
        raise ValueError("Agent task case must be an object.")
    case_id = _bounded_text(case.get("id"), "Agent task case id")
    expected = case.get("expectations")
    if not isinstance(expected, dict):
        raise ValueError(f"Agent task case {case_id} needs expectations.")
    if set(expected) != _EXPECTED_KEYS:
        missing = sorted(_EXPECTED_KEYS - set(expected))
        extra = sorted(set(expected) - _EXPECTED_KEYS)
        raise ValueError(
            f"Agent task case {case_id} expectation fields mismatch: "
            f"missing={missing}, extra={extra}."
        )

    terminal_status = expected["terminal_status"]
    if terminal_status != "completed":
        raise ValueError(
            f"Agent task case {case_id} terminal_status must be completed."
        )
    required_tools = _normalize_required_tools(expected["required_tools"])
    allowed_tools = _normalize_text_list(
        expected["allowed_tools"],
        "allowed_tools",
        max_chars=MAX_TOOL_NAME_CHARS,
    )
    if not set(required_tools) <= set(allowed_tools):
        raise ValueError(
            f"Agent task case {case_id} required_tools must be allowed."
        )
    candidate_ids = _normalize_text_list(
        expected["candidate_ids"], "candidate_ids"
    )
    download_ids = _normalize_text_list(
        expected["download_ids"], "download_ids"
    )
    evidence_ids = _normalize_text_list(
        expected["evidence_ids"], "evidence_ids"
    )
    if not set(download_ids) <= set(candidate_ids):
        raise ValueError(
            f"Agent task case {case_id} download_ids must be candidates."
        )
    if not set(evidence_ids) <= set(download_ids):
        raise ValueError(
            f"Agent task case {case_id} evidence_ids must be downloads."
        )
    expect_no_results = expected["expect_no_results"]
    if not isinstance(expect_no_results, bool):
        raise ValueError(
            f"Agent task case {case_id} expect_no_results must be boolean."
        )
    if expect_no_results and (candidate_ids or download_ids or evidence_ids):
        raise ValueError(
            f"Agent task case {case_id} no-result expectations conflict."
        )
    return case_id, {
        "terminal_status": terminal_status,
        "required_tools": required_tools,
        "allowed_tools": allowed_tools,
        "candidate_ids": candidate_ids,
        "download_ids": download_ids,
        "evidence_ids": evidence_ids,
        "expect_no_results": expect_no_results,
    }


def format_task_success_report(report: dict, *, debug: bool = False) -> str:
    """Format one bounded task scorecard."""
    marker = "PASS" if report["passed"] else "FAIL"
    lines = [
        (
            f"[{marker}] {report['case_id']} · "
            f"task success {report['score']:.1f}/100"
        )
    ]
    if not debug:
        return "\n".join(lines)

    dimension_labels = {
        "completion": "completion",
        "tool_use": "tools",
        "candidate_discovery": "discovery",
        "paper_selection": "selection",
        "page_evidence": "evidence",
        "no_result_safety": "no-result",
    }
    rendered_dimensions = []
    for name, dimension in report["dimensions"].items():
        score = "N/A" if dimension is None else f"{dimension['score']:.1f}"
        rendered_dimensions.append(f"{dimension_labels[name]}={score}")
    observations = report["observations"]
    lines.extend(
        [
            "  " + " | ".join(rendered_dimensions),
            f"  tools={observations['tool_counts']}",
            f"  candidates={observations['candidate_ids']}",
            f"  downloads={observations['download_ids']}",
            f"  evidence_pages={observations['evidence_pages']}",
            f"  note={report['notice']}",
        ]
    )
    return "\n".join(lines)


def _collect_observations(state: object) -> dict:
    if not isinstance(state, dict) or not isinstance(state.get("messages"), list):
        raise ValueError("Agent task evaluation needs a State with messages.")
    calls: dict[str, str] = {}
    completed_calls = set()
    tool_counts: Counter[str] = Counter()
    tool_error_types = []
    candidate_ids = set()
    download_ids = set()
    evidence_pages: dict[str, list[int]] = {}
    normal_no_result_count = 0

    for index, message in enumerate(state["messages"]):
        if not isinstance(message, dict):
            raise ValueError(f"State message {index} must be an object.")
        tool_call = message.get("tool_call")
        if tool_call is not None:
            if message.get("role") != "assistant" or not isinstance(
                tool_call, dict
            ):
                raise ValueError(f"State Tool Call {index} must be an object.")
            if len(calls) >= MAX_OBSERVED_TOOL_CALLS:
                raise ValueError("State exceeds the Tool Call observation limit.")
            call_id = _observation_text(tool_call.get("id"))
            name = _observation_text(
                tool_call.get("name"), max_chars=MAX_TOOL_NAME_CHARS
            )
            if call_id is None or name is None or call_id in calls:
                raise ValueError(f"State Tool Call {index} is invalid.")
            calls[call_id] = name
        if message.get("role") != "tool":
            continue

        call_id = message.get("tool_call_id")
        name = message.get("name")
        result = message.get("content")
        if (
            call_id not in calls
            or call_id in completed_calls
            or name != calls[call_id]
            or not isinstance(result, dict)
        ):
            raise ValueError(f"State Tool Result {index} is invalid.")
        completed_calls.add(call_id)
        tool_counts[name] += 1
        error_type = _error_type(result)
        if error_type is not None:
            tool_error_types.append(error_type)
            continue
        if name == "search_paper":
            papers = result.get("papers")
            if isinstance(papers, list):
                for paper in papers:
                    candidate_id = (
                        paper.get("candidate_id")
                        if isinstance(paper, dict)
                        else None
                    )
                    candidate_id = _observation_text(candidate_id)
                    if candidate_id is not None:
                        _add_bounded_identifier(candidate_ids, candidate_id)
            if result.get("found") is False and papers == []:
                normal_no_result_count += 1
        elif name == "download_paper":
            paper_id = result.get("paper_id")
            paper_id = _observation_text(paper_id)
            if (
                paper_id is not None
                and (
                    result.get("downloaded") is True
                    or result.get("cached") is True
                )
            ):
                _add_bounded_identifier(download_ids, paper_id)
        elif name == "retrieve_paper_chunks":
            paper_id = _observation_text(result.get("paper_id"))
            pages = _valid_evidence_pages(result)
            if paper_id is not None and pages:
                if (
                    paper_id not in evidence_pages
                    and len(evidence_pages) >= MAX_OBSERVED_IDENTIFIERS
                ):
                    raise ValueError(
                        "State exceeds the evidence identifier observation limit."
                    )
                evidence_pages[paper_id] = sorted(
                    set(evidence_pages.get(paper_id, []) + pages)
                )

    if set(calls) != completed_calls:
        raise ValueError("State contains Tool Calls without Tool Results.")

    return {
        "terminal_status": _terminal_status(state),
        "tool_counts": tool_counts,
        "tool_error_count": len(tool_error_types),
        "tool_error_types": sorted(tool_error_types)[:MAX_EXPECTED_ITEMS],
        "candidate_ids": candidate_ids,
        "download_ids": download_ids,
        "evidence_pages": evidence_pages,
        "normal_no_result_count": normal_no_result_count,
    }


def _completion_dimension(expected: dict, observed: dict) -> dict:
    passed = observed["terminal_status"] == expected["terminal_status"]
    return {"score": 100.0 if passed else 0.0, "passed": passed}


def _tool_use_dimension(expected: dict, observed: dict) -> dict:
    checks = {
        f"required:{name}": observed["tool_counts"].get(name, 0) >= count
        for name, count in expected["required_tools"].items()
    }
    checks["only_allowed_tools"] = set(observed["tool_counts"]) <= set(
        expected["allowed_tools"]
    )
    checks["no_tool_errors"] = observed["tool_error_count"] == 0
    return _checks_dimension(checks)


def _coverage_dimension(expected_ids: list[str], observed: object) -> dict | None:
    if not expected_ids:
        return None
    observed_ids = set(observed)
    checks = {
        paper_id: paper_id in observed_ids for paper_id in expected_ids
    }
    return _checks_dimension(checks)


def _no_result_dimension(expected: dict, observed: dict) -> dict | None:
    if not expected["expect_no_results"]:
        return None
    checks = {
        "normal_empty_search": observed["normal_no_result_count"] > 0,
        "no_candidates": not observed["candidate_ids"],
        "no_downloads": not observed["download_ids"],
        "no_evidence": not observed["evidence_pages"],
    }
    return _checks_dimension(checks)


def _checks_dimension(checks: dict[str, bool]) -> dict:
    passed_count = sum(checks.values())
    score = round(passed_count / len(checks) * 100, 1)
    return {"score": score, "passed": score == 100.0, "checks": checks}


def _terminal_status(state: dict) -> str:
    plan = state.get("plan")
    if isinstance(plan, dict):
        status = plan.get("status")
        return status if isinstance(status, str) else "invalid"
    messages = state["messages"]
    if messages:
        final = messages[-1]
        if (
            isinstance(final, dict)
            and final.get("role") == "assistant"
            and isinstance(final.get("content"), str)
            and final["content"].strip()
            and "tool_call" not in final
        ):
            return "completed"
    return "incomplete"


def _valid_evidence_pages(result: dict) -> list[int]:
    matches = result.get("matches")
    if result.get("found") is not True or not isinstance(matches, list) or not matches:
        return []
    pages = []
    for match in matches:
        if not isinstance(match, dict):
            return []
        page = match.get("page")
        text = match.get("text")
        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page < 1
            or page > MAX_EVIDENCE_PAGE
            or not isinstance(text, str)
            or not text.strip()
        ):
            return []
        pages.append(page)
    return pages


def _error_type(result: dict) -> str | None:
    if "error" not in result:
        return None
    error = result["error"]
    if isinstance(error, dict) and isinstance(error.get("type"), str):
        return error["type"][:MAX_IDENTIFIER_CHARS]
    return "unknown_tool_error"


def _normalize_required_tools(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError("Agent task required_tools must be a non-empty object.")
    if len(value) > MAX_EXPECTED_ITEMS:
        raise ValueError("Agent task required_tools exceeds the item limit.")
    normalized = {}
    for name, count in value.items():
        normalized_name = _bounded_text(
            name, "Agent task tool name", max_chars=MAX_TOOL_NAME_CHARS
        )
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= MAX_EXPECTED_ITEMS
        ):
            raise ValueError("Agent task required tool counts must be positive.")
        normalized[normalized_name] = count
    return normalized


def _normalize_text_list(
    value: object,
    label: str,
    *,
    max_chars: int = MAX_IDENTIFIER_CHARS,
) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_EXPECTED_ITEMS:
        raise ValueError(f"Agent task {label} must be a bounded list.")
    normalized = [
        _bounded_text(item, f"Agent task {label} item", max_chars=max_chars)
        for item in value
    ]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Agent task {label} must not contain duplicates.")
    return normalized


def _bounded_text(
    value: object,
    label: str,
    *,
    max_chars: int = MAX_IDENTIFIER_CHARS,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text.")
    normalized = value.strip()
    if len(normalized) > max_chars:
        raise ValueError(f"{label} exceeds {max_chars} characters.")
    return normalized


def _observation_text(
    value: object,
    *,
    max_chars: int = MAX_IDENTIFIER_CHARS,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if len(normalized) > max_chars:
        return None
    return normalized


def _add_bounded_identifier(target: set[str], value: str) -> None:
    if value not in target and len(target) >= MAX_OBSERVED_IDENTIFIERS:
        raise ValueError("State exceeds the identifier observation limit.")
    target.add(value)
