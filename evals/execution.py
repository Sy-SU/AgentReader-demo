"""Deterministic execution-health scoring for one Runtime turn.

This module intentionally does not judge answer correctness.  It summarizes
observable Runtime control flow so Debug mode can expose regressions without
calling another model as a judge.
"""

from __future__ import annotations

from events import AgentEvent


EXECUTION_EVALUATION_VERSION = 1

_CONTROL_WEIGHTS = {
    "turn_control": 45,
    "protocol_integrity": 25,
    "tool_success": 20,
    "budget_health": 10,
}
_HEALTHY_TERMINAL_STATUSES = {"completed", "planned", "blocked"}


class ExecutionEvaluationTracker:
    """Collect bounded counters from Runtime events for one user turn."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._event_count = 0
        self._llm_started = 0
        self._llm_finished = 0
        self._tool_started = 0
        self._tool_finished = 0
        self._tool_errors = 0
        self._runtime_guard_errors = 0
        self._run_failures = 0
        self._replans = 0
        self._checkpoint_saves = 0
        self._llm_duration_ms = 0.0
        self._tool_duration_ms = 0.0
        self._terminal_status: str | None = None

    @property
    def has_events(self) -> bool:
        return self._event_count > 0

    def observe(self, event: AgentEvent) -> None:
        """Consume one event without retaining arguments, results, or text."""
        self._event_count += 1
        kind = event.get("kind")

        if kind == "llm_started":
            self._llm_started += 1
        elif kind == "llm_finished":
            self._llm_finished += 1
            self._llm_duration_ms += _duration(event)
        elif kind == "tool_started":
            self._tool_started += 1
        elif kind == "tool_finished":
            self._tool_finished += 1
            self._tool_duration_ms += _duration(event)
            error_type = _tool_error_type(event.get("result"))
            if error_type is not None:
                self._tool_errors += 1
                if error_type in {
                    "duplicate_search_query",
                    "search_limit_reached",
                    "task_tool_budget_exhausted",
                }:
                    self._runtime_guard_errors += 1
        elif kind == "plan_replanned":
            self._replans += 1
        elif kind == "checkpoint_saved":
            self._checkpoint_saves += 1
        elif kind == "run_failed":
            self._run_failures += 1
        elif kind == "turn_finished":
            status = event.get("status")
            if isinstance(status, str):
                self._terminal_status = status

    def evaluate(self) -> dict:
        """Return a machine-readable scorecard for the observed turn."""
        if not self.has_events:
            return {
                "evaluation_version": EXECUTION_EVALUATION_VERSION,
                "score": None,
                "status": "unavailable",
                "reason": "no_runtime_events",
                "notice": _notice(),
                "dimensions": {},
                "metrics": self._metrics(),
            }

        if self._terminal_status == "cancelled":
            return {
                "evaluation_version": EXECUTION_EVALUATION_VERSION,
                "score": None,
                "status": "not_scored",
                "reason": "user_cancelled",
                "notice": _notice(),
                "dimensions": {},
                "metrics": self._metrics(),
            }

        dimensions = {
            "turn_control": self._turn_control_score(),
            "protocol_integrity": self._protocol_score(),
            "tool_success": self._tool_success_score(),
            "budget_health": self._budget_score(),
        }
        score = _weighted_score(dimensions)
        status = (
            "healthy"
            if score >= 90
            else "warning" if score >= 60 else "unhealthy"
        )
        return {
            "evaluation_version": EXECUTION_EVALUATION_VERSION,
            "score": score,
            "status": status,
            "reason": None,
            "notice": _notice(),
            "dimensions": dimensions,
            "metrics": self._metrics(),
        }

    def _turn_control_score(self) -> float:
        if self._run_failures:
            return 0.0
        if self._terminal_status in _HEALTHY_TERMINAL_STATUSES:
            return 100.0
        if self._terminal_status == "max_steps_reached":
            return 50.0
        return 0.0

    def _protocol_score(self) -> float:
        if self._run_failures:
            return 0.0
        balanced = (
            self._llm_started == self._llm_finished
            and self._tool_started == self._tool_finished
            and self._terminal_status is not None
        )
        return 100.0 if balanced else 0.0

    def _tool_success_score(self) -> float | None:
        if self._tool_finished == 0:
            return None
        successful = self._tool_finished - self._tool_errors
        return round(successful / self._tool_finished * 100, 1)

    def _budget_score(self) -> float:
        exhausted = (
            self._terminal_status in {"failed", "max_steps_reached"}
            or self._runtime_guard_errors > 0
        )
        return 0.0 if exhausted else 100.0

    def _metrics(self) -> dict:
        return {
            "event_count": self._event_count,
            "terminal_status": self._terminal_status,
            "llm_started": self._llm_started,
            "llm_finished": self._llm_finished,
            "tool_started": self._tool_started,
            "tool_finished": self._tool_finished,
            "tool_errors": self._tool_errors,
            "runtime_guard_errors": self._runtime_guard_errors,
            "run_failures": self._run_failures,
            "replans": self._replans,
            "checkpoint_saves": self._checkpoint_saves,
            "llm_duration_ms": round(self._llm_duration_ms, 1),
            "tool_duration_ms": round(self._tool_duration_ms, 1),
        }


def format_execution_evaluation(report: dict) -> str:
    """Format the compact Debug-only execution scorecard."""
    score = report.get("score")
    if score is None:
        if report.get("reason") == "user_cancelled":
            return "[Eval] 运行健康分：未评分（用户取消）"
        return "[Eval] 运行健康分：不可用（未收到 Runtime Event）"

    dimensions = report["dimensions"]
    tool_score = dimensions["tool_success"]
    tool_text = "N/A" if tool_score is None else _format_score(tool_score)
    metrics = report["metrics"]
    lines = [
        (
            f"[Eval] 运行健康分 {_format_score(score)}/100 · "
            f"本轮状态 {metrics['terminal_status'] or 'unknown'}"
        ),
        (
            "  控制 "
            f"{_format_score(dimensions['turn_control'])} | "
            "协议 "
            f"{_format_score(dimensions['protocol_integrity'])} | "
            f"Tool {tool_text} | "
            f"预算 {_format_score(dimensions['budget_health'])}"
        ),
        (
            "  LLM "
            f"{metrics['llm_finished']} 次/{metrics['llm_duration_ms']:.1f} ms | "
            "Tool "
            f"{metrics['tool_finished']} 次/{metrics['tool_duration_ms']:.1f} ms | "
            f"Tool 错误 {metrics['tool_errors']}"
        ),
        f"  注：{report['notice']}",
    ]
    return "\n".join(lines)


def _weighted_score(dimensions: dict[str, float | None]) -> float:
    available = {
        name: value
        for name, value in dimensions.items()
        if value is not None
    }
    weight_total = sum(_CONTROL_WEIGHTS[name] for name in available)
    if weight_total == 0:
        return 0.0
    weighted = sum(
        _CONTROL_WEIGHTS[name] * value
        for name, value in available.items()
    )
    return round(weighted / weight_total, 1)


def _tool_error_type(result: object) -> str | None:
    if not isinstance(result, dict) or "error" not in result:
        return None
    error = result["error"]
    if isinstance(error, dict) and isinstance(error.get("type"), str):
        return error["type"]
    return "unknown_tool_error"


def _duration(event: dict) -> float:
    value = event.get("duration_ms")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, float(value))


def _format_score(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.1f}"


def _notice() -> str:
    return "该分数只衡量执行健康，不代表答案正确率；检索质量请看 LitSearch。"
