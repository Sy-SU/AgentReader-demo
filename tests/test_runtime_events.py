import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from planning import create_plan
from runtime import cancel_active_task, run_agent
from state import create_state


TOOL_CALL = {
    "type": "tool_call",
    "content": None,
    "tool_call_id": "search-call",
    "tool_name": "search_paper",
    "tool_arguments": {"query": "TASA"},
}
FINAL = {
    "type": "final",
    "content": "完成。",
    "tool_call_id": None,
    "tool_name": None,
    "tool_arguments": None,
}
PLAN = {
    "type": "plan",
    "content": None,
    "tool_call_id": "plan-call",
    "step_descriptions": ["Search for evidence", "Compare evidence"],
}
BLOCKED = {
    "type": "blocked",
    "content": "Which candidate should I use?",
    "tool_call_id": "blocked-call",
}
REPLAN = {
    "type": "replan",
    "content": None,
    "tool_call_id": "replan-call",
    "step_descriptions": ["Use another evidence source"],
}


class RuntimeEventTests(unittest.TestCase):
    def test_plan_creation_and_checkpoint_emit_bounded_metadata(self):
        state = create_state("Compare two papers")
        events = []

        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            with (
                patch(
                    "runtime.uuid4",
                    return_value=SimpleNamespace(hex="events"),
                ),
                patch("runtime.decide_next_action", return_value=PLAN),
            ):
                run_agent(
                    state,
                    checkpoint_file=path,
                    on_event=events.append,
                )

        self.assertEqual(
            [event["kind"] for event in events],
            [
                "llm_started",
                "llm_finished",
                "plan_created",
                "checkpoint_saved",
                "turn_finished",
            ],
        )
        self.assertEqual(events[2]["task_id"], "task-events")
        self.assertEqual(events[2]["step_count"], 2)
        self.assertGreater(events[3]["checkpoint_size_bytes"], 0)
        self.assertNotIn("reasoning_content", events[3])

    def test_step_start_completion_and_blocking_emit_transitions(self):
        completed_state = create_state("Find evidence")
        completed_state["task_id"] = "task-complete"
        completed_state["plan"] = create_plan(
            "task-complete",
            "Find evidence",
            ["Search for evidence"],
        )
        completed_events = []

        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            with patch("runtime.decide_plan_step", return_value=FINAL):
                run_agent(
                    completed_state,
                    checkpoint_file=path,
                    on_event=completed_events.append,
                )

        transitions = [
            event
            for event in completed_events
            if event["kind"] == "plan_step_changed"
        ]
        self.assertEqual(
            [
                (event["previous_status"], event["status"])
                for event in transitions
            ],
            [("pending", "running"), ("running", "completed")],
        )

        blocked_state = create_state("Choose a candidate")
        blocked_state["task_id"] = "task-blocked"
        blocked_state["plan"] = create_plan(
            "task-blocked",
            "Choose a candidate",
            ["Resolve the candidate"],
        )
        blocked_events = []
        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            with patch("runtime.decide_plan_step", return_value=BLOCKED):
                run_agent(
                    blocked_state,
                    checkpoint_file=path,
                    on_event=blocked_events.append,
                )

        self.assertIn("task_blocked", [e["kind"] for e in blocked_events])
        blocked_event = next(
            event
            for event in blocked_events
            if event["kind"] == "task_blocked"
        )
        self.assertEqual(blocked_event["status"], "blocked")
        self.assertEqual(blocked_event["step_id"], "step-001")

    def test_replan_and_explicit_cancel_emit_lifecycle_events(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-replan"
        state["plan"] = create_plan(
            "task-replan",
            "Find evidence",
            ["Search for evidence"],
        )
        events = []
        search = Mock(
            return_value={
                "found": False,
                "papers": [],
                "relevance_assessment": {"status": "no_candidates"},
            }
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            with (
                patch(
                    "runtime.decide_plan_step",
                    side_effect=[TOOL_CALL, REPLAN],
                ),
                patch.dict(
                    "runtime.TOOL_REGISTRY",
                    {"search_paper": search},
                    clear=True,
                ),
            ):
                run_agent(
                    state,
                    max_steps=2,
                    checkpoint_file=path,
                    on_event=events.append,
                )

            replan_event = next(
                event
                for event in events
                if event["kind"] == "plan_replanned"
            )
            self.assertEqual(replan_event["plan_revision"], 2)
            self.assertEqual(replan_event["replan_count"], 1)
            self.assertIn("no usable candidate", replan_event["reason"])

            cancel_events = []
            cancel_active_task(
                state,
                checkpoint_file=path,
                on_event=cancel_events.append,
            )

        self.assertEqual(cancel_events[-1]["kind"], "task_cancelled")
        self.assertEqual(cancel_events[-1]["status"], "cancelled")

    def test_final_only_turn_emits_ordered_events(self):
        state = create_state("你好")
        events = []

        with patch("runtime.decide_next_action", return_value=FINAL):
            answer = run_agent(state, on_event=events.append)

        self.assertEqual(answer, "完成。")
        self.assertEqual(
            [event["kind"] for event in events],
            ["llm_started", "llm_finished", "turn_finished"],
        )
        self.assertEqual(events[1]["response_type"], "final")
        self.assertEqual(events[2]["status"], "completed")
        self.assertEqual(events[2]["total_step"], 1)

    def test_tool_turn_emits_result_after_state_is_updated(self):
        state = create_state("请搜索 TASA")
        events = []
        state_sizes_at_finish = []

        def observe(event):
            events.append(event)
            if event["kind"] == "tool_finished":
                state_sizes_at_finish.append(len(state["messages"]))

        search = Mock(return_value={"found": False, "papers": []})
        with (
            patch(
                "runtime.decide_next_action",
                side_effect=[TOOL_CALL, FINAL],
            ),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            run_agent(state, on_event=observe)

        self.assertEqual(
            [event["kind"] for event in events],
            [
                "llm_started",
                "llm_finished",
                "tool_started",
                "tool_finished",
                "llm_started",
                "llm_finished",
                "turn_finished",
            ],
        )
        self.assertEqual(state_sizes_at_finish, [3])
        self.assertEqual(events[3]["tool_name"], "search_paper")
        self.assertEqual(events[3]["result"], {"found": False, "papers": []})
        self.assertIsInstance(events[3]["duration_ms"], float)

    def test_guard_error_is_emitted_as_a_finished_tool_result(self):
        repeated_call = {**TOOL_CALL, "tool_call_id": "search-call-2"}
        state = create_state("请搜索 TASA")
        events = []
        search = Mock(return_value={"found": False, "papers": []})

        with (
            patch(
                "runtime.decide_next_action",
                side_effect=[TOOL_CALL, repeated_call, FINAL],
            ),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            run_agent(state, on_event=events.append)

        finished = [
            event for event in events if event["kind"] == "tool_finished"
        ]
        self.assertEqual(len(finished), 2)
        self.assertEqual(
            finished[1]["result"]["error"]["type"],
            "duplicate_search_query",
        )
        self.assertEqual(search.call_count, 1)

    def test_observer_failure_does_not_change_agent_result(self):
        state = create_state("你好")

        def broken_observer(event):
            raise RuntimeError("display failed")

        with patch("runtime.decide_next_action", return_value=FINAL):
            answer = run_agent(state, on_event=broken_observer)

        self.assertEqual(answer, "完成。")
        self.assertEqual(state["messages"][-1]["content"], "完成。")

    def test_observer_cannot_mutate_tool_arguments_or_stored_result(self):
        state = create_state("请搜索 TASA")
        search_result = {"found": False, "papers": []}
        search = Mock(return_value=search_result)

        def mutating_observer(event):
            if event["kind"] == "tool_started":
                event["arguments"].clear()
            if event["kind"] == "tool_finished":
                event["result"].clear()

        with (
            patch(
                "runtime.decide_next_action",
                side_effect=[TOOL_CALL, FINAL],
            ),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            run_agent(state, on_event=mutating_observer)

        search.assert_called_once_with(query="TASA")
        self.assertEqual(
            state["messages"][1]["tool_call"]["arguments"],
            {"query": "TASA"},
        )
        self.assertEqual(state["messages"][2]["content"], search_result)

    def test_llm_failure_emits_run_failed_and_preserves_exception(self):
        state = create_state("你好")
        events = []

        with patch(
            "runtime.decide_next_action",
            side_effect=RuntimeError("provider unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                run_agent(state, on_event=events.append)

        self.assertEqual(
            [event["kind"] for event in events],
            ["llm_started", "run_failed"],
        )
        self.assertEqual(events[-1]["error_type"], "RuntimeError")

    def test_cancelled_tool_gets_a_result_and_leaves_valid_state(self):
        state = create_state("请搜索 TASA")
        events = []
        cancelled_search = Mock(side_effect=KeyboardInterrupt)

        with (
            patch("runtime.decide_next_action", return_value=TOOL_CALL),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": cancelled_search},
                clear=True,
            ),
        ):
            answer = run_agent(state, on_event=events.append)

        self.assertEqual(answer, "当前工具操作已取消。")
        self.assertEqual(
            [message["role"] for message in state["messages"]],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(
            state["messages"][2]["content"]["error"]["type"],
            "cancelled",
        )
        self.assertEqual(events[-2]["kind"], "tool_finished")
        self.assertEqual(events[-2]["result"]["error"]["type"], "cancelled")
        self.assertEqual(events[-1]["kind"], "turn_finished")
        self.assertEqual(events[-1]["status"], "cancelled")

    def test_cancelled_llm_call_closes_the_turn_with_assistant_message(self):
        state = create_state("请搜索 TASA")
        events = []

        with patch(
            "runtime.decide_next_action",
            side_effect=KeyboardInterrupt,
        ):
            answer = run_agent(state, on_event=events.append)

        self.assertEqual(answer, "当前模型调用已取消。")
        self.assertEqual(
            [message["role"] for message in state["messages"]],
            ["user", "assistant"],
        )
        self.assertEqual(state["messages"][-1]["content"], answer)
        self.assertEqual(state["step"], 0)
        self.assertEqual(
            [event["kind"] for event in events],
            ["llm_started", "turn_finished"],
        )
        self.assertEqual(events[-1]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
