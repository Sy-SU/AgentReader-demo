import unittest
from unittest.mock import Mock, patch

from runtime import run_agent
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


class RuntimeEventTests(unittest.TestCase):
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
