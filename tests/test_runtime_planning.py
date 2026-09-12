import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from checkpoint import (
    CheckpointError,
    checkpoint_exists,
    load_checkpoint,
    save_checkpoint,
)
from planning import (
    MAX_TASK_LLM_STEPS,
    MAX_TASK_TOOL_CALLS,
    create_plan,
    fail_current_step,
    record_plan_usage,
    revise_plan,
    start_next_step,
)
from llm import _to_api_messages
from runtime import (
    DEFAULT_MAX_TASK_LLM_STEPS,
    cancel_active_task,
    run_agent,
)
from state import append_user_message, create_state


PLAN_ACTION = {
    "type": "plan",
    "content": None,
    "tool_call_id": "plan-call-1",
    "step_descriptions": [
        "Identify both papers",
        "Collect page-level evidence",
        "Compare the methods and experiments",
    ],
}
STEP_FINAL = {
    "type": "final",
    "content": "Current step completed.",
    "tool_call_id": None,
    "tool_name": None,
    "tool_arguments": None,
}
SEARCH_CALL = {
    "type": "tool_call",
    "content": None,
    "tool_call_id": "unsafe/call id",
    "tool_name": "search_paper",
    "tool_arguments": {"query": "paper A"},
}
DOWNLOAD_CALL = {
    "type": "tool_call",
    "content": None,
    "tool_call_id": "download-call-1",
    "tool_name": "download_paper",
    "tool_arguments": {"paper_id": "arxiv:1234.56789"},
}
RETRIEVE_CALL = {
    "type": "tool_call",
    "content": None,
    "tool_call_id": "retrieve-call-1",
    "tool_name": "retrieve_paper_chunks",
    "tool_arguments": {
        "paper_id": "arxiv:1234.56789",
        "query": "architecture",
    },
}
REPLAN_ACTION = {
    "type": "replan",
    "content": None,
    "tool_call_id": "replan-call-1",
    "step_descriptions": ["Use the alternate evidence source"],
}
BLOCK_ACTION = {
    "type": "blocked",
    "content": "Which candidate should I use?",
    "tool_call_id": "clarify-call-1",
}
SUCCESS_SEARCH_RESULT = {
    "found": True,
    "papers": [
        {
            "candidate_id": "arxiv:1234.56789",
            "local_relevance": {"exact_title_match": True},
        }
    ],
}
NO_SEARCH_RESULT = {
    "found": False,
    "papers": [],
    "relevance_assessment": {"status": "no_candidates"},
}
AMBIGUOUS_SEARCH_RESULT = {
    "found": True,
    "papers": [
        {
            "candidate_id": "arxiv:1111.11111",
            "local_relevance": {"exact_title_match": False},
        },
        {
            "candidate_id": "arxiv:2222.22222",
            "local_relevance": {"exact_title_match": False},
        },
    ],
}
CHECKPOINT_SEARCH_RESULT = {
    "found": True,
    "count": 1,
    "papers": [
        {
            "candidate_id": "arxiv:1234.56789",
            "source": "arxiv",
            "arxiv_id": "1234.56789v1",
            "doi": None,
            "title": "Checkpoint Test Paper",
            "authors": ["Test Author"],
            "abstract": "Test abstract.",
            "published": "2025-01-01T00:00:00Z",
            "paper_url": "https://arxiv.org/abs/1234.56789v1",
            "pdf_url": "https://arxiv.org/pdf/1234.56789v1",
        }
    ],
}


class RuntimePlanningTests(unittest.TestCase):
    def test_new_state_has_no_task_or_plan(self):
        state = create_state("Compare two papers")

        self.assertIsNone(state["task_id"])
        self.assertIsNone(state["plan"])

    @patch("runtime.uuid4", return_value=SimpleNamespace(hex="abc123"))
    @patch("runtime.decide_next_action", return_value=PLAN_ACTION)
    def test_runtime_creates_trusted_plan_without_executing_tool(
        self,
        decide,
        _uuid,
    ):
        state = create_state("Compare paper A and paper B")
        events = []

        answer = run_agent(state, on_event=events.append)

        decide.assert_called_once_with(state, allow_planning=True)
        self.assertEqual(state["task_id"], "task-abc123")
        self.assertEqual(state["plan"]["task_id"], "task-abc123")
        self.assertEqual(
            state["plan"]["goal"],
            "Compare paper A and paper B",
        )
        self.assertEqual(state["plan"]["budget"]["llm_steps"], 1)
        self.assertEqual(state["plan"]["budget"]["tool_calls"], 0)
        self.assertTrue(
            all(
                step["status"] == "pending"
                for step in state["plan"]["steps"]
            )
        )
        self.assertFalse(
            any(message["role"] == "tool" for message in state["messages"])
        )
        self.assertNotIn("submit_plan", str(state["messages"]))
        self.assertIn("已创建任务计划", answer)
        self.assertEqual(state["messages"][-1]["content"], answer)
        self.assertEqual(state["step"], 1)
        self.assertEqual(
            [event["kind"] for event in events],
            [
                "llm_started",
                "llm_finished",
                "plan_created",
                "turn_finished",
            ],
        )
        self.assertEqual(events[1]["response_type"], "plan")
        self.assertEqual(events[2]["step_count"], 3)
        self.assertEqual(events[3]["status"], "planned")

    @patch("runtime.uuid4", return_value=SimpleNamespace(hex="checkpoint"))
    @patch("runtime.decide_next_action", return_value=PLAN_ACTION)
    def test_runtime_checkpoints_a_new_plan(
        self,
        _decide,
        _uuid,
    ):
        state = create_state("Compare paper A and paper B")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            run_agent(state, checkpoint_file=path)

            self.assertTrue(checkpoint_exists(path))
            self.assertEqual(load_checkpoint(path), state)
            self.assertEqual(state["plan"]["status"], "running")

    @patch("runtime.uuid4", return_value=SimpleNamespace(hex="latest"))
    @patch("runtime.decide_next_action", return_value=PLAN_ACTION)
    def test_plan_goal_uses_latest_user_turn(self, _decide, _uuid):
        state = create_state("First simple request")
        append_user_message(state, "Now compare paper A and paper B")

        run_agent(state)

        self.assertEqual(
            state["plan"]["goal"],
            "Now compare paper A and paper B",
        )

    @patch("runtime.uuid4", return_value=SimpleNamespace(hex="reasoning"))
    @patch(
        "runtime.decide_next_action",
        return_value={
            **PLAN_ACTION,
            "reasoning_content": "opaque planning reasoning",
        },
    )
    def test_plan_preview_preserves_reasoning_for_the_next_turn(
        self,
        _decide,
        _uuid,
    ):
        state = create_state("Compare paper A and paper B")

        run_agent(state)

        self.assertEqual(
            state["messages"][-1]["reasoning_content"],
            "opaque planning reasoning",
        )

    @patch("runtime.decide_plan_step", return_value=STEP_FINAL)
    @patch("runtime.decide_next_action")
    def test_active_plan_uses_executor_without_silent_replanning(
        self,
        decide,
        decide_step,
    ):
        state = create_state("Compare two papers")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Compare two papers",
            ["Identify papers", "Compare evidence"],
        )
        original_descriptions = [
            step["description"] for step in state["plan"]["steps"]
        ]

        answer = run_agent(state, max_steps=1)

        decide.assert_not_called()
        decide_step.assert_called_once()
        self.assertEqual(state["task_id"], "task-existing")
        self.assertEqual(
            [step["description"] for step in state["plan"]["steps"]],
            original_descriptions,
        )
        self.assertEqual(state["plan"]["steps"][0]["status"], "completed")
        self.assertEqual(state["plan"]["steps"][1]["status"], "pending")
        self.assertIn("max_steps=1", answer)
        self.assertIn("当前计划仍在进行中", answer)
        self.assertIn("请输入“继续”", answer)

    @patch("runtime.decide_next_action")
    def test_executor_completes_two_steps_and_records_runtime_evidence(
        self,
        initial_decision,
    ):
        state = create_state("Compare paper A and paper B")
        state["task_id"] = "task-existing"
        state["plan"] = record_plan_usage(
            create_plan(
                "task-existing",
                "Compare paper A and paper B",
                ["Identify both papers", "Compare their methods"],
            ),
            llm_steps=1,
        )
        state["step"] = 1
        search = Mock(return_value=SUCCESS_SEARCH_RESULT)
        step_plans = []

        def decide_step(_state, plan, *, replan_reason=None):
            self.assertIsNone(replan_reason)
            step_plans.append(deepcopy(plan))
            return [SEARCH_CALL, STEP_FINAL, {**STEP_FINAL, "content": "Done."}][
                len(step_plans) - 1
            ]

        with (
            patch("runtime.decide_plan_step", side_effect=decide_step),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=3)

        initial_decision.assert_not_called()
        search.assert_called_once_with(query="paper A")
        self.assertEqual(answer, "Done.")
        self.assertEqual(state["plan"]["status"], "completed")
        self.assertEqual(
            [step["status"] for step in state["plan"]["steps"]],
            ["completed", "completed"],
        )
        self.assertEqual(
            state["plan"]["steps"][0]["evidence_refs"],
            ["tool-result-001"],
        )
        self.assertEqual(state["plan"]["steps"][1]["evidence_refs"], [])
        self.assertEqual(
            state["plan"]["budget"],
            {"llm_steps": 4, "tool_calls": 1, "replans": 0},
        )
        self.assertEqual(state["step"], 4)
        tool_message = next(
            message for message in state["messages"] if message["role"] == "tool"
        )
        self.assertEqual(tool_message["evidence_ref"], "tool-result-001")
        self.assertNotIn("unsafe/call id", str(state["plan"]))
        self.assertEqual(step_plans[0]["current_step_id"], "step-001")
        self.assertEqual(step_plans[1]["current_step_id"], "step-001")
        self.assertEqual(step_plans[2]["current_step_id"], "step-002")

    def test_tool_result_does_not_complete_the_current_step(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find evidence",
            ["Search for the evidence"],
        )
        search = Mock(return_value=NO_SEARCH_RESULT)

        with (
            patch("runtime.decide_plan_step", return_value=SEARCH_CALL),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=1)

        self.assertIn("max_steps=1", answer)
        self.assertIn("当前计划仍在进行中", answer)
        self.assertEqual(state["plan"]["status"], "running")
        self.assertEqual(state["plan"]["steps"][0]["status"], "running")
        self.assertEqual(
            state["plan"]["steps"][0]["evidence_refs"],
            ["tool-result-001"],
        )

    def test_checkpoint_resumes_at_the_current_step_and_clears_on_completion(
        self,
    ):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find evidence",
            ["Search for the evidence"],
        )
        search = Mock(return_value=CHECKPOINT_SEARCH_RESULT)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            with (
                patch(
                    "runtime.decide_plan_step",
                    side_effect=[SEARCH_CALL, STEP_FINAL],
                ),
                patch.dict(
                    "runtime.TOOL_REGISTRY",
                    {"search_paper": search},
                    clear=True,
                ),
            ):
                paused = run_agent(
                    state,
                    max_steps=1,
                    checkpoint_file=path,
                )
                restored = load_checkpoint(path)
                append_user_message(restored, "继续")
                answer = run_agent(
                    restored,
                    max_steps=1,
                    checkpoint_file=path,
                )

            self.assertIn("max_steps=1", paused)
            self.assertEqual(answer, STEP_FINAL["content"])
            self.assertEqual(restored["plan"]["status"], "completed")
            self.assertFalse(checkpoint_exists(path))
            search.assert_called_once_with(query="paper A")

    def test_recoverable_runtime_error_keeps_the_latest_checkpoint(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find evidence",
            ["Search for the evidence"],
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            with (
                patch(
                    "runtime.decide_plan_step",
                    side_effect=RuntimeError("provider unavailable"),
                ),
                self.assertRaisesRegex(RuntimeError, "provider unavailable"),
            ):
                run_agent(state, checkpoint_file=path)

            restored = load_checkpoint(path)
            self.assertEqual(restored["plan"]["status"], "running")
            self.assertEqual(
                restored["plan"]["current_step_id"],
                "step-001",
            )

    def test_keyboard_interrupt_keeps_a_resumable_checkpoint(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find evidence",
            ["Search for the evidence"],
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            with patch(
                "runtime.decide_plan_step",
                side_effect=KeyboardInterrupt,
            ):
                answer = run_agent(state, checkpoint_file=path)

            restored = load_checkpoint(path)
            self.assertEqual(answer, "当前模型调用已取消。")
            self.assertEqual(restored["plan"]["status"], "running")
            self.assertEqual(
                restored["plan"]["current_step_id"],
                "step-001",
            )
            self.assertEqual(
                restored["messages"][-1]["content"],
                "当前模型调用已取消。",
            )

    def test_terminal_plan_failure_clears_existing_checkpoint(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = record_plan_usage(
            create_plan(
                "task-existing",
                "Find evidence",
                ["Search for the evidence"],
            ),
            llm_steps=DEFAULT_MAX_TASK_LLM_STEPS,
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            save_checkpoint(state, path)

            answer = run_agent(state, checkpoint_file=path)

            self.assertIn("LLM 决策上限", answer)
            self.assertEqual(state["plan"]["status"], "failed")
            self.assertFalse(checkpoint_exists(path))

    def test_cancel_active_task_clears_checkpoint_and_closes_plan(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = start_next_step(
            create_plan(
                "task-existing",
                "Find evidence",
                ["Search for the evidence", "Summarize it"],
            )
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            save_checkpoint(state, path)

            cancelled = cancel_active_task(state, checkpoint_file=path)

            self.assertEqual(cancelled["status"], "cancelled")
            self.assertEqual(state["plan"]["status"], "cancelled")
            self.assertEqual(
                state["plan"]["steps"][0]["status"],
                "failed",
            )
            self.assertIsNone(state["plan"]["current_step_id"])
            self.assertFalse(checkpoint_exists(path))

    def test_cancel_failure_preserves_active_state_and_checkpoint(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find evidence",
            ["Search for the evidence"],
        )
        original_plan = deepcopy(state["plan"])

        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            save_checkpoint(state, path)
            with (
                patch(
                    "runtime.clear_checkpoint",
                    side_effect=CheckpointError("cannot clear"),
                ),
                self.assertRaisesRegex(CheckpointError, "cannot clear"),
            ):
                cancel_active_task(state, checkpoint_file=path)

            self.assertEqual(state["plan"], original_plan)
            self.assertTrue(checkpoint_exists(path))

    @patch("runtime.decide_plan_step")
    def test_executor_stops_before_exceeding_llm_budget(self, decide_step):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = record_plan_usage(
            create_plan(
                "task-existing",
                "Find evidence",
                ["Search for evidence"],
            ),
            llm_steps=DEFAULT_MAX_TASK_LLM_STEPS,
        )

        answer = run_agent(state)

        decide_step.assert_not_called()
        self.assertEqual(state["plan"]["status"], "failed")
        self.assertEqual(
            state["plan"]["budget"]["llm_steps"],
            DEFAULT_MAX_TASK_LLM_STEPS,
        )
        self.assertIn("LLM 决策上限", answer)

    @patch("runtime.decide_plan_step", return_value=STEP_FINAL)
    def test_higher_task_llm_limit_allows_thinking_plan_to_continue(
        self,
        decide_step,
    ):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = record_plan_usage(
            create_plan(
                "task-existing",
                "Find evidence",
                ["Search for evidence"],
            ),
            llm_steps=DEFAULT_MAX_TASK_LLM_STEPS,
        )

        answer = run_agent(
            state,
            max_steps=1,
            max_task_llm_steps=MAX_TASK_LLM_STEPS,
        )

        self.assertEqual(answer, STEP_FINAL["content"])
        self.assertEqual(state["plan"]["status"], "completed")
        self.assertEqual(
            state["plan"]["budget"]["llm_steps"],
            DEFAULT_MAX_TASK_LLM_STEPS + 1,
        )
        decide_step.assert_called_once()

    def test_executor_stops_before_exceeding_tool_budget(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = record_plan_usage(
            create_plan(
                "task-existing",
                "Find evidence",
                ["Search for evidence"],
            ),
            tool_calls=MAX_TASK_TOOL_CALLS,
        )
        search = Mock(return_value={"found": False, "papers": []})

        with (
            patch("runtime.decide_plan_step", return_value=SEARCH_CALL),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            answer = run_agent(state)

        search.assert_not_called()
        self.assertEqual(state["plan"]["status"], "failed")
        self.assertEqual(
            state["plan"]["budget"]["tool_calls"],
            MAX_TASK_TOOL_CALLS,
        )
        self.assertIn("Tool 执行上限", answer)
        self.assertFalse(
            any(message.get("tool_call") for message in state["messages"])
        )

    def test_executor_preserves_reasoning_across_plan_steps(self):
        state = create_state("Compare two papers")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Compare two papers",
            ["Find the papers", "Compare them"],
        )
        responses = [
            {**SEARCH_CALL, "reasoning_content": "search reasoning"},
            {**STEP_FINAL, "reasoning_content": "first step reasoning"},
            {
                **STEP_FINAL,
                "content": "Comparison complete.",
                "reasoning_content": "second step reasoning",
            },
        ]
        search = Mock(return_value=SUCCESS_SEARCH_RESULT)

        with (
            patch("runtime.decide_plan_step", side_effect=responses),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=3)

        self.assertEqual(answer, "Comparison complete.")
        assistant_messages = [
            message
            for message in state["messages"]
            if message["role"] == "assistant"
        ]
        self.assertEqual(
            [
                message.get("reasoning_content")
                for message in assistant_messages
            ],
            [
                "search reasoning",
                "first step reasoning",
                "second step reasoning",
            ],
        )
        replayed = _to_api_messages(state["messages"])
        self.assertEqual(
            [
                message.get("reasoning_content")
                for message in replayed
                if message["role"] == "assistant"
            ],
            [
                "search reasoning",
                "first step reasoning",
                "second step reasoning",
            ],
        )

    def test_runtime_replans_only_after_a_trusted_failure_signal(self):
        state = create_state("Find evidence and compare it")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find evidence and compare it",
            ["Find the primary evidence", "Old comparison step"],
        )
        decisions = Mock(
            side_effect=[SEARCH_CALL, REPLAN_ACTION, STEP_FINAL]
        )
        search = Mock(return_value=NO_SEARCH_RESULT)

        with (
            patch("runtime.decide_plan_step", decisions),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=3)

        self.assertEqual(answer, STEP_FINAL["content"])
        self.assertEqual(state["plan"]["status"], "completed")
        self.assertEqual(state["plan"]["revision"], 2)
        self.assertEqual(state["plan"]["budget"]["replans"], 1)
        self.assertEqual(
            [step["description"] for step in state["plan"]["steps"]],
            ["Find the primary evidence", "Use the alternate evidence source"],
        )
        self.assertEqual(
            [step["status"] for step in state["plan"]["steps"]],
            ["failed", "completed"],
        )
        self.assertEqual(
            state["plan"]["steps"][0]["evidence_refs"],
            ["tool-result-001"],
        )
        replan_reasons = [
            call.kwargs["replan_reason"] for call in decisions.call_args_list
        ]
        self.assertEqual(
            replan_reasons,
            [None, "the paper search produced no usable candidate", None],
        )
        self.assertIn("任务计划已修订", str(state["messages"]))

    def test_replan_starts_a_new_bounded_search_phase(self):
        def search_call(call_id, query):
            return {
                "type": "tool_call",
                "content": None,
                "tool_call_id": call_id,
                "tool_name": "search_paper",
                "tool_arguments": {"query": query},
            }

        state = create_state("Find two papers and recover failed sources")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find two papers and recover failed sources",
            ["Find both papers", "Old comparison step"],
        )
        decisions = Mock(
            side_effect=[
                search_call("search-1", "first paper"),
                search_call("search-2", "second paper"),
                REPLAN_ACTION,
                search_call("search-3", "arxiv:1706.03762"),
                search_call("search-4", "arxiv:1810.04805"),
                STEP_FINAL,
            ]
        )
        search = Mock(
            side_effect=[
                SUCCESS_SEARCH_RESULT,
                NO_SEARCH_RESULT,
                SUCCESS_SEARCH_RESULT,
                SUCCESS_SEARCH_RESULT,
            ]
        )

        with (
            patch("runtime.decide_plan_step", decisions),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=6)

        self.assertEqual(answer, STEP_FINAL["content"])
        self.assertEqual(state["plan"]["status"], "completed")
        self.assertEqual(state["plan"]["revision"], 2)
        self.assertEqual(search.call_count, 4)
        self.assertNotIn("search_limit_reached", str(state["messages"]))

    def test_replan_failure_signal_survives_a_turn_step_pause(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find evidence",
            ["Find the primary evidence"],
        )
        decisions = Mock(
            side_effect=[SEARCH_CALL, REPLAN_ACTION, STEP_FINAL]
        )

        with (
            patch("runtime.decide_plan_step", decisions),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": Mock(return_value=NO_SEARCH_RESULT)},
                clear=True,
            ),
        ):
            paused = run_agent(state, max_steps=1)
            append_user_message(state, "继续")
            answer = run_agent(state, max_steps=2)

        self.assertIn("max_steps=1", paused)
        self.assertEqual(answer, STEP_FINAL["content"])
        self.assertEqual(state["plan"]["status"], "completed")
        self.assertEqual(
            [call.kwargs["replan_reason"] for call in decisions.call_args_list],
            [None, "the paper search produced no usable candidate", None],
        )

    def test_ambiguous_search_allows_replan_but_not_force_it(self):
        state = create_state("Find the most relevant paper")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find the most relevant paper",
            ["Resolve the best candidate"],
        )
        decisions = Mock(side_effect=[SEARCH_CALL, STEP_FINAL])

        with (
            patch("runtime.decide_plan_step", decisions),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {
                    "search_paper": Mock(
                        return_value=AMBIGUOUS_SEARCH_RESULT
                    )
                },
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=2)

        self.assertEqual(answer, STEP_FINAL["content"])
        self.assertEqual(state["plan"]["status"], "completed")
        self.assertEqual(
            decisions.call_args_list[1].kwargs["replan_reason"],
            "the paper search returned ambiguous candidates",
        )

    def test_replan_does_not_repeat_a_completed_download_side_effect(self):
        state = create_state("Download and analyze one paper")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Download and analyze one paper",
            ["Collect usable evidence", "Old analysis step"],
        )
        repeated_download = {
            **DOWNLOAD_CALL,
            "tool_call_id": "download-call-2",
        }
        decisions = Mock(
            side_effect=[
                SEARCH_CALL,
                DOWNLOAD_CALL,
                RETRIEVE_CALL,
                REPLAN_ACTION,
                repeated_download,
                STEP_FINAL,
            ]
        )
        download = Mock(
            return_value={
                "downloaded": True,
                "cached": False,
                "paper_id": "arxiv:1234.56789",
                "local_path": "/tmp/test-paper.pdf",
            }
        )
        retrieve = Mock(
            return_value={
                "found": False,
                "rejected_low_query_coverage": True,
                "chunks": [],
            }
        )

        with (
            patch("runtime.decide_plan_step", decisions),
            patch("runtime.validate_cached_pdf_path"),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {
                    "search_paper": Mock(return_value=SUCCESS_SEARCH_RESULT),
                    "download_paper": download,
                    "retrieve_paper_chunks": retrieve,
                },
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=6)

        self.assertEqual(answer, STEP_FINAL["content"])
        self.assertEqual(state["plan"]["status"], "completed")
        download.assert_called_once()
        retrieve.assert_called_once()
        download_results = [
            message["content"]
            for message in state["messages"]
            if message.get("role") == "tool"
            and message.get("name") == "download_paper"
        ]
        self.assertEqual(len(download_results), 2)
        self.assertNotIn("runtime_reused", download_results[0])
        self.assertTrue(download_results[1]["runtime_reused"])
        self.assertEqual(
            download_results[1]["runtime_reuse_reason"],
            "duplicate_side_effect",
        )

    def test_invalid_cached_download_is_not_reused(self):
        state = create_state("Download and analyze one paper")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Download and analyze one paper",
            ["Collect usable evidence"],
        )
        state["messages"].extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_call": {
                        "id": SEARCH_CALL["tool_call_id"],
                        "name": "search_paper",
                        "arguments": SEARCH_CALL["tool_arguments"],
                    },
                },
                {
                    "role": "tool",
                    "tool_call_id": SEARCH_CALL["tool_call_id"],
                    "name": "search_paper",
                    "content": SUCCESS_SEARCH_RESULT,
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_call": {
                        "id": DOWNLOAD_CALL["tool_call_id"],
                        "name": "download_paper",
                        "arguments": DOWNLOAD_CALL["tool_arguments"],
                    },
                },
                {
                    "role": "tool",
                    "tool_call_id": DOWNLOAD_CALL["tool_call_id"],
                    "name": "download_paper",
                    "content": {
                        "downloaded": True,
                        "cached": False,
                        "paper_id": "arxiv:1234.56789",
                        "local_path": "/tmp/missing-paper.pdf",
                    },
                },
            ]
        )
        repeated_download = {
            **DOWNLOAD_CALL,
            "tool_call_id": "download-call-2",
        }
        download = Mock(
            return_value={
                "downloaded": True,
                "cached": False,
                "paper_id": "arxiv:1234.56789",
                "local_path": "/tmp/replaced-paper.pdf",
            }
        )

        with (
            patch(
                "runtime.decide_plan_step",
                side_effect=[repeated_download, STEP_FINAL],
            ),
            patch(
                "runtime.validate_cached_pdf_path",
                side_effect=RuntimeError("invalid cache"),
            ),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"download_paper": download},
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=2)

        self.assertEqual(answer, STEP_FINAL["content"])
        download.assert_called_once()
        self.assertNotIn("runtime_reused", state["messages"][-2]["content"])

    def test_hard_failure_blocks_when_replan_budget_is_exhausted(self):
        plan = create_plan("task-existing", "Find evidence", ["first"])
        plan = fail_current_step(start_next_step(plan))
        plan = revise_plan(plan, ["second"])
        plan = fail_current_step(start_next_step(plan))
        plan = revise_plan(plan, ["third"])
        plan = start_next_step(plan)
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = plan
        decisions = Mock(side_effect=[SEARCH_CALL, STEP_FINAL])

        with (
            patch("runtime.decide_plan_step", decisions),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": Mock(return_value=NO_SEARCH_RESULT)},
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=2)

        self.assertEqual(answer, STEP_FINAL["content"])
        self.assertEqual(state["plan"]["status"], "blocked")
        self.assertEqual(state["plan"]["budget"]["replans"], 2)
        self.assertEqual(
            [call.kwargs["replan_reason"] for call in decisions.call_args_list],
            [None, None],
        )

    def test_runtime_rejects_replan_without_a_failure_or_ambiguity(self):
        state = create_state("Find evidence")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find evidence",
            ["Find the evidence"],
        )
        search = Mock(return_value=SUCCESS_SEARCH_RESULT)

        with (
            patch(
                "runtime.decide_plan_step",
                side_effect=[SEARCH_CALL, REPLAN_ACTION],
            ),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": search},
                clear=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "not authorized"):
                run_agent(state, max_steps=2)

        self.assertEqual(state["plan"]["revision"], 1)
        self.assertEqual(state["plan"]["budget"]["replans"], 0)

    def test_runtime_blocks_and_resumes_after_user_clarification(self):
        state = create_state("Choose and compare a paper")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Choose and compare a paper",
            ["Resolve the paper choice"],
        )

        with patch(
            "runtime.decide_plan_step",
            side_effect=[
                {**BLOCK_ACTION, "reasoning_content": "opaque block reasoning"},
                STEP_FINAL,
            ],
        ):
            question = run_agent(state, max_steps=1)
            self.assertEqual(question, BLOCK_ACTION["content"])
            self.assertEqual(state["plan"]["status"], "blocked")
            self.assertEqual(
                state["messages"][-1]["reasoning_content"],
                "opaque block reasoning",
            )

            append_user_message(state, "Use the first candidate.")
            answer = run_agent(state, max_steps=1)

        self.assertEqual(answer, STEP_FINAL["content"])
        self.assertEqual(state["plan"]["status"], "completed")
        self.assertEqual(state["plan"]["steps"][0]["attempts"], 2)

    def test_blocked_checkpoint_resumes_and_clears_on_completion(self):
        state = create_state("Choose and compare a paper")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Choose and compare a paper",
            ["Resolve the paper choice"],
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            with patch(
                "runtime.decide_plan_step",
                side_effect=[BLOCK_ACTION, STEP_FINAL],
            ):
                run_agent(state, max_steps=1, checkpoint_file=path)
                restored = load_checkpoint(path)
                self.assertEqual(restored["plan"]["status"], "blocked")

                append_user_message(restored, "Use the first candidate.")
                answer = run_agent(
                    restored,
                    max_steps=1,
                    checkpoint_file=path,
                )

            self.assertEqual(answer, STEP_FINAL["content"])
            self.assertEqual(restored["plan"]["status"], "completed")
            self.assertFalse(checkpoint_exists(path))

    def test_hard_tool_failure_final_blocks_instead_of_completing_step(self):
        state = create_state("Find one paper")
        state["task_id"] = "task-existing"
        state["plan"] = create_plan(
            "task-existing",
            "Find one paper",
            ["Find the paper"],
        )
        clarification = {
            **STEP_FINAL,
            "content": "Please provide another title or an author.",
        }

        with (
            patch(
                "runtime.decide_plan_step",
                side_effect=[SEARCH_CALL, clarification],
            ),
            patch.dict(
                "runtime.TOOL_REGISTRY",
                {"search_paper": Mock(return_value=NO_SEARCH_RESULT)},
                clear=True,
            ),
        ):
            answer = run_agent(state, max_steps=2)

        self.assertEqual(answer, clarification["content"])
        self.assertEqual(state["plan"]["status"], "blocked")
        self.assertEqual(state["plan"]["steps"][0]["status"], "blocked")

    @patch("runtime.decide_next_action")
    def test_simple_final_keeps_v2_path_without_creating_plan(self, decide):
        decide.return_value = {
            "type": "final",
            "content": "Please provide another title.",
            "tool_call_id": None,
            "tool_name": None,
            "tool_arguments": None,
        }
        state = create_state("Find one paper")

        answer = run_agent(state)

        decide.assert_called_once_with(state, allow_planning=True)
        self.assertEqual(answer, "Please provide another title.")
        self.assertIsNone(state["task_id"])
        self.assertIsNone(state["plan"])

    @patch("runtime.decide_next_action")
    def test_invalid_plan_action_does_not_partially_update_state(self, decide):
        decide.return_value = {
            **PLAN_ACTION,
            "task_id": "model-controlled-task",
        }
        state = create_state("Compare two papers")
        events = []

        with self.assertRaisesRegex(ValueError, "plan action contract"):
            run_agent(state, on_event=events.append)

        self.assertIsNone(state["task_id"])
        self.assertIsNone(state["plan"])
        self.assertEqual(events[-1]["kind"], "run_failed")


if __name__ == "__main__":
    unittest.main()
