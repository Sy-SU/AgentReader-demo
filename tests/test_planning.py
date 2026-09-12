import unittest
from copy import deepcopy

from planning import (
    MAX_EVIDENCE_REFS_PER_STEP,
    MAX_PLAN_STEPS,
    MAX_TASK_LLM_STEPS,
    MAX_TASK_REPLANS,
    PlanValidationError,
    block_current_step,
    cancel_plan,
    complete_current_step,
    create_plan,
    fail_current_step,
    fail_plan,
    record_current_step_evidence,
    record_plan_usage,
    revise_plan,
    resume_blocked_step,
    start_next_step,
    validate_plan,
)


class PlanContractTests(unittest.TestCase):
    def test_create_plan_builds_runtime_owned_fields(self):
        descriptions = ["  Search for paper A  ", "Compare the evidence"]

        plan = create_plan("task-123", "  Compare two papers  ", descriptions)

        self.assertEqual(plan["version"], 1)
        self.assertEqual(plan["task_id"], "task-123")
        self.assertEqual(plan["goal"], "Compare two papers")
        self.assertEqual(plan["status"], "running")
        self.assertEqual(plan["revision"], 1)
        self.assertIsNone(plan["current_step_id"])
        self.assertEqual(
            [step["id"] for step in plan["steps"]],
            ["step-001", "step-002"],
        )
        self.assertEqual(
            [step["description"] for step in plan["steps"]],
            ["Search for paper A", "Compare the evidence"],
        )
        self.assertTrue(
            all(step["status"] == "pending" for step in plan["steps"])
        )
        self.assertEqual(
            plan["budget"],
            {"llm_steps": 0, "tool_calls": 0, "replans": 0},
        )
        self.assertEqual(
            descriptions,
            ["  Search for paper A  ", "Compare the evidence"],
        )

    def test_create_plan_rejects_invalid_boundaries(self):
        invalid_cases = [
            ("bad/task", "goal", ["step"]),
            ("task-1", "", ["step"]),
            ("task-1", "x" * 2_001, ["step"]),
            ("task-1", "goal", []),
            ("task-1", "goal", ["step"] * (MAX_PLAN_STEPS + 1)),
            ("task-1", "goal", [""]),
            ("task-1", "goal", ["x" * 501]),
        ]

        for task_id, goal, steps in invalid_cases:
            with self.subTest(task_id=task_id, goal_length=len(goal), steps=steps):
                with self.assertRaises(PlanValidationError):
                    create_plan(task_id, goal, steps)

    def test_validate_plan_rejects_missing_extra_and_duplicate_fields(self):
        plan = create_plan("task-1", "goal", ["one", "two"])

        missing = deepcopy(plan)
        missing.pop("revision")
        with self.assertRaisesRegex(PlanValidationError, "missing"):
            validate_plan(missing)

        extra = deepcopy(plan)
        extra["model_note"] = "untrusted"
        with self.assertRaisesRegex(PlanValidationError, "extra"):
            validate_plan(extra)

        duplicate = deepcopy(plan)
        duplicate["steps"][1]["id"] = duplicate["steps"][0]["id"]
        with self.assertRaisesRegex(PlanValidationError, "duplicate step id"):
            validate_plan(duplicate)

    def test_validate_plan_rejects_inconsistent_current_step(self):
        plan = start_next_step(create_plan("task-1", "goal", ["one", "two"]))

        missing_current = deepcopy(plan)
        missing_current["current_step_id"] = None
        with self.assertRaisesRegex(PlanValidationError, "active step"):
            validate_plan(missing_current)

        two_active = deepcopy(plan)
        two_active["steps"][1]["status"] = "running"
        two_active["steps"][1]["attempts"] = 1
        with self.assertRaisesRegex(PlanValidationError, "at most one"):
            validate_plan(two_active)

        unknown_current = deepcopy(plan)
        unknown_current["current_step_id"] = "step-999"
        with self.assertRaisesRegex(PlanValidationError, "existing step"):
            validate_plan(unknown_current)

    def test_validate_plan_rejects_status_and_counter_inconsistencies(self):
        plan = create_plan("task-1", "goal", ["one"])

        invalid_status = deepcopy(plan)
        invalid_status["status"] = "waiting"
        with self.assertRaisesRegex(PlanValidationError, "task status"):
            validate_plan(invalid_status)

        unhashable_status = deepcopy(plan)
        unhashable_status["steps"][0]["status"] = []
        with self.assertRaisesRegex(PlanValidationError, "invalid status"):
            validate_plan(unhashable_status)

        pending_attempt = deepcopy(plan)
        pending_attempt["steps"][0]["attempts"] = 1
        with self.assertRaisesRegex(PlanValidationError, "zero attempts"):
            validate_plan(pending_attempt)

        boolean_counter = deepcopy(plan)
        boolean_counter["budget"]["llm_steps"] = True
        with self.assertRaisesRegex(PlanValidationError, "llm_steps"):
            validate_plan(boolean_counter)

        boolean_revision = deepcopy(plan)
        boolean_revision["revision"] = True
        with self.assertRaisesRegex(PlanValidationError, "revision"):
            validate_plan(boolean_revision)

        boolean_version = deepcopy(plan)
        boolean_version["version"] = True
        with self.assertRaisesRegex(PlanValidationError, "version"):
            validate_plan(boolean_version)

        over_budget = deepcopy(plan)
        over_budget["budget"]["llm_steps"] = MAX_TASK_LLM_STEPS + 1
        with self.assertRaisesRegex(PlanValidationError, "llm_steps"):
            validate_plan(over_budget)

        fake_completed = deepcopy(plan)
        fake_completed["status"] = "completed"
        with self.assertRaisesRegex(PlanValidationError, "completed task"):
            validate_plan(fake_completed)

    def test_validate_plan_rejects_untrusted_evidence_references(self):
        plan = start_next_step(create_plan("task-1", "goal", ["one"]))

        unsafe = deepcopy(plan)
        unsafe["steps"][0]["evidence_refs"] = ["../../secret"]
        with self.assertRaisesRegex(PlanValidationError, "evidence reference"):
            validate_plan(unsafe)

        wrong_type = deepcopy(plan)
        wrong_type["steps"][0]["evidence_refs"] = [{"tool": 1}]
        with self.assertRaisesRegex(PlanValidationError, "evidence reference"):
            validate_plan(wrong_type)

        duplicate = deepcopy(plan)
        duplicate["steps"][0]["evidence_refs"] = ["tool:001", "tool:001"]
        with self.assertRaisesRegex(PlanValidationError, "unique"):
            validate_plan(duplicate)

        too_many = deepcopy(plan)
        too_many["steps"][0]["evidence_refs"] = [
            f"tool:{index}" for index in range(MAX_EVIDENCE_REFS_PER_STEP + 1)
        ]
        with self.assertRaisesRegex(PlanValidationError, "too many"):
            validate_plan(too_many)


class PlanTransitionTests(unittest.TestCase):
    def test_successful_lifecycle_is_sequential_and_immutable(self):
        created = create_plan("task-1", "goal", ["one", "two"])
        created_snapshot = deepcopy(created)

        first_running = start_next_step(created)
        self.assertEqual(created, created_snapshot)
        self.assertEqual(first_running["current_step_id"], "step-001")
        self.assertEqual(first_running["steps"][0]["attempts"], 1)

        first_done = complete_current_step(first_running, ["tool:001"])
        self.assertEqual(first_running["steps"][0]["status"], "running")
        self.assertEqual(first_done["steps"][0]["status"], "completed")
        self.assertEqual(first_done["steps"][0]["evidence_refs"], ["tool:001"])
        self.assertEqual(first_done["status"], "running")

        second_running = start_next_step(first_done)
        completed = complete_current_step(second_running, ["tool:002"])
        self.assertEqual(completed["status"], "completed")
        self.assertIsNone(completed["current_step_id"])
        self.assertTrue(
            all(step["status"] == "completed" for step in completed["steps"])
        )

    def test_only_one_step_can_run_at_a_time(self):
        running = start_next_step(create_plan("task-1", "goal", ["one", "two"]))

        with self.assertRaisesRegex(PlanValidationError, "active step"):
            start_next_step(running)
        with self.assertRaisesRegex(PlanValidationError, "no current step"):
            complete_current_step(create_plan("task-2", "goal", ["one"]))

    def test_blocked_step_can_resume_after_user_clarification(self):
        running = start_next_step(create_plan("task-1", "goal", ["choose paper"]))
        blocked = block_current_step(running, ["tool:ambiguous-search"])

        self.assertEqual(blocked["status"], "blocked")
        self.assertEqual(blocked["steps"][0]["status"], "blocked")
        self.assertEqual(blocked["current_step_id"], "step-001")

        resumed = resume_blocked_step(blocked)
        self.assertEqual(resumed["status"], "running")
        self.assertEqual(resumed["steps"][0]["status"], "running")
        self.assertEqual(resumed["steps"][0]["attempts"], 2)
        self.assertEqual(blocked["steps"][0]["attempts"], 1)

    def test_failed_step_requires_replanning_before_another_step(self):
        running = start_next_step(create_plan("task-1", "goal", ["one", "two"]))
        awaiting_replan = fail_current_step(running, ["tool:failure"])

        self.assertEqual(awaiting_replan["status"], "running")
        self.assertEqual(awaiting_replan["steps"][0]["status"], "failed")
        self.assertIsNone(awaiting_replan["current_step_id"])
        with self.assertRaisesRegex(PlanValidationError, "replanning"):
            start_next_step(awaiting_replan)

        terminal = fail_plan(awaiting_replan)
        self.assertEqual(terminal["status"], "failed")

    def test_replan_preserves_history_and_replaces_only_pending_steps(self):
        created = create_plan(
            "task-1",
            "goal",
            ["completed work", "failed work", "old pending", "old tail"],
        )
        first_running = start_next_step(created)
        first_done = complete_current_step(first_running, ["tool:001"])
        second_running = start_next_step(first_done)
        awaiting_replan = fail_current_step(second_running, ["tool:002"])
        snapshot = deepcopy(awaiting_replan)

        revised = revise_plan(
            awaiting_replan,
            ["alternate evidence", "finish comparison"],
        )

        self.assertEqual(awaiting_replan, snapshot)
        self.assertEqual(revised["revision"], 2)
        self.assertEqual(revised["budget"]["replans"], 1)
        self.assertEqual(
            [step["id"] for step in revised["steps"]],
            ["step-001", "step-002", "step-005", "step-006"],
        )
        self.assertEqual(
            [step["status"] for step in revised["steps"]],
            ["completed", "failed", "pending", "pending"],
        )
        self.assertEqual(
            [step["evidence_refs"] for step in revised["steps"][:2]],
            [["tool:001"], ["tool:002"]],
        )
        self.assertNotIn("old pending", str(revised))
        self.assertNotIn("old tail", str(revised))

        replacement_running = start_next_step(revised)
        replacement_done = complete_current_step(replacement_running)
        final_running = start_next_step(replacement_done)
        completed = complete_current_step(final_running)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(
            [step["status"] for step in completed["steps"]],
            ["completed", "failed", "completed", "completed"],
        )

    def test_replan_requires_failure_budget_and_available_capacity(self):
        created = create_plan("task-1", "goal", ["one", "two"])
        with self.assertRaisesRegex(PlanValidationError, "failed step"):
            revise_plan(created, ["replacement"])

        running = start_next_step(created)
        with self.assertRaisesRegex(PlanValidationError, "must fail"):
            revise_plan(running, ["replacement"])

        awaiting_replan = fail_current_step(running)
        once_revised = revise_plan(awaiting_replan, ["retry one"])
        once_failed = fail_current_step(start_next_step(once_revised))
        at_limit = revise_plan(once_failed, ["retry two"])
        at_limit = fail_current_step(start_next_step(at_limit))
        self.assertEqual(
            at_limit["budget"]["replans"],
            MAX_TASK_REPLANS,
        )
        with self.assertRaisesRegex(PlanValidationError, "replan limit"):
            revise_plan(at_limit, ["replacement"])

        full = create_plan(
            "task-full",
            "goal",
            [f"step {index}" for index in range(MAX_PLAN_STEPS)],
        )
        for _ in range(MAX_PLAN_STEPS - 1):
            full = complete_current_step(start_next_step(full))
        full = fail_current_step(start_next_step(full))
        with self.assertRaisesRegex(PlanValidationError, "capacity"):
            revise_plan(full, ["replacement"])

    def test_cancel_and_fail_close_an_active_step(self):
        running = start_next_step(create_plan("task-1", "goal", ["one"]))
        cancelled = cancel_plan(running)

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["steps"][0]["status"], "failed")
        self.assertIsNone(cancelled["current_step_id"])

        blocked = block_current_step(running)
        failed = fail_plan(blocked)
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["steps"][0]["status"], "failed")

    def test_terminal_plan_cannot_transition_again(self):
        running = start_next_step(create_plan("task-1", "goal", ["one"]))
        completed = complete_current_step(running)

        with self.assertRaisesRegex(PlanValidationError, "running or blocked"):
            cancel_plan(completed)
        with self.assertRaisesRegex(PlanValidationError, "running task"):
            start_next_step(completed)

    def test_transition_rejects_duplicate_new_evidence(self):
        running = start_next_step(create_plan("task-1", "goal", ["one"]))

        with self.assertRaisesRegex(PlanValidationError, "unique"):
            complete_current_step(running, ["tool:001", "tool:001"])

    def test_runtime_usage_is_bounded_and_does_not_mutate_old_plan(self):
        plan = create_plan("task-1", "goal", ["one"])

        updated = record_plan_usage(plan, llm_steps=1, tool_calls=2)

        self.assertEqual(
            plan["budget"],
            {"llm_steps": 0, "tool_calls": 0, "replans": 0},
        )
        self.assertEqual(
            updated["budget"],
            {"llm_steps": 1, "tool_calls": 2, "replans": 0},
        )
        with self.assertRaisesRegex(PlanValidationError, "llm_steps"):
            record_plan_usage(plan, llm_steps=MAX_TASK_LLM_STEPS + 1)
        with self.assertRaisesRegex(PlanValidationError, "non-negative"):
            record_plan_usage(plan, tool_calls=-1)
        with self.assertRaisesRegex(PlanValidationError, "non-negative"):
            record_plan_usage(plan, replans=True)

    def test_runtime_records_evidence_without_completing_the_step(self):
        running = start_next_step(create_plan("task-1", "goal", ["one"]))
        snapshot = deepcopy(running)

        updated = record_current_step_evidence(
            running,
            ["tool-result-001"],
        )

        self.assertEqual(running, snapshot)
        self.assertEqual(updated["steps"][0]["status"], "running")
        self.assertEqual(
            updated["steps"][0]["evidence_refs"],
            ["tool-result-001"],
        )
        with self.assertRaisesRegex(PlanValidationError, "unique"):
            record_current_step_evidence(updated, ["tool-result-001"])


if __name__ == "__main__":
    unittest.main()
