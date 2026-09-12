import json
import os
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    MAX_CHECKPOINT_SIZE_BYTES,
    CheckpointConflictError,
    CheckpointError,
    CheckpointValidationError,
    checkpoint_exists,
    checkpoint_path,
    clear_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_state,
)
from planning import (
    complete_current_step,
    create_plan,
    record_current_step_evidence,
    record_plan_usage,
    start_next_step,
)
from state import create_state


PAPER = {
    "candidate_id": "arxiv:1706.03762",
    "source": "arxiv",
    "arxiv_id": "1706.03762v7",
    "doi": None,
    "title": "Attention Is All You Need",
    "authors": ["Ashish Vaswani"],
    "abstract": "Transformer paper.",
    "published": "2017-06-12T17:57:34Z",
    "paper_url": "https://arxiv.org/abs/1706.03762v7",
    "pdf_url": "https://arxiv.org/pdf/1706.03762v7",
}


def active_state(task_id="task-checkpoint"):
    state = create_state("Compare two papers")
    state["task_id"] = task_id
    state["plan"] = create_plan(
        task_id,
        "Compare two papers",
        ["Find the papers", "Compare their evidence"],
    )
    return state


def state_with_search_result():
    state = active_state()
    state["plan"] = start_next_step(state["plan"])
    state["plan"] = record_plan_usage(state["plan"], tool_calls=1)
    state["plan"] = record_current_step_evidence(
        state["plan"],
        ["tool-result-001"],
    )
    state["messages"].extend(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_call": {
                    "id": "search-call-1",
                    "name": "search_paper",
                    "arguments": {"query": "Attention Is All You Need"},
                },
                "reasoning_content": "opaque provider protocol data",
            },
            {
                "role": "tool",
                "tool_call_id": "search-call-1",
                "name": "search_paper",
                "content": {
                    "found": True,
                    "count": 1,
                    "papers": [deepcopy(PAPER)],
                },
                "evidence_ref": "tool-result-001",
            },
        ]
    )
    state["step"] = 1
    return state


class CheckpointTests(unittest.TestCase):
    def test_round_trip_preserves_active_state_and_reasoning(self):
        state = state_with_search_result()
        snapshot = deepcopy(state)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoints" / "active.json"

            metadata = save_checkpoint(state, path)
            loaded = load_checkpoint(path)

            self.assertTrue(checkpoint_exists(path))
            self.assertEqual(loaded, snapshot)
            self.assertEqual(
                loaded["messages"][1]["reasoning_content"],
                "opaque provider protocol data",
            )
            self.assertEqual(metadata["path"], str(path.resolve()))
            self.assertLessEqual(
                metadata["size_bytes"],
                MAX_CHECKPOINT_SIZE_BYTES,
            )
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["version"], CHECKPOINT_FORMAT_VERSION)
            self.assertIn("saved_at", record)

        self.assertEqual(state, snapshot)

    def test_configured_path_and_idempotent_clear(self):
        with TemporaryDirectory() as directory:
            configured = Path(directory) / "custom.json"
            with patch.dict(
                os.environ,
                {"AGENT_READER_CHECKPOINT_PATH": str(configured)},
                clear=False,
            ):
                self.assertEqual(checkpoint_path(), configured.resolve())
                save_checkpoint(active_state())
                clear_checkpoint()
                clear_checkpoint()
                self.assertFalse(checkpoint_exists())

    def test_different_active_task_cannot_be_overwritten(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            save_checkpoint(active_state("task-one"), path)
            before = path.read_bytes()

            with self.assertRaisesRegex(
                CheckpointConflictError,
                "different active task",
            ):
                save_checkpoint(active_state("task-two"), path)

            self.assertEqual(path.read_bytes(), before)

    def test_atomic_write_failure_preserves_previous_checkpoint(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            state = active_state()
            save_checkpoint(state, path)
            before = path.read_bytes()
            state["messages"].append(
                {"role": "assistant", "content": "new progress"}
            )

            with (
                patch(
                    "checkpoint.os.replace",
                    side_effect=OSError("simulated interruption"),
                ),
                self.assertRaisesRegex(
                    CheckpointError,
                    "simulated interruption",
                ),
            ):
                save_checkpoint(state, path)

            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob(".active.json.*.tmp")), [])

    def test_corrupt_incompatible_and_oversized_files_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(
                CheckpointValidationError,
                "valid UTF-8 JSON",
            ):
                load_checkpoint(path)

            record = {
                "version": CHECKPOINT_FORMAT_VERSION + 1,
                "saved_at": "2026-09-13T00:00:00+00:00",
                "state": active_state(),
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(
                CheckpointValidationError,
                "Unsupported checkpoint version",
            ):
                load_checkpoint(path)

            record["version"] = True
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(
                CheckpointValidationError,
                "Unsupported checkpoint version",
            ):
                load_checkpoint(path)

            path.write_bytes(b"x" * (MAX_CHECKPOINT_SIZE_BYTES + 1))
            with self.assertRaisesRegex(
                CheckpointValidationError,
                "4 MiB",
            ):
                load_checkpoint(path)

    def test_save_rejects_oversized_or_non_json_state(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            oversized = active_state()
            oversized["messages"].append(
                {
                    "role": "assistant",
                    "content": "x" * MAX_CHECKPOINT_SIZE_BYTES,
                }
            )
            with self.assertRaisesRegex(
                CheckpointValidationError,
                "4 MiB",
            ):
                save_checkpoint(oversized, path)

            non_json = active_state()
            non_json["messages"].append(
                {
                    "role": "assistant",
                    "content": "opaque response",
                    "reasoning_details": [b"x"],
                }
            )
            with self.assertRaisesRegex(
                CheckpointValidationError,
                "valid JSON data",
            ):
                save_checkpoint(non_json, path)

            self.assertFalse(path.exists())

    def test_only_active_matching_plan_state_is_accepted(self):
        state = active_state()
        state["task_id"] = "different-task"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            "does not match",
        ):
            validate_checkpoint_state(state)

        terminal = active_state()
        terminal["plan"] = complete_current_step(
            start_next_step(terminal["plan"])
        )
        terminal["plan"] = complete_current_step(
            start_next_step(terminal["plan"])
        )
        with self.assertRaisesRegex(
            CheckpointValidationError,
            "running or blocked",
        ):
            validate_checkpoint_state(terminal)

        empty_assistant = active_state()
        empty_assistant["messages"].append(
            {"role": "assistant", "content": None}
        )
        with self.assertRaisesRegex(
            CheckpointValidationError,
            "content or a Tool Call",
        ):
            validate_checkpoint_state(empty_assistant)

    def test_tampered_stable_id_or_evidence_reference_is_rejected(self):
        bad_id = state_with_search_result()
        bad_id["messages"][2]["content"]["papers"][0][
            "candidate_id"
        ] = "arxiv:9999.99999"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            "stable candidate_id",
        ):
            validate_checkpoint_state(bad_id)

        bad_evidence = state_with_search_result()
        bad_evidence["messages"][2]["evidence_ref"] = "tool-result-999"
        with self.assertRaisesRegex(
            CheckpointValidationError,
            "evidence references",
        ):
            validate_checkpoint_state(bad_evidence)

    def test_tampered_library_id_is_rejected(self):
        state = active_state()
        state["plan"] = start_next_step(state["plan"])
        state["plan"] = record_plan_usage(
            state["plan"],
            tool_calls=1,
        )
        state["plan"] = record_current_step_evidence(
            state["plan"],
            ["tool-result-001"],
        )
        state["messages"].extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_call": {
                        "id": "list-call-1",
                        "name": "list_library",
                        "arguments": {"limit": 20},
                    },
                },
                {
                    "role": "tool",
                    "tool_call_id": "list-call-1",
                    "name": "list_library",
                    "content": {
                        "count": 1,
                        "total": 1,
                        "truncated": False,
                        "papers": [
                            {
                                "id": "arxiv:1706.03762",
                                "title": "Attention Is All You Need",
                                "authors": [],
                                "source": "arxiv",
                                "published": None,
                                "paper_url": (
                                    "https://arxiv.org/abs/1706.03762v7"
                                ),
                                "pdf_url": (
                                    "https://arxiv.org/pdf/1706.03762v7"
                                ),
                                "saved_at": "2026-01-01T00:00:00+00:00",
                            }
                        ],
                    },
                    "evidence_ref": "tool-result-001",
                },
            ]
        )

        validate_checkpoint_state(state)
        state["messages"][-1]["content"]["papers"][0][
            "id"
        ] = "arxiv:9999.99999"

        with self.assertRaisesRegex(
            CheckpointValidationError,
            "stable paper ID",
        ):
            validate_checkpoint_state(state)

    def test_restore_revalidates_downloaded_pdf_cache(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "pdfs"
            cache_dir.mkdir()
            pdf_path = cache_dir / "paper.pdf"
            pdf_path.write_bytes(b"%PDF-1.7\ncheckpoint test")
            checkpoint_file = Path(directory) / "active.json"
            state = state_with_search_result()
            state["plan"] = record_plan_usage(
                state["plan"],
                tool_calls=1,
            )
            state["plan"] = record_current_step_evidence(
                state["plan"],
                ["tool-result-002"],
            )
            state["messages"].extend(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_call": {
                            "id": "download-call-1",
                            "name": "download_paper",
                            "arguments": {"paper_id": PAPER["candidate_id"]},
                        },
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "download-call-1",
                        "name": "download_paper",
                        "content": {
                            "downloaded": True,
                            "cached": False,
                            "paper_id": PAPER["candidate_id"],
                            "local_path": str(pdf_path),
                            "size_bytes": pdf_path.stat().st_size,
                        },
                        "evidence_ref": "tool-result-002",
                    },
                ]
            )

            with patch.dict(
                os.environ,
                {"PAPER_CACHE_DIR": str(cache_dir)},
                clear=False,
            ):
                save_checkpoint(state, checkpoint_file)
                self.assertEqual(
                    load_checkpoint(checkpoint_file),
                    state,
                )
                pdf_path.unlink()
                with self.assertRaisesRegex(
                    CheckpointValidationError,
                    "cache is no longer valid",
                ):
                    load_checkpoint(checkpoint_file)


if __name__ == "__main__":
    unittest.main()
