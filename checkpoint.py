"""Versioned, bounded, atomic persistence for one active V3 task."""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

from planning import validate_plan
from tools.download import validate_cached_pdf_path
from tools.library import paper_id


CHECKPOINT_FORMAT_VERSION = 1
MAX_CHECKPOINT_SIZE_BYTES = 4 * 1024 * 1024
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent / "data/checkpoints/active.json"
)
CHECKPOINT_PATH_ENV = "AGENT_READER_CHECKPOINT_PATH"

_CHECKPOINT_KEYS = frozenset({"version", "saved_at", "state"})
_STATE_KEYS = frozenset(
    {"user_query", "messages", "step", "task_id", "plan"}
)
_ASSISTANT_MESSAGE_KEYS = frozenset(
    {
        "role",
        "content",
        "tool_call",
        "reasoning_content",
        "reasoning_details",
        "runtime_control",
    }
)
_TOOL_MESSAGE_KEYS = frozenset(
    {"role", "tool_call_id", "name", "content", "evidence_ref"}
)
_PAPER_ID_PATTERN = re.compile(r"^(?:arxiv|doi|paper):\S+$")


class CheckpointError(RuntimeError):
    """Base error for checkpoint validation and persistence failures."""


class CheckpointValidationError(CheckpointError):
    """Raised when checkpoint data does not satisfy the versioned contract."""


class CheckpointConflictError(CheckpointError):
    """Raised when a different active task already owns the checkpoint."""


def checkpoint_path(value: str | Path | None = None) -> Path:
    """Resolve an explicit, configured, or default checkpoint path."""
    selected = value
    if selected is None:
        selected = os.getenv(CHECKPOINT_PATH_ENV) or DEFAULT_CHECKPOINT_PATH
    return Path(selected).expanduser().resolve()


def checkpoint_exists(path: str | Path | None = None) -> bool:
    """Return whether an active checkpoint path currently exists."""
    return checkpoint_path(path).exists()


def save_checkpoint(
    state: dict,
    path: str | Path | None = None,
) -> dict:
    """Validate and atomically persist one active task State."""
    validate_checkpoint_state(state)
    target = checkpoint_path(path)

    if target.exists():
        existing = _load_record(target)
        existing_task_id = existing["state"]["task_id"]
        if existing_task_id != state["task_id"]:
            raise CheckpointConflictError(
                "A different active task checkpoint already exists: "
                f"{existing_task_id}. Resume or cancel it before starting "
                "another task."
            )

    record = {
        "version": CHECKPOINT_FORMAT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state": deepcopy(state),
    }
    serialized = _serialize_record(record)

    temporary_path = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    except OSError as error:
        raise CheckpointError(
            f"Could not write checkpoint '{target}': {error}"
        ) from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    return {"path": str(target), "size_bytes": len(serialized)}


def load_checkpoint(path: str | Path | None = None) -> dict:
    """Load and revalidate one active task State."""
    target = checkpoint_path(path)
    return deepcopy(_load_record(target)["state"])


def clear_checkpoint(path: str | Path | None = None) -> None:
    """Remove the active checkpoint, doing nothing when it is absent."""
    target = checkpoint_path(path)
    try:
        target.unlink(missing_ok=True)
    except OSError as error:
        raise CheckpointError(
            f"Could not clear checkpoint '{target}': {error}"
        ) from error


def validate_checkpoint_state(state: object) -> None:
    """Validate State, Plan, messages, trusted IDs, evidence, and caches."""
    if not isinstance(state, dict):
        raise CheckpointValidationError("Checkpoint state must be a dictionary.")
    _require_exact_keys(state, _STATE_KEYS, "checkpoint state")

    user_query = state["user_query"]
    if not isinstance(user_query, str) or not user_query.strip():
        raise CheckpointValidationError(
            "Checkpoint user_query must be non-empty text."
        )
    step = state["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise CheckpointValidationError(
            "Checkpoint step must be a non-negative integer."
        )

    task_id = state["task_id"]
    plan = state["plan"]
    if not isinstance(task_id, str) or not task_id:
        raise CheckpointValidationError(
            "Checkpoint task_id must identify one active task."
        )
    try:
        validate_plan(plan)
    except (TypeError, ValueError) as error:
        raise CheckpointValidationError(
            f"Checkpoint Plan is invalid: {error}"
        ) from error
    if plan["task_id"] != task_id:
        raise CheckpointValidationError(
            "Checkpoint task_id does not match its Plan."
        )
    if plan["status"] not in {"running", "blocked"}:
        raise CheckpointValidationError(
            "Checkpoint must contain a running or blocked task."
        )

    messages = state["messages"]
    if not isinstance(messages, list) or not messages:
        raise CheckpointValidationError(
            "Checkpoint messages must be a non-empty list."
        )
    _validate_message_history(messages, plan)

    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CheckpointValidationError(
            f"Checkpoint State is not valid JSON data: {error}"
        ) from error


def _load_record(path: Path) -> dict:
    try:
        size_bytes = path.stat().st_size
    except FileNotFoundError as error:
        raise CheckpointError(
            f"Checkpoint does not exist: '{path}'."
        ) from error
    except OSError as error:
        raise CheckpointError(
            f"Could not inspect checkpoint '{path}': {error}"
        ) from error

    if size_bytes > MAX_CHECKPOINT_SIZE_BYTES:
        raise CheckpointValidationError(
            "Checkpoint exceeds the 4 MiB size limit."
        )
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CheckpointError(
            f"Could not read checkpoint '{path}': {error}"
        ) from error
    if len(raw) > MAX_CHECKPOINT_SIZE_BYTES:
        raise CheckpointValidationError(
            "Checkpoint exceeds the 4 MiB size limit."
        )
    try:
        record = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointValidationError(
            f"Checkpoint is not valid UTF-8 JSON: {error}"
        ) from error

    _validate_record(record)
    return record


def _validate_record(record: object) -> None:
    if not isinstance(record, dict):
        raise CheckpointValidationError(
            "Checkpoint root must be a dictionary."
        )
    _require_exact_keys(record, _CHECKPOINT_KEYS, "checkpoint")
    version = record["version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != CHECKPOINT_FORMAT_VERSION
    ):
        raise CheckpointValidationError(
            "Unsupported checkpoint version: "
            f"{version!r}."
        )
    saved_at = record["saved_at"]
    if not isinstance(saved_at, str):
        raise CheckpointValidationError(
            "Checkpoint saved_at must be text."
        )
    try:
        parsed_saved_at = datetime.fromisoformat(saved_at)
    except ValueError as error:
        raise CheckpointValidationError(
            "Checkpoint saved_at is not a valid ISO timestamp."
        ) from error
    if parsed_saved_at.tzinfo is None:
        raise CheckpointValidationError(
            "Checkpoint saved_at must include a timezone."
        )
    validate_checkpoint_state(record["state"])


def _serialize_record(record: dict) -> bytes:
    try:
        serialized = (
            json.dumps(
                record,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CheckpointValidationError(
            f"Checkpoint is not valid JSON data: {error}"
        ) from error
    if len(serialized) > MAX_CHECKPOINT_SIZE_BYTES:
        raise CheckpointValidationError(
            "Checkpoint exceeds the 4 MiB size limit."
        )
    return serialized


def _validate_message_history(messages: list, plan: dict) -> None:
    calls = {}
    completed_call_ids = set()
    evidence_refs = set()
    trusted_paper_ids = set()
    downloaded_paper_ids = set()
    has_user_message = False

    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise CheckpointValidationError(
                f"Checkpoint message {index} must be a dictionary."
            )
        role = message.get("role")
        if role == "user":
            _require_exact_keys(
                message,
                frozenset({"role", "content"}),
                f"user message {index}",
            )
            if not isinstance(message["content"], str):
                raise CheckpointValidationError(
                    f"User message {index} content must be text."
                )
            has_user_message = True
            continue
        if role == "assistant":
            _validate_assistant_message(message, index, calls)
            continue
        if role == "tool":
            _validate_tool_message(
                message,
                index,
                calls,
                completed_call_ids,
                evidence_refs,
                trusted_paper_ids,
                downloaded_paper_ids,
            )
            continue
        raise CheckpointValidationError(
            f"Checkpoint message {index} has invalid role {role!r}."
        )

    if not has_user_message:
        raise CheckpointValidationError(
            "Checkpoint messages must contain a user message."
        )
    unmatched_calls = set(calls) - completed_call_ids
    if unmatched_calls:
        raise CheckpointValidationError(
            "Checkpoint contains Tool Calls without Tool Results: "
            f"{sorted(unmatched_calls)}."
        )

    plan_evidence_refs = {
        reference
        for step in plan["steps"]
        for reference in step["evidence_refs"]
    }
    if plan_evidence_refs != evidence_refs:
        raise CheckpointValidationError(
            "Checkpoint Plan evidence references do not match Tool Results."
        )


def _validate_assistant_message(
    message: dict,
    index: int,
    calls: dict,
) -> None:
    if not set(message) <= _ASSISTANT_MESSAGE_KEYS:
        extra = sorted(set(message) - _ASSISTANT_MESSAGE_KEYS)
        raise CheckpointValidationError(
            f"Assistant message {index} has unsupported fields: {extra}."
        )
    if "content" not in message:
        raise CheckpointValidationError(
            f"Assistant message {index} is missing content."
        )
    content = message["content"]
    if content is not None and not isinstance(content, str):
        raise CheckpointValidationError(
            f"Assistant message {index} content must be text or null."
        )

    reasoning_fields = {
        field
        for field in ("reasoning_content", "reasoning_details")
        if field in message
    }
    if len(reasoning_fields) > 1:
        raise CheckpointValidationError(
            f"Assistant message {index} has two reasoning representations."
        )
    if "reasoning_content" in message and not isinstance(
        message["reasoning_content"], str
    ):
        raise CheckpointValidationError(
            f"Assistant message {index} reasoning_content must be text."
        )
    if "reasoning_details" in message and not isinstance(
        message["reasoning_details"], list
    ):
        raise CheckpointValidationError(
            f"Assistant message {index} reasoning_details must be a list."
        )
    if "runtime_control" in message and message["runtime_control"] != (
        "max_steps_reached"
    ):
        raise CheckpointValidationError(
            f"Assistant message {index} has invalid Runtime control."
        )

    tool_call = message.get("tool_call")
    if tool_call is None:
        if content is None:
            raise CheckpointValidationError(
                f"Assistant message {index} requires content or a Tool Call."
            )
        return
    if not isinstance(tool_call, dict):
        raise CheckpointValidationError(
            f"Assistant message {index} Tool Call must be a dictionary."
        )
    _require_exact_keys(
        tool_call,
        frozenset({"id", "name", "arguments"}),
        f"Tool Call in message {index}",
    )
    call_id = tool_call["id"]
    name = tool_call["name"]
    arguments = tool_call["arguments"]
    if not isinstance(call_id, str) or not call_id:
        raise CheckpointValidationError(
            f"Tool Call in message {index} requires a non-empty ID."
        )
    if call_id in calls:
        raise CheckpointValidationError(
            f"Checkpoint contains duplicate Tool Call ID {call_id!r}."
        )
    if not isinstance(name, str) or not name:
        raise CheckpointValidationError(
            f"Tool Call in message {index} requires a non-empty name."
        )
    if not isinstance(arguments, dict):
        raise CheckpointValidationError(
            f"Tool Call in message {index} arguments must be a dictionary."
        )
    calls[call_id] = {"name": name, "arguments": arguments}


def _validate_tool_message(
    message: dict,
    index: int,
    calls: dict,
    completed_call_ids: set,
    evidence_refs: set,
    trusted_paper_ids: set,
    downloaded_paper_ids: set,
) -> None:
    if not set(message) <= _TOOL_MESSAGE_KEYS:
        extra = sorted(set(message) - _TOOL_MESSAGE_KEYS)
        raise CheckpointValidationError(
            f"Tool message {index} has unsupported fields: {extra}."
        )
    required = {"role", "tool_call_id", "name", "content"}
    if not required <= set(message):
        missing = sorted(required - set(message))
        raise CheckpointValidationError(
            f"Tool message {index} is missing fields: {missing}."
        )
    call_id = message["tool_call_id"]
    name = message["name"]
    if call_id not in calls:
        raise CheckpointValidationError(
            f"Tool message {index} has no matching Assistant Tool Call."
        )
    if call_id in completed_call_ids:
        raise CheckpointValidationError(
            f"Tool Call {call_id!r} has more than one Tool Result."
        )
    if name != calls[call_id]["name"]:
        raise CheckpointValidationError(
            f"Tool message {index} name does not match its Tool Call."
        )
    result = message["content"]
    if not isinstance(result, dict):
        raise CheckpointValidationError(
            f"Tool message {index} content must be a dictionary."
        )
    completed_call_ids.add(call_id)

    evidence_ref = message.get("evidence_ref")
    if evidence_ref is not None:
        if not isinstance(evidence_ref, str) or not evidence_ref:
            raise CheckpointValidationError(
                f"Tool message {index} evidence_ref must be non-empty text."
            )
        if evidence_ref in evidence_refs:
            raise CheckpointValidationError(
                f"Duplicate Tool evidence reference {evidence_ref!r}."
            )
        evidence_refs.add(evidence_ref)

    if "error" in result:
        return
    if name == "search_paper":
        _record_search_ids(result, trusted_paper_ids)
    elif name == "list_library":
        _record_library_ids(result, trusted_paper_ids)
    elif name == "download_paper":
        _validate_download_result(
            result,
            trusted_paper_ids,
            downloaded_paper_ids,
        )
    elif name == "save_paper":
        _validate_save_result(result, trusted_paper_ids)
    elif name in {"extract_paper_text", "retrieve_paper_chunks"}:
        _validate_read_result(result, downloaded_paper_ids, name)


def _record_search_ids(result: dict, trusted_paper_ids: set) -> None:
    papers = result.get("papers", [])
    if not isinstance(papers, list):
        raise CheckpointValidationError(
            "search_paper result papers must be a list."
        )
    for paper in papers:
        if not isinstance(paper, dict):
            raise CheckpointValidationError(
                "search_paper returned an invalid paper record."
            )
        candidate_id = paper.get("candidate_id")
        if not _is_paper_id(candidate_id) or paper_id(paper) != candidate_id:
            raise CheckpointValidationError(
                "search_paper returned an invalid stable candidate_id."
            )
        trusted_paper_ids.add(candidate_id)


def _record_library_ids(result: dict, trusted_paper_ids: set) -> None:
    papers = result.get("papers", [])
    if not isinstance(papers, list):
        raise CheckpointValidationError(
            "list_library result papers must be a list."
        )
    for paper in papers:
        if not isinstance(paper, dict) or not _library_id_matches(paper):
            raise CheckpointValidationError(
                "list_library returned an invalid stable paper ID."
            )
        trusted_paper_ids.add(paper["id"])


def _library_id_matches(paper: dict) -> bool:
    stable_id = paper.get("id")
    if not _is_paper_id(stable_id):
        return False
    if stable_id.startswith("paper:"):
        return paper_id(paper) == stable_id

    identity = stable_id.split(":", maxsplit=1)[1].casefold()
    urls = " ".join(
        str(paper.get(field) or "").casefold()
        for field in ("paper_url", "pdf_url")
    )
    return identity in urls


def _validate_download_result(
    result: dict,
    trusted_paper_ids: set,
    downloaded_paper_ids: set,
) -> None:
    if result.get("downloaded") is not True and result.get("cached") is not True:
        return
    resolved_paper_id = result.get("paper_id")
    if resolved_paper_id not in trusted_paper_ids:
        raise CheckpointValidationError(
            "download_paper result does not reference a trusted paper ID."
        )
    local_path = result.get("local_path")
    if not isinstance(local_path, str):
        raise CheckpointValidationError(
            "download_paper result requires a cached local_path."
        )
    try:
        validate_cached_pdf_path(local_path)
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointValidationError(
            f"download_paper cache is no longer valid: {error}"
        ) from error
    downloaded_paper_ids.add(resolved_paper_id)


def _validate_save_result(result: dict, trusted_paper_ids: set) -> None:
    completed = result.get("saved") is True or result.get("reason") == (
        "already_exists"
    )
    if not completed:
        return
    if result.get("paper_id") not in trusted_paper_ids:
        raise CheckpointValidationError(
            "save_paper result does not reference a trusted paper ID."
        )


def _validate_read_result(
    result: dict,
    downloaded_paper_ids: set,
    tool_name: str,
) -> None:
    resolved_paper_id = result.get("paper_id")
    if resolved_paper_id is None:
        return
    if resolved_paper_id not in downloaded_paper_ids:
        raise CheckpointValidationError(
            f"{tool_name} result does not reference a valid download."
        )


def _is_paper_id(value: object) -> bool:
    return isinstance(value, str) and _PAPER_ID_PATTERN.fullmatch(value) is not None


def _require_exact_keys(
    value: dict,
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise CheckpointValidationError(
        f"{label} fields do not match the contract; "
        f"missing={missing}, extra={extra}."
    )
