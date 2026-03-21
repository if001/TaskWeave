from __future__ import annotations

import json
from pathlib import Path

from typing import Literal, Protocol, TypedDict

from ..infra import TaskNotFoundError, get_logger
from ..types import JsonValue, Task, TaskStatus

_ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    "queued": {"leased", "cancelled"},
    "leased": {"running", "queued", "failed", "cancelled"},
    "running": {"succeeded", "failed", "queued", "cancelled"},
    "succeeded": set(),
    "failed": set(),
    "cancelled": set(),
}

DedupePolicy = Literal["raise", "drop"]

logger = get_logger("taskweave.runtime_core.repository")


class TransitionPolicy(Protocol):
    def validate(self, from_status: TaskStatus, to_status: TaskStatus) -> None: ...


class DefaultTransitionPolicy:
    def validate(self, from_status: TaskStatus, to_status: TaskStatus) -> None:
        if to_status not in _ALLOWED_TRANSITIONS[from_status]:
            raise ValueError(f"Invalid task transition: {from_status} -> {to_status}")


class _TaskSerialized(TypedDict):
    id: str
    kind: str
    payload: dict[str, JsonValue]
    status: TaskStatus
    run_after: float | None
    parent_task_id: str | None
    dedupe_key: str | None
    metadata: dict[str, JsonValue]


class _RepositoryState(TypedDict):
    tasks: dict[str, _TaskSerialized]
    order: list[str]
    attempts: dict[str, int]
    task_id_by_dedupe_key: dict[str, str]


class TaskRepository(Protocol):
    def enqueue(self, task: Task) -> None: ...

    def enqueue_many(self, tasks: list[Task]) -> None: ...

    def lease_next_ready(self, now_unix: float) -> Task | None: ...

    def lease_next_ready_by_kinds(
        self, now_unix: float, kinds: list[str]
    ) -> Task | None: ...

    def mark_status(
        self, task_id: str, to_status: TaskStatus, reason: str | None = None
    ) -> None: ...

    def increment_attempt(self, task_id: str) -> int: ...

    def set_run_after(self, task_id: str, run_after: float | None) -> None: ...

    def get(self, task_id: str) -> Task | None: ...


class _TaskRepositoryBase(TaskRepository):
    def __init__(
        self,
        transition_policy: TransitionPolicy | None = None,
        dedupe_policy: DedupePolicy = "raise",
    ) -> None:
        self._tasks: dict[str, Task] = {}
        self._order: list[str] = []
        self._attempts: dict[str, int] = {}
        self._transition_policy = transition_policy or DefaultTransitionPolicy()
        self._dedupe_policy = dedupe_policy
        self._task_id_by_dedupe_key: dict[str, str] = {}

    def enqueue(self, task: Task) -> None:
        if not self._should_enqueue(task):
            return
        if task.id in self._tasks:
            logger.error(f"Task already exists: {task.id}")
            raise ValueError(f"Task already exists: {task.id}")
        self._tasks[task.id] = task
        self._order.append(task.id)
        self._attempts[task.id] = 0
        if task.dedupe_key:
            self._task_id_by_dedupe_key[task.dedupe_key] = task.id
        self._persist()
        logger.info("Task enqueued id=%s kind=%s", task.id, task.kind)

    def enqueue_many(self, tasks: list[Task]) -> None:
        for task in tasks:
            self.enqueue(task)

    def lease_next_ready(self, now_unix: float) -> Task | None:
        return self._lease_next_ready(now_unix, None)

    def lease_next_ready_by_kinds(
        self, now_unix: float, kinds: list[str]
    ) -> Task | None:
        if not kinds:
            return None
        return self._lease_next_ready(now_unix, set(kinds))

    def mark_status(
        self, task_id: str, to_status: TaskStatus, reason: str | None = None
    ) -> None:
        task = self._require_task(task_id)
        from_status = task.status
        self._transition_policy.validate(from_status, to_status)
        task.status = to_status
        self._persist()
        logger.info("Task status updated id=%s %s->%s", task_id, from_status, to_status)

    def increment_attempt(self, task_id: str) -> int:
        self._require_task(task_id)
        self._attempts[task_id] += 1
        self._persist()
        logger.info(
            "Task attempt incremented id=%s attempt=%s",
            task_id,
            self._attempts[task_id],
        )
        return self._attempts[task_id]

    def set_run_after(self, task_id: str, run_after: float | None) -> None:
        self._require_task(task_id).run_after = run_after
        self._persist()
        logger.debug("Task run_after set id=%s run_after=%s", task_id, run_after)

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def _require_task(self, task_id: str) -> Task:
        task = self._tasks.get(task_id)
        if task is None:
            raise TaskNotFoundError(f"Task not found: {task_id}")
        return task

    def _is_ready(self, task: Task, now_unix: float) -> bool:
        if task.status != "queued":
            return False
        if task.run_after is None:
            return True
        return task.run_after <= now_unix

    def _should_enqueue(self, task: Task) -> bool:
        if not task.dedupe_key:
            return True

        if task.dedupe_key not in self._task_id_by_dedupe_key:
            return True
        if self._dedupe_policy == "raise":
            existing = self._task_id_by_dedupe_key[task.dedupe_key]
            logger.error(
                "Dedupe key collision key=%s task_id=%s", task.dedupe_key, existing
            )
            raise ValueError(
                f"Dedupe key already exists: {task.dedupe_key} (task_id={existing})"
            )

        logger.warning("Task dropped due to dedupe key=%s", task.dedupe_key)
        return False

    def _lease_next_ready(self, now_unix: float, kinds: set[str] | None) -> Task | None:
        for task_id in self._order:
            task = self._tasks[task_id]
            if kinds is not None and task.kind not in kinds:
                continue
            if not self._is_ready(task, now_unix):
                continue
            self.mark_status(task.id, "leased")
            logger.debug("Task leased id=%s", task.id)
            return task
        return None

    def _persist(self) -> None:
        """Hook point for repositories that need durable persistence."""


class InMemoryTaskRepository(_TaskRepositoryBase):
    pass


class FileTaskRepository(_TaskRepositoryBase):
    def __init__(
        self,
        file_path: str | Path,
        transition_policy: TransitionPolicy | None = None,
        dedupe_policy: DedupePolicy = "raise",
    ) -> None:
        self._file_path = Path(file_path)
        super().__init__(
            transition_policy=transition_policy, dedupe_policy=dedupe_policy
        )
        self._load()
        self._persist()
        logger.info("FileTaskRepository initialized path=%s", self._file_path)

    def _persist(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(
            json.dumps(
                self._serialize_state(), ensure_ascii=False, indent=2, sort_keys=True
            ),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if not self._file_path.exists():
            logger.warning(
                "Repository file does not exist yet path=%s", self._file_path
            )
            return

        state = _parse_state(
            json.loads(self._file_path.read_text(encoding="utf-8"))
        )
        self._tasks = {
            task_id: Task(
                id=task_data["id"],
                kind=task_data["kind"],
                payload=dict(task_data["payload"]),
                status=task_data["status"],
                run_after=task_data["run_after"],
                parent_task_id=task_data["parent_task_id"],
                dedupe_key=task_data["dedupe_key"],
                metadata=dict(task_data["metadata"]),
            )
            for task_id, task_data in state["tasks"].items()
        }
        self._order = list(state["order"])
        self._attempts = dict(state["attempts"])
        self._task_id_by_dedupe_key = dict(state["task_id_by_dedupe_key"])
        logger.debug(
            "Repository state loaded tasks=%s",
            len(self._tasks),
        )

    def _serialize_state(self) -> _RepositoryState:
        return {
            "tasks": {
                task_id: _TaskSerialized(
                    id=task.id,
                    kind=task.kind,
                    payload=dict(task.payload),
                    status=task.status,
                    run_after=task.run_after,
                    parent_task_id=task.parent_task_id,
                    dedupe_key=task.dedupe_key,
                    metadata=dict(task.metadata),
                )
                for task_id, task in self._tasks.items()
            },
            "order": list(self._order),
            "attempts": dict(self._attempts),
            "task_id_by_dedupe_key": dict(self._task_id_by_dedupe_key),
        }


def _parse_state(value: JsonValue) -> _RepositoryState:
    if not isinstance(value, dict):
        raise ValueError("Invalid repository state")
    tasks_raw = value.get("tasks")
    order_raw = value.get("order")
    attempts_raw = value.get("attempts")
    dedupe_raw = value.get("task_id_by_dedupe_key")
    if not isinstance(tasks_raw, dict):
        raise ValueError("Invalid tasks in repository state")
    if not isinstance(order_raw, list):
        raise ValueError("Invalid order in repository state")
    if not isinstance(attempts_raw, dict):
        raise ValueError("Invalid attempts in repository state")
    if not isinstance(dedupe_raw, dict):
        raise ValueError("Invalid dedupe index in repository state")

    tasks: dict[str, _TaskSerialized] = {}
    for task_id, task_data in tasks_raw.items():
        if not isinstance(task_data, dict):
            raise ValueError("Invalid task entry")
        tasks[task_id] = _parse_task(task_data)

    order = [item for item in order_raw if isinstance(item, str)]
    attempts = {
        key: int(value)
        for key, value in attempts_raw.items()
        if isinstance(value, (int, float, bool))
    }
    task_id_by_dedupe_key = {
        key: value
        for key, value in dedupe_raw.items()
        if isinstance(value, str)
    }
    return _RepositoryState(
        tasks=tasks,
        order=order,
        attempts=attempts,
        task_id_by_dedupe_key=task_id_by_dedupe_key,
    )


def _parse_task(value: dict[str, JsonValue]) -> _TaskSerialized:
    task_id = _require_str(value.get("id"))
    kind = _require_str(value.get("kind"))
    status = _parse_task_status(value.get("status"))
    payload = _require_dict(value.get("payload"))
    metadata = _require_dict(value.get("metadata"))
    run_after = _parse_optional_float(value.get("run_after"))
    parent_task_id = _parse_optional_str(value.get("parent_task_id"))
    dedupe_key = _parse_optional_str(value.get("dedupe_key"))
    return _TaskSerialized(
        id=task_id,
        kind=kind,
        payload=payload,
        status=status,
        run_after=run_after,
        parent_task_id=parent_task_id,
        dedupe_key=dedupe_key,
        metadata=metadata,
    )


def _require_str(value: JsonValue | None) -> str:
    if isinstance(value, str):
        return value
    raise ValueError("Expected string")


def _parse_optional_str(value: JsonValue | None) -> str | None:
    if value is None:
        return None
    return _require_str(value)


def _require_float(value: JsonValue | None) -> float:
    if isinstance(value, (int, float, bool)):
        return float(value)
    raise ValueError("Expected number")


def _parse_optional_float(value: JsonValue | None) -> float | None:
    if value is None:
        return None
    return _require_float(value)


def _require_dict(value: JsonValue | None) -> dict[str, JsonValue]:
    if isinstance(value, dict):
        return dict(value)
    raise ValueError("Expected dict")


def _parse_task_status(value: JsonValue | None) -> TaskStatus:
    if value in _ALLOWED_TRANSITIONS:
        return value
    raise ValueError("Invalid task status")
