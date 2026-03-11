from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import time
from typing import Literal, Protocol, TypedDict, cast

from ..infra import TaskNotFoundError, get_logger
from ..types import Task, TaskStatus

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


@dataclass(slots=True)
class TaskTransition:
    task_id: str
    from_status: str
    to_status: str
    timestamp_unix: float
    reason: str | None = None


class _TaskSerialized(TypedDict):
    id: str
    kind: str
    payload: dict[str, object]
    status: str
    run_after: float | None
    parent_task_id: str | None
    dedupe_key: str | None
    metadata: dict[str, object]


class _TransitionSerialized(TypedDict):
    task_id: str
    from_status: str
    to_status: str
    timestamp_unix: float
    reason: str | None


class _RepositoryState(TypedDict):
    tasks: dict[str, _TaskSerialized]
    order: list[str]
    attempts: dict[str, int]
    task_id_by_dedupe_key: dict[str, str]
    transitions: list[_TransitionSerialized]


class TaskRepository(Protocol):
    def enqueue(self, task: Task) -> None: ...

    def enqueue_many(self, tasks: list[Task]) -> None: ...

    def lease_next_ready(self, now_unix: float) -> Task | None: ...

    def lease_next_ready_by_kinds(
        self, now_unix: float, kinds: list[str]
    ) -> Task | None: ...

    def mark_status(
        self, task_id: str, to_status: str, reason: str | None = None
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
        self.transitions: list[TaskTransition] = []

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
        self, task_id: str, to_status: str, reason: str | None = None
    ) -> None:
        task = self._require_task(task_id)
        from_status = task.status
        self._transition_policy.validate(from_status, cast(TaskStatus, to_status))
        task.status = cast(TaskStatus, to_status)
        self.transitions.append(
            TaskTransition(
                task_id=task_id,
                from_status=from_status,
                to_status=to_status,
                timestamp_unix=time(),
                reason=reason,
            )
        )
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

        state = cast(
            _RepositoryState, json.loads(self._file_path.read_text(encoding="utf-8"))
        )
        self._tasks = {
            task_id: Task(
                id=task_data["id"],
                kind=task_data["kind"],
                payload=dict(task_data["payload"]),
                status=cast(TaskStatus, task_data["status"]),
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
        self.transitions = [
            TaskTransition(
                task_id=transition["task_id"],
                from_status=transition["from_status"],
                to_status=transition["to_status"],
                timestamp_unix=transition["timestamp_unix"],
                reason=transition["reason"],
            )
            for transition in state["transitions"]
        ]
        logger.debug(
            "Repository state loaded tasks=%s transitions=%s",
            len(self._tasks),
            len(self.transitions),
        )

    def _serialize_state(self) -> _RepositoryState:
        return {
            "tasks": {
                task_id: cast(_TaskSerialized, asdict(task))
                for task_id, task in self._tasks.items()
            },
            "order": list(self._order),
            "attempts": dict(self._attempts),
            "task_id_by_dedupe_key": dict(self._task_id_by_dedupe_key),
            "transitions": [
                cast(_TransitionSerialized, asdict(transition))
                for transition in self.transitions
            ],
        }
