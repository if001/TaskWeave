from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Literal, Protocol

from runtime_core.errors import TaskNotFoundError
from runtime_core.models import Task, TaskStatus

_ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    "queued": {"leased", "cancelled"},
    "leased": {"running", "queued", "failed", "cancelled"},
    "running": {"succeeded", "failed", "queued", "cancelled"},
    "succeeded": set(),
    "failed": set(),
    "cancelled": set(),
}

DedupePolicy = Literal["raise", "drop"]


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


class TaskRepository(Protocol):
    def enqueue(self, task: Task) -> None: ...

    def enqueue_many(self, tasks: list[Task]) -> None: ...

    def lease_next_ready(self, now_unix: float) -> Task | None: ...

    def mark_status(self, task_id: str, to_status: str, reason: str | None = None) -> None: ...

    def increment_attempt(self, task_id: str) -> int: ...

    def set_run_after(self, task_id: str, run_after: float | None) -> None: ...

    def get(self, task_id: str) -> Task | None: ...


class InMemoryTaskRepository:
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
            raise ValueError(f"Task already exists: {task.id}")
        self._tasks[task.id] = task
        self._order.append(task.id)
        self._attempts[task.id] = 0
        if task.dedupe_key:
            self._task_id_by_dedupe_key[task.dedupe_key] = task.id

    def enqueue_many(self, tasks: list[Task]) -> None:
        for task in tasks:
            self.enqueue(task)

    def lease_next_ready(self, now_unix: float) -> Task | None:
        for task_id in self._order:
            task = self._tasks[task_id]
            if not self._is_ready(task, now_unix):
                continue
            self.mark_status(task.id, "leased")
            return task
        return None

    def mark_status(self, task_id: str, to_status: str, reason: str | None = None) -> None:
        task = self._require_task(task_id)
        from_status = task.status
        self._transition_policy.validate(from_status, to_status)
        task.status = to_status
        self.transitions.append(
            TaskTransition(
                task_id=task_id,
                from_status=from_status,
                to_status=to_status,
                timestamp_unix=time(),
                reason=reason,
            )
        )

    def increment_attempt(self, task_id: str) -> int:
        self._require_task(task_id)
        self._attempts[task_id] += 1
        return self._attempts[task_id]

    def set_run_after(self, task_id: str, run_after: float | None) -> None:
        self._require_task(task_id).run_after = run_after

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
            raise ValueError(f"Dedupe key already exists: {task.dedupe_key} (task_id={existing})")

        # drop は新規 enqueue しない方針（同一依頼を既存 task に集約）
        return False
