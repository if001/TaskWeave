from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

TaskStatus = Literal[
    "queued",
    "leased",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]

ResultStatus = Literal["succeeded", "failed", "retry"]


@dataclass(slots=True)
class Task:
    id: str
    kind: str
    payload: dict[str, Any]
    status: TaskStatus = "queued"
    run_after: float | None = None
    parent_task_id: str | None = None
    dedupe_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskContext:
    task: Task
    attempt: int
    deadline_unix: float | None = None
    cancellation_requested: bool = False


@dataclass(slots=True)
class TaskResult:
    status: ResultStatus
    output: dict[str, Any] = field(default_factory=dict)
    next_tasks: list[Task] = field(default_factory=list)
    error: str | None = None
