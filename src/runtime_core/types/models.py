from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .json_types import JsonValue


def _empty_json_dict() -> dict[str, JsonValue]:
    return {}


def _empty_task_list() -> list["Task"]:
    return []

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
    payload: dict[str, JsonValue]
    status: TaskStatus = "queued"
    run_after: float | None = None
    parent_task_id: str | None = None
    dedupe_key: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=_empty_json_dict)


@dataclass(slots=True)
class TaskContext:
    task: Task
    attempt: int
    deadline_unix: float | None = None
    cancellation_requested: bool = False


@dataclass(slots=True)
class TaskResult:
    status: ResultStatus
    output: dict[str, JsonValue] = field(default_factory=_empty_json_dict)
    next_tasks: list[Task] = field(default_factory=_empty_task_list)
    error: str | None = None
