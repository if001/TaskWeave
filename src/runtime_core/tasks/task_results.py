from __future__ import annotations

from dataclasses import dataclass

from ..types import MainAgentRawResult, Task, TaskContext, TaskResult, WorkerAgentOutput
from ..notifications import (
    NotificationPayload,
    extract_notification_metadata,
    render_output_message,
)
from .task_plans import parse_float, parse_int


@dataclass(slots=True)
class TaskResultConfig:
    worker_task_kind: str
    notification_task_kind: str
    notification_kind_main: str = "main_result"
    notification_kind_worker: str = "worker_result"


def build_main_task_result(
    ctx: TaskContext,
    raw: MainAgentRawResult,
    *,
    config: TaskResultConfig,
) -> TaskResult:
    metadata = extract_notification_metadata(ctx.task.metadata)
    metadata["discord_request_task_id"] = ctx.task.id
    next_tasks = _build_main_tasks(ctx, raw, metadata, config)
    next_tasks.append(
        _new_notification_task(
            task_id=f"notification:{ctx.task.id}:main",
            parent_task_id=ctx.task.id,
            message=f"main result: {render_output_message(raw['agent_output'])}",
            metadata=metadata,
            notification_kind=config.notification_kind_main,
            notification_task_kind=config.notification_task_kind,
        )
    )
    return TaskResult(
        status="succeeded",
        output={"agent_output": raw["agent_output"]},
        next_tasks=next_tasks,
    )


def build_worker_task_result(
    ctx: TaskContext,
    raw: WorkerAgentOutput,
    *,
    config: TaskResultConfig,
) -> TaskResult:
    metadata = extract_notification_metadata(ctx.task.metadata)
    request_id = str(
        ctx.task.metadata.get(
            "discord_request_task_id", ctx.task.parent_task_id or ctx.task.id
        )
    )
    metadata["discord_request_task_id"] = request_id
    next_tasks = _build_worker_tasks(ctx, metadata, config)
    next_tasks.append(
        _new_notification_task(
            task_id=f"notification:{ctx.task.id}:worker_done",
            parent_task_id=ctx.task.parent_task_id,
            message=f"worker finished ({ctx.task.id}): {render_output_message(raw)}",
            metadata=metadata,
            notification_kind=config.notification_kind_worker,
            notification_task_kind=config.notification_task_kind,
        )
    )
    return TaskResult(
        status="succeeded",
        output={"worker_output": raw},
        next_tasks=next_tasks,
    )


def _build_main_tasks(
    ctx: TaskContext,
    raw: MainAgentRawResult,
    metadata: NotificationPayload,
    config: TaskResultConfig,
) -> list[Task]:
    base_time = parse_float(ctx.task.metadata.get("enqueued_at_unix"), default=0.0)
    immediate_tasks = [
        _new_worker_task(
            task_id=f"worker:{ctx.task.id}:now:{index}",
            parent_task_id=ctx.task.id,
            query=query,
            run_after=None,
            metadata=metadata,
            worker_task_kind=config.worker_task_kind,
        )
        for index, query in enumerate(raw["immediate_queries"], start=1)
    ]
    delayed_tasks = [
        _new_worker_task(
            task_id=f"worker:{ctx.task.id}:delayed:{index}",
            parent_task_id=ctx.task.id,
            query=plan["query"],
            run_after=base_time + plan["delay_seconds"],
            metadata=metadata,
            worker_task_kind=config.worker_task_kind,
        )
        for index, plan in enumerate(raw["delayed_queries"], start=1)
    ]
    periodic_tasks = [
        _new_periodic_worker_task(
            task_id=f"worker:{ctx.task.id}:periodic:{index}:1",
            parent_task_id=ctx.task.id,
            query=plan["query"],
            periodic_root_id=f"worker:{ctx.task.id}:periodic:{index}",
            iteration=1,
            remaining_runs=plan["repeat_count"],
            interval_seconds=plan["interval_seconds"],
            run_after=base_time + plan["start_in_seconds"],
            metadata=metadata,
            worker_task_kind=config.worker_task_kind,
        )
        for index, plan in enumerate(raw["periodic_queries"], start=1)
    ]
    return [*immediate_tasks, *delayed_tasks, *periodic_tasks]


def _build_worker_tasks(
    ctx: TaskContext,
    metadata: NotificationPayload,
    config: TaskResultConfig,
) -> list[Task]:
    periodic = _extract_periodic_state(ctx)
    if periodic is None:
        return []
    next_task = _new_periodic_worker_task(
        task_id=f"{periodic.root_id}:{periodic.iteration + 1}",
        parent_task_id=ctx.task.parent_task_id,
        query=periodic.query,
        periodic_root_id=periodic.root_id,
        iteration=periodic.iteration + 1,
        remaining_runs=periodic.remaining_runs - 1,
        interval_seconds=periodic.interval_seconds,
        run_after=periodic.next_run_after,
        metadata=metadata,
        worker_task_kind=config.worker_task_kind,
    )
    return [next_task]


@dataclass(frozen=True, slots=True)
class _PeriodicState:
    root_id: str
    iteration: int
    remaining_runs: int
    interval_seconds: float
    next_run_after: float
    query: str


def _extract_periodic_state(ctx: TaskContext) -> _PeriodicState | None:
    remaining_runs = parse_int(ctx.task.payload.get("remaining_runs"), default=1)
    interval_seconds = parse_float(
        ctx.task.payload.get("periodic_interval_seconds"), default=0.0
    )
    if remaining_runs <= 1 or interval_seconds <= 0.0:
        return None
    root_id = str(ctx.task.payload.get("periodic_root_id", ctx.task.id))
    iteration = parse_int(ctx.task.payload.get("periodic_iteration"), default=1)
    return _PeriodicState(
        root_id=root_id,
        iteration=iteration,
        remaining_runs=remaining_runs,
        interval_seconds=interval_seconds,
        next_run_after=(ctx.task.run_after or 0.0) + interval_seconds,
        query=str(ctx.task.payload.get("query", "")),
    )


def _new_worker_task(
    *,
    task_id: str,
    parent_task_id: str | None,
    query: str,
    run_after: float | None,
    metadata: NotificationPayload,
    worker_task_kind: str,
) -> Task:
    return Task(
        id=task_id,
        kind=worker_task_kind,
        payload={"query": query},
        parent_task_id=parent_task_id,
        run_after=run_after,
        metadata=dict(metadata),
    )


def _new_periodic_worker_task(
    *,
    task_id: str,
    parent_task_id: str | None,
    query: str,
    periodic_root_id: str,
    iteration: int,
    remaining_runs: int,
    interval_seconds: float,
    run_after: float,
    metadata: NotificationPayload,
    worker_task_kind: str,
) -> Task:
    task = _new_worker_task(
        task_id=task_id,
        parent_task_id=parent_task_id,
        query=query,
        run_after=run_after,
        metadata=metadata,
        worker_task_kind=worker_task_kind,
    )
    task.payload.update(
        {
            "periodic_interval_seconds": interval_seconds,
            "remaining_runs": remaining_runs,
            "periodic_root_id": periodic_root_id,
            "periodic_iteration": iteration,
        }
    )
    return task


def _new_notification_task(
    *,
    task_id: str,
    parent_task_id: str | None,
    message: str,
    metadata: NotificationPayload,
    notification_kind: str,
    notification_task_kind: str,
) -> Task:
    payload: NotificationPayload = {
        "message": message,
        "notification_kind": notification_kind,
        **metadata,
    }
    return Task(
        id=task_id,
        kind=notification_task_kind,
        payload=dict(payload),
        parent_task_id=parent_task_id,
    )
