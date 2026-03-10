from __future__ import annotations

from dataclasses import dataclass

from runtime_core.agent_types import (
    DelayedWorkerPlan,
    MainAgentRawResult,
    PeriodicWorkerPlan,
    WorkerAgentOutput,
)
from runtime_core.models import Task, TaskContext, TaskResult
from runtime_core.notifications import (
    NotificationPayload,
    extract_notification_metadata,
    render_output_message,
)


@dataclass(slots=True)
class TaskFlowConfig:
    worker_task_kind: str
    notification_task_kind: str
    periodic_min_interval_seconds: float = 1.0
    notification_kind_main: str = "main_result"
    notification_kind_worker: str = "worker_result"


def build_main_task_result(
    ctx: TaskContext,
    raw: MainAgentRawResult,
    *,
    config: TaskFlowConfig,
) -> TaskResult:
    metadata = _build_main_metadata(ctx)
    next_tasks = [
        *_build_main_tasks(ctx, raw, metadata, config),
        *_build_main_notifications(ctx, raw, metadata, config),
    ]
    return _success_result("agent_output", raw["agent_output"], next_tasks)


def build_worker_task_result(
    ctx: TaskContext,
    raw: WorkerAgentOutput,
    *,
    config: TaskFlowConfig,
) -> TaskResult:
    metadata = _build_worker_metadata(ctx)
    next_tasks = [
        *_build_worker_tasks(ctx, metadata, config),
        *_build_worker_notifications(ctx, raw, metadata, config),
    ]
    return _success_result("worker_output", raw, next_tasks)


def resolve_enqueued_at(ctx: TaskContext) -> float:
    return _to_float(ctx.task.metadata.get("enqueued_at_unix"), default=0.0)


def _build_main_metadata(ctx: TaskContext) -> NotificationPayload:
    return _build_notification_metadata(ctx, ctx.task.id)


def _build_worker_metadata(ctx: TaskContext) -> NotificationPayload:
    request_id = str(
        ctx.task.metadata.get(
            "discord_request_task_id", ctx.task.parent_task_id or ctx.task.id
        )
    )
    return _build_notification_metadata(ctx, request_id)


def _build_main_tasks(
    ctx: TaskContext,
    raw: MainAgentRawResult,
    metadata: NotificationPayload,
    config: TaskFlowConfig,
) -> list[Task]:
    base_time = resolve_enqueued_at(ctx)
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
    config: TaskFlowConfig,
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


def _build_main_notifications(
    ctx: TaskContext,
    raw: MainAgentRawResult,
    metadata: NotificationPayload,
    config: TaskFlowConfig,
) -> list[Task]:
    return [
        _new_notification_task(
            task_id=f"notification:{ctx.task.id}:main",
            parent_task_id=ctx.task.id,
            message=f"main result: {render_output_message(raw['agent_output'])}",
            metadata=metadata,
            notification_kind=config.notification_kind_main,
            notification_task_kind=config.notification_task_kind,
        )
    ]


def _build_worker_notifications(
    ctx: TaskContext,
    raw: WorkerAgentOutput,
    metadata: NotificationPayload,
    config: TaskFlowConfig,
) -> list[Task]:
    return [
        _new_notification_task(
            task_id=f"notification:{ctx.task.id}:worker_done",
            parent_task_id=ctx.task.parent_task_id,
            message=f"worker finished ({ctx.task.id}): {render_output_message(raw)}",
            metadata=metadata,
            notification_kind=config.notification_kind_worker,
            notification_task_kind=config.notification_task_kind,
        )
    ]


def _build_notification_metadata(
    ctx: TaskContext, request_id: str
) -> NotificationPayload:
    metadata = extract_notification_metadata(ctx.task.metadata)
    metadata["discord_request_task_id"] = request_id
    return metadata


def _success_result(
    output_key: str,
    output_value: object,
    next_tasks: list[Task],
) -> TaskResult:
    return TaskResult(
        status="succeeded",
        output={output_key: output_value},
        next_tasks=next_tasks,
    )


@dataclass(frozen=True, slots=True)
class _PeriodicState:
    root_id: str
    iteration: int
    remaining_runs: int
    interval_seconds: float
    next_run_after: float
    query: str


def _extract_periodic_state(ctx: TaskContext) -> _PeriodicState | None:
    remaining_runs = _to_int(ctx.task.payload.get("remaining_runs"), default=1)
    interval_seconds = _to_float(
        ctx.task.payload.get("periodic_interval_seconds"), default=0.0
    )
    if remaining_runs <= 1 or interval_seconds <= 0.0:
        return None
    root_id = str(ctx.task.payload.get("periodic_root_id", ctx.task.id))
    iteration = _to_int(ctx.task.payload.get("periodic_iteration"), default=1)
    return _PeriodicState(
        root_id=root_id,
        iteration=iteration,
        remaining_runs=remaining_runs,
        interval_seconds=interval_seconds,
        next_run_after=(ctx.task.run_after or 0.0) + interval_seconds,
        query=str(ctx.task.payload.get("query", "")),
    )


def to_delayed_plans(value: object) -> list[DelayedWorkerPlan]:
    plans: list[DelayedWorkerPlan] = []
    for item in _iter_dict_items(value):
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        plans.append(
            DelayedWorkerPlan(
                query=query,
                delay_seconds=max(_to_float(item.get("delay_seconds"), 0.0), 0.0),
            )
        )
    return plans


def to_periodic_plans(
    value: object, *, min_interval_seconds: float = 1.0
) -> list[PeriodicWorkerPlan]:
    plans: list[PeriodicWorkerPlan] = []
    for item in _iter_dict_items(value):
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        plans.append(
            PeriodicWorkerPlan(
                query=query,
                start_in_seconds=max(_to_float(item.get("start_in_seconds"), 0.0), 0.0),
                interval_seconds=max(
                    _to_float(item.get("interval_seconds"), 60.0),
                    min_interval_seconds,
                ),
                repeat_count=max(_to_int(item.get("repeat_count"), 1), 1),
            )
        )
    return plans


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


def _iter_dict_items(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _to_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _to_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
