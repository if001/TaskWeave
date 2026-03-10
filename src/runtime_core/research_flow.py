from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

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
class ResearchFlowConfig:
    worker_task_kind: str
    notification_task_kind: str
    periodic_min_interval_seconds: float = 1.0
    notification_kind_main: str = "main_result"
    notification_kind_worker: str = "worker_result"


RawT = TypeVar("RawT")
ParsedT = TypeVar("ParsedT")


class _FlowTemplate(Generic[RawT, ParsedT]):
    def __init__(self, config: ResearchFlowConfig) -> None:
        self._config = config

    def build(self, ctx: TaskContext, raw: RawT) -> TaskResult:
        parsed = self._parse(raw, ctx)
        metadata = self._build_metadata(ctx)
        tasks = self._build_tasks(ctx, parsed, metadata)
        notifications = self._build_notifications(ctx, parsed, metadata)
        return self._finalize(parsed, tasks, notifications)

    def _parse(self, raw: RawT, ctx: TaskContext) -> ParsedT:
        _ = (raw, ctx)
        raise NotImplementedError

    def _build_metadata(self, ctx: TaskContext) -> NotificationPayload:
        metadata = extract_notification_metadata(ctx.task.metadata)
        request_id = str(
            ctx.task.metadata.get(
                "discord_request_task_id", ctx.task.parent_task_id or ctx.task.id
            )
        )
        metadata["discord_request_task_id"] = request_id
        return metadata

    def _build_tasks(
        self,
        ctx: TaskContext,
        parsed: ParsedT,
        metadata: NotificationPayload,
    ) -> list[Task]:
        raise NotImplementedError

    def _build_notifications(
        self,
        ctx: TaskContext,
        parsed: ParsedT,
        metadata: NotificationPayload,
    ) -> list[Task]:
        raise NotImplementedError

    def _finalize(
        self,
        parsed: ParsedT,
        tasks: list[Task],
        notifications: list[Task],
    ) -> TaskResult:
        raise NotImplementedError


class _MainFlow(_FlowTemplate[MainAgentRawResult, MainAgentRawResult]):
    def _parse(self, raw: MainAgentRawResult, ctx: TaskContext) -> MainAgentRawResult:
        _ = ctx
        return raw

    def _build_metadata(self, ctx: TaskContext) -> NotificationPayload:
        metadata = extract_notification_metadata(ctx.task.metadata)
        metadata["discord_request_task_id"] = ctx.task.id
        return metadata

    def _build_tasks(
        self,
        ctx: TaskContext,
        parsed: MainAgentRawResult,
        metadata: NotificationPayload,
    ) -> list[Task]:
        base_time = resolve_enqueued_at(ctx)
        immediate_tasks = [
            _new_worker_task(
                task_id=f"worker:{ctx.task.id}:now:{index}",
                parent_task_id=ctx.task.id,
                query=query,
                run_after=None,
                metadata=metadata,
                worker_task_kind=self._config.worker_task_kind,
            )
            for index, query in enumerate(parsed["immediate_queries"], start=1)
        ]
        delayed_tasks = [
            _new_worker_task(
                task_id=f"worker:{ctx.task.id}:delayed:{index}",
                parent_task_id=ctx.task.id,
                query=plan["query"],
                run_after=base_time + plan["delay_seconds"],
                metadata=metadata,
                worker_task_kind=self._config.worker_task_kind,
            )
            for index, plan in enumerate(parsed["delayed_queries"], start=1)
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
                worker_task_kind=self._config.worker_task_kind,
            )
            for index, plan in enumerate(parsed["periodic_queries"], start=1)
        ]
        return [*immediate_tasks, *delayed_tasks, *periodic_tasks]

    def _build_notifications(
        self,
        ctx: TaskContext,
        parsed: MainAgentRawResult,
        metadata: NotificationPayload,
    ) -> list[Task]:
        return [
            _new_notification_task(
                task_id=f"notification:{ctx.task.id}:main",
                parent_task_id=ctx.task.id,
                message=f"main result: {render_output_message(parsed['agent_output'])}",
                metadata=metadata,
                notification_kind=self._config.notification_kind_main,
                notification_task_kind=self._config.notification_task_kind,
            )
        ]

    def _finalize(
        self,
        parsed: MainAgentRawResult,
        tasks: list[Task],
        notifications: list[Task],
    ) -> TaskResult:
        return TaskResult(
            status="succeeded",
            output={"agent_output": parsed["agent_output"]},
            next_tasks=[*tasks, *notifications],
        )


class _WorkerFlow(_FlowTemplate[WorkerAgentOutput, WorkerAgentOutput]):
    def _parse(self, raw: WorkerAgentOutput, ctx: TaskContext) -> WorkerAgentOutput:
        _ = ctx
        return raw

    def _build_tasks(
        self,
        ctx: TaskContext,
        parsed: WorkerAgentOutput,
        metadata: NotificationPayload,
    ) -> list[Task]:
        remaining_runs = _to_int(ctx.task.payload.get("remaining_runs"), default=1)
        interval_seconds = _to_float(
            ctx.task.payload.get("periodic_interval_seconds"), default=0.0
        )
        if remaining_runs <= 1 or interval_seconds <= 0.0:
            return []

        root_id = str(ctx.task.payload.get("periodic_root_id", ctx.task.id))
        iteration = _to_int(ctx.task.payload.get("periodic_iteration"), default=1)
        next_task = _new_periodic_worker_task(
            task_id=f"{root_id}:{iteration + 1}",
            parent_task_id=ctx.task.parent_task_id,
            query=str(ctx.task.payload.get("query", "")),
            periodic_root_id=root_id,
            iteration=iteration + 1,
            remaining_runs=remaining_runs - 1,
            interval_seconds=interval_seconds,
            run_after=(ctx.task.run_after or 0.0) + interval_seconds,
            metadata=metadata,
            worker_task_kind=self._config.worker_task_kind,
        )
        return [next_task]

    def _build_notifications(
        self,
        ctx: TaskContext,
        parsed: WorkerAgentOutput,
        metadata: NotificationPayload,
    ) -> list[Task]:
        return [
            _new_notification_task(
                task_id=f"notification:{ctx.task.id}:worker_done",
                parent_task_id=ctx.task.parent_task_id,
                message=f"worker finished ({ctx.task.id}): {render_output_message(parsed)}",
                metadata=metadata,
                notification_kind=self._config.notification_kind_worker,
                notification_task_kind=self._config.notification_task_kind,
            )
        ]

    def _finalize(
        self,
        parsed: WorkerAgentOutput,
        tasks: list[Task],
        notifications: list[Task],
    ) -> TaskResult:
        return TaskResult(
            status="succeeded",
            output={"worker_output": parsed},
            next_tasks=[*tasks, *notifications],
        )


@dataclass(slots=True)
class ResearchFlow:
    config: ResearchFlowConfig

    def build_main_result(self, ctx: TaskContext, raw: MainAgentRawResult) -> TaskResult:
        return _MainFlow(self.config).build(ctx, raw)

    def build_worker_result(self, ctx: TaskContext, raw: WorkerAgentOutput) -> TaskResult:
        return _WorkerFlow(self.config).build(ctx, raw)


def build_main_task_result(
    ctx: TaskContext,
    raw: MainAgentRawResult,
    *,
    config: ResearchFlowConfig,
) -> TaskResult:
    return ResearchFlow(config).build_main_result(ctx, raw)


def build_worker_task_result(
    ctx: TaskContext,
    raw: WorkerAgentOutput,
    *,
    config: ResearchFlowConfig,
) -> TaskResult:
    return ResearchFlow(config).build_worker_result(ctx, raw)


def resolve_enqueued_at(ctx: TaskContext) -> float:
    return _to_float(ctx.task.metadata.get("enqueued_at_unix"), default=0.0)


def to_query_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


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
