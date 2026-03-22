from __future__ import annotations

from dataclasses import dataclass

from ..types import (
    JsonValue,
    MainAgentOutput,
    MainAgentRawResult,
    Task,
    TaskContext,
    TaskResult,
    WorkerAgentOutput,
)
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
    memory_reflection_task_kind: str | None = None
    memory_reflection_delay_seconds: float = 0.0
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
    memory_task = _build_memory_reflection_task(ctx, raw, config)
    if memory_task is not None:
        next_tasks.append(memory_task)
    next_tasks.append(
        _new_notification_task(
            task_id=f"notification:{ctx.task.id}:main",
            parent_task_id=ctx.task.id,
            message=f"{render_output_message(raw['agent_output'])}",
            metadata=metadata,
            notification_kind=config.notification_kind_main,
            notification_task_kind=config.notification_task_kind,
        )
    )
    agent_output: JsonValue = _as_output_dict(raw["agent_output"])
    return TaskResult(
        status="succeeded",
        output={"agent_output": agent_output},
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
    worker_output: JsonValue = _as_output_dict(raw)
    return TaskResult(
        status="succeeded",
        output={"worker_output": worker_output},
        next_tasks=next_tasks,
    )


def _build_main_tasks(
    ctx: TaskContext,
    raw: MainAgentRawResult,
    metadata: NotificationPayload,
    config: TaskResultConfig,
) -> list[Task]:
    json_metadata = _as_json_dict(metadata)
    base_time = parse_float(ctx.task.metadata.get("enqueued_at_unix"), default=0.0)
    immediate_tasks = [
        _new_worker_task(
            task_id=f"worker:{ctx.task.id}:now:{index}",
            parent_task_id=ctx.task.id,
            query=query,
            run_after=None,
            metadata=json_metadata,
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
            metadata=json_metadata,
            worker_task_kind=config.worker_task_kind,
        )
        for index, plan in enumerate(raw["delayed_queries"], start=1)
    ]
    periodic_tasks = [
        _new_worker_task(
            task_id=f"worker:{ctx.task.id}:periodic:{index}:1",
            parent_task_id=ctx.task.id,
            query=plan["query"],
            periodic_root_id=f"worker:{ctx.task.id}:periodic:{index}",
            iteration=1,
            remaining_runs=plan["repeat_count"],
            interval_seconds=plan["interval_seconds"],
            run_after=base_time + plan["start_in_seconds"],
            metadata=json_metadata,
            worker_task_kind=config.worker_task_kind,
        )
        for index, plan in enumerate(raw["periodic_queries"], start=1)
    ]
    return [*immediate_tasks, *delayed_tasks, *periodic_tasks]


def _build_memory_reflection_task(
    ctx: TaskContext,
    raw: MainAgentRawResult,
    config: TaskResultConfig,
) -> Task | None:
    if config.memory_reflection_task_kind is None:
        return None
    user_input = str(ctx.task.payload.get("topic", "")).strip()
    assistant_output = render_output_message(raw["agent_output"])
    if not user_input or not assistant_output:
        return None
    metadata = _select_reflection_metadata(ctx.task.metadata)
    enqueued_at = parse_float(ctx.task.metadata.get("enqueued_at_unix"), default=0.0)
    run_after = enqueued_at + max(config.memory_reflection_delay_seconds, 0.0)
    dedupe_key = _build_memory_reflection_dedupe_key(metadata)
    if dedupe_key is None:
        return None
    metadata["replace_pending"] = True
    return Task(
        id=f"memory:{ctx.task.id}",
        kind=config.memory_reflection_task_kind,
        payload={
            "user_input": user_input,
            "assistant_output": assistant_output,
        },
        parent_task_id=ctx.task.id,
        dedupe_key=dedupe_key,
        run_after=run_after,
        metadata=metadata,
    )


def _select_reflection_metadata(
    metadata: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    keys = [
        "user_id",
        "conversation_id",
        "agent_id",
        "speaker_id",
        "speaker_type",
        "root_trace_id",
        "discord_requester_id",
        "enqueued_at_unix",
    ]
    selected: dict[str, JsonValue] = {}
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, (str, int, float, bool)):
            selected[key] = value
    return selected


def _build_memory_reflection_dedupe_key(
    metadata: dict[str, JsonValue],
) -> str | None:
    for key in ("conversation_id", "user_id", "discord_requester_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return f"memory_reflection:{key}:{value.strip()}"
        if isinstance(value, int):
            return f"memory_reflection:{key}:{value}"
    return None


def _build_worker_tasks(
    ctx: TaskContext,
    metadata: NotificationPayload,
    config: TaskResultConfig,
) -> list[Task]:
    json_metadata = _as_json_dict(metadata)
    periodic = _extract_periodic_state(ctx)
    if periodic is None:
        return []
    next_task = _new_worker_task(
        task_id=f"{periodic.root_id}:{periodic.iteration + 1}",
        parent_task_id=ctx.task.parent_task_id,
        query=periodic.query,
        periodic_root_id=periodic.root_id,
        iteration=periodic.iteration + 1,
        remaining_runs=periodic.remaining_runs - 1,
        interval_seconds=periodic.interval_seconds,
        run_after=periodic.next_run_after,
        metadata=json_metadata,
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
    metadata: dict[str, JsonValue],
    worker_task_kind: str,
    periodic_root_id: str | None = None,
    iteration: int | None = None,
    remaining_runs: int | None = None,
    interval_seconds: float | None = None,
) -> Task:
    payload: dict[str, JsonValue] = {"query": query}
    if periodic_root_id is not None:
        payload["periodic_root_id"] = periodic_root_id
    if iteration is not None:
        payload["periodic_iteration"] = iteration
    if remaining_runs is not None:
        payload["remaining_runs"] = remaining_runs
    if interval_seconds is not None:
        payload["periodic_interval_seconds"] = interval_seconds
    return Task(
        id=task_id,
        kind=worker_task_kind,
        payload=payload,
        parent_task_id=parent_task_id,
        run_after=run_after,
        metadata=dict(metadata),
    )


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
        payload=_as_json_dict(payload),
        parent_task_id=parent_task_id,
    )


def _as_json_dict(payload: NotificationPayload) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in payload.items():
        if isinstance(value, (int, str)):
            result[key] = value
    return result


def _as_output_dict(
    output: MainAgentOutput | WorkerAgentOutput,
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in output.items():
        result[key] = str(value)
    return result
