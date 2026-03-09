from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypedDict

from runtime_core.models import Task, TaskContext, TaskResult
from runtime_core.registry import HandlerRegistry
from runtime_core.repository import InMemoryTaskRepository
from runtime_core.runtime import Runtime
from runtime_langchain.runnable_handler import RunnableTaskHandler
from examples.deep_agent_runtime.common import normalize_text
from examples.deep_agent_runtime.main_agent_runnables import (
    MainAgentRawResult,
    MainAgentRunnable,
    DelayedWorkerPlan,
    PeriodicWorkerPlan,
    WorkerLaunchRecorder,
    build_main_agent_runnable,
)
from examples.deep_agent_runtime.worker_agent_runnables import (
    WorkerAgentRunnable,
    build_worker_agent_runnable,
    resolve_deepagent_artifact_dir,
)

TASK_KIND_MAIN_RESEARCH = "main_research"
TASK_KIND_WORKER_RESEARCH = "worker_research"
EXAMPLE_TASK_ID = "example:main:1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
_REAL_AGENT_ENV = "EXAMPLE_USE_REAL_DEEP_AGENT"
_MODEL_ENV = "EXAMPLE_MODEL"
_BACKEND_ENV = "EXAMPLE_REAL_AGENT_BACKEND"
_BACKEND_LANGCHAIN = "langchain"
_BACKEND_DEEPAGENT = "deepagent"
_PERIODIC_MIN_INTERVAL_SECONDS = 1.0
_DEEPAGENT_ARTIFACT_DIR_ENV = "EXAMPLE_DEEPAGENT_ARTIFACT_DIR"

@dataclass(slots=True)
class ExampleRuntimeBundle:
    runtime: Runtime
    repository: InMemoryTaskRepository


def build_example_runtime() -> ExampleRuntimeBundle:
    repository = InMemoryTaskRepository()
    registry = HandlerRegistry()

    worker_recorder = WorkerLaunchRecorder()
    registry.register(
        TASK_KIND_MAIN_RESEARCH,
        RunnableTaskHandler(
            runnable=_build_main_agent_runnable(recorder=worker_recorder),
            input_mapper=_build_main_agent_input,
            config_mapper=_build_agent_config,
            output_mapper=_build_main_task_result,
        ),
    )
    registry.register(
        TASK_KIND_WORKER_RESEARCH,
        RunnableTaskHandler(
            runnable=_build_worker_agent_runnable(),
            input_mapper=_build_worker_agent_input,
            config_mapper=_build_agent_config,
            output_mapper=_build_worker_task_result,
        ),
    )

    return ExampleRuntimeBundle(runtime=Runtime(repository=repository, registry=registry), repository=repository)


def seed_example_task(repository: InMemoryTaskRepository, topic: str, needs_worker: bool = True) -> Task:
    task = Task(
        id=EXAMPLE_TASK_ID,
        kind=TASK_KIND_MAIN_RESEARCH,
        payload={"topic": topic, "needs_worker": needs_worker},
        metadata={"enqueued_at_unix": 0.0},
    )
    repository.enqueue(task)
    return task


def _build_main_agent_runnable(recorder: WorkerLaunchRecorder) -> MainAgentRunnable:
    return build_main_agent_runnable(
        use_real_agent=_is_real_agent_enabled(),
        model_name=os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME),
        recorder=recorder,
    )


def _build_worker_agent_runnable() -> WorkerAgentRunnable:
    return build_worker_agent_runnable(
        use_real_agent=_is_real_agent_enabled(),
        backend=_resolve_real_agent_backend(),
        model_name=os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME),
        artifact_dir=resolve_deepagent_artifact_dir(_DEEPAGENT_ARTIFACT_DIR_ENV),
    )


def _is_real_agent_enabled() -> bool:
    return os.getenv(_REAL_AGENT_ENV, "0") == "1"


def _resolve_real_agent_backend() -> str:
    selected = os.getenv(_BACKEND_ENV, _BACKEND_LANGCHAIN).strip().lower()
    if selected == _BACKEND_DEEPAGENT:
        return _BACKEND_DEEPAGENT
    return _BACKEND_LANGCHAIN


def _build_main_agent_input(ctx: TaskContext) -> AgentRequest:
    topic = str(ctx.task.payload["topic"])
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Handle this user request. "
                    "Use worker tools for heavy deep research work if needed. "
                    f"topic={topic}, needs_worker={bool(ctx.task.payload.get('needs_worker', False))}"
                ),
            }
        ],
        "topic": topic,
        "needs_worker": bool(ctx.task.payload.get("needs_worker", False)),
        "delayed_jobs": _to_delayed_plans(ctx.task.payload.get("delayed_jobs", [])),
        "periodic_jobs": _to_periodic_plans(ctx.task.payload.get("periodic_jobs", [])),
    }


def _build_worker_agent_input(ctx: TaskContext) -> AgentRequest:
    query = str(ctx.task.payload["query"])
    return {
        "messages": [{"role": "user", "content": f"Perform deep research and summarize: {query}"}],
        "query": query,
    }


def _build_agent_config(ctx: TaskContext) -> AgentConfig:
    return {"task_id": ctx.task.id, "attempt": ctx.attempt}


def _build_main_task_result(ctx: TaskContext, raw: object) -> TaskResult:
    main_raw = _to_main_raw_result(raw)
    base_time = _resolve_enqueued_at(ctx)

    immediate_tasks = [
        _new_worker_task(
            task_id=f"worker:{ctx.task.id}:now:{index}",
            parent_task_id=ctx.task.id,
            query=query,
            run_after=None,
        )
        for index, query in enumerate(main_raw["immediate_queries"], start=1)
    ]

    delayed_tasks = [
        _new_worker_task(
            task_id=f"worker:{ctx.task.id}:delayed:{index}",
            parent_task_id=ctx.task.id,
            query=plan["query"],
            run_after=base_time + plan["delay_seconds"],
        )
        for index, plan in enumerate(main_raw["delayed_queries"], start=1)
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
        )
        for index, plan in enumerate(main_raw["periodic_queries"], start=1)
    ]

    return TaskResult(
        status="succeeded",
        output={"agent_output": main_raw["agent_output"]},
        next_tasks=[*immediate_tasks, *delayed_tasks, *periodic_tasks],
    )


def _build_worker_task_result(ctx: TaskContext, raw: object) -> TaskResult:
    remaining_runs = _to_int(ctx.task.payload.get("remaining_runs"), default=1)
    interval_seconds = _to_float(ctx.task.payload.get("periodic_interval_seconds"), default=0.0)
    if remaining_runs <= 1 or interval_seconds <= 0.0:
        return TaskResult(status="succeeded", output={"worker_output": raw})

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
    )
    return TaskResult(status="succeeded", output={"worker_output": raw}, next_tasks=[next_task])


def _new_worker_task(
    task_id: str,
    parent_task_id: str | None,
    query: str,
    run_after: float | None,
) -> Task:
    return Task(
        id=task_id,
        kind=TASK_KIND_WORKER_RESEARCH,
        payload={"query": query},
        parent_task_id=parent_task_id,
        run_after=run_after,
    )


def _new_periodic_worker_task(
    task_id: str,
    parent_task_id: str | None,
    query: str,
    periodic_root_id: str,
    iteration: int,
    remaining_runs: int,
    interval_seconds: float,
    run_after: float,
) -> Task:
    task = _new_worker_task(
        task_id=task_id,
        parent_task_id=parent_task_id,
        query=query,
        run_after=run_after,
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


def _to_main_raw_result(raw: object) -> MainAgentRawResult:
    if not isinstance(raw, dict):
        return MainAgentRawResult(
            agent_output=raw,
            immediate_queries=[],
            delayed_queries=[],
            periodic_queries=[],
        )

    return MainAgentRawResult(
        agent_output=raw.get("agent_output", raw),
        immediate_queries=_to_query_list(raw.get("immediate_queries", [])),
        delayed_queries=_to_delayed_plans(raw.get("delayed_queries", [])),
        periodic_queries=_to_periodic_plans(raw.get("periodic_queries", [])),
    )


def _to_agent_request(inp: AgentRequest | str) -> AgentRequest:
    if isinstance(inp, dict):
        return inp
    return {"messages": [{"role": "user", "content": inp}]}


def _resolve_enqueued_at(ctx: TaskContext) -> float:
    return _to_float(ctx.task.metadata.get("enqueued_at_unix"), default=0.0)


def _to_query_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [query for query in (normalize_text(item) for item in value) if query]


def _to_delayed_plans(value: object) -> list[DelayedWorkerPlan]:
    plans: list[DelayedWorkerPlan] = []
    for item in _iter_dict_items(value):
        query = normalize_text(item.get("query", ""))
        if not query:
            continue
        plans.append(DelayedWorkerPlan(query=query, delay_seconds=max(_to_float(item.get("delay_seconds"), 0.0), 0.0)))
    return plans


def _to_periodic_plans(value: object) -> list[PeriodicWorkerPlan]:
    plans: list[PeriodicWorkerPlan] = []
    for item in _iter_dict_items(value):
        query = normalize_text(item.get("query", ""))
        if not query:
            continue
        plans.append(
            PeriodicWorkerPlan(
                query=query,
                start_in_seconds=max(_to_float(item.get("start_in_seconds"), 0.0), 0.0),
                interval_seconds=max(_to_float(item.get("interval_seconds"), 60.0), _PERIODIC_MIN_INTERVAL_SECONDS),
                repeat_count=max(_to_int(item.get("repeat_count"), 1), 1),
            )
        )
    return plans


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
