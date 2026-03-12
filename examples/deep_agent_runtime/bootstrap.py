from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from runtime_core.types import Task
from runtime_core.runtime import (
    FileTaskRepository,
    HandlerRegistry,
    RetryPolicy,
    Runtime,
    TaskRepository,
    TaskScheduler,
)
from langgraph.graph.state import CompiledStateGraph
from runtime_langchain.task_orchestrator import GraphInput
from runtime_core.notifications import NotificationSender
from runtime_core.tasks import TaskResultConfig
from runtime_langchain.runtime_builder import ResearchRuntimeBuilder
from examples.deep_agent_runtime.main_agent_runnables import build_main_deep_agent_graph
from examples.deep_agent_runtime.worker_agent_runnables import (
    build_worker_agent_graph,
    resolve_deepagent_artifact_dir,
)

TASK_KIND_MAIN_RESEARCH = "main_research"
TASK_KIND_WORKER_RESEARCH = "worker_research"
TASK_KIND_NOTIFICATION = "notification"
DEFAULT_MODEL_NAME = "gpt-oss:20b"
_REAL_AGENT_ENV = "USE_REAL_DEEP_AGENT"
_MODEL_ENV = "MODEL_NAME"
_BACKEND_ENV = "REAL_AGENT_BACKEND"
_BACKEND_LANGCHAIN = "langchain"
_BACKEND_DEEPAGENT = "deepagent"
_DEEPAGENT_ARTIFACT_DIR_ENV = "DEEPAGENT_ARTIFACT_DIR"


@dataclass(slots=True)
class ExampleRuntimeBundle:
    runtime: Runtime
    repository: TaskRepository


@asynccontextmanager
async def build_example_runtime(
    notification_sender: NotificationSender | None = None,
    repository: TaskRepository | None = None,
    retry_policy: RetryPolicy | None = None,
    scheduler: TaskScheduler | None = None,
) -> AsyncIterator[ExampleRuntimeBundle]:
    repository = repository or FileTaskRepository("./.state/task.json")
    registry = HandlerRegistry()
    runtime = Runtime(
        repository=repository,
        registry=registry,
        retry_policy=retry_policy,
        scheduler=scheduler,
    )
    builder = ResearchRuntimeBuilder(
        runtime,
        config=TaskResultConfig(
            worker_task_kind=TASK_KIND_WORKER_RESEARCH,
            notification_task_kind=TASK_KIND_NOTIFICATION,
        ),
    )
    async with _build_main_agent_graph(builder) as main_graph:
        builder.register_main(
            registry,
            kind=TASK_KIND_MAIN_RESEARCH,
            runnable=main_graph,
        )
        builder.register_worker(
            registry,
            kind=TASK_KIND_WORKER_RESEARCH,
            runnable=_build_worker_agent_graph(),
        )
        builder.register_notification(
            registry,
            kind=TASK_KIND_NOTIFICATION,
            sender=notification_sender,
        )
        yield ExampleRuntimeBundle(
            runtime=runtime,
            repository=repository,
        )


def seed_example_task(
    repository: TaskRepository,
    topic: str,
    *,
    turn: int = 1,
) -> Task:
    task_id = build_example_task_id(turn=turn)
    task = Task(
        id=task_id,
        kind=TASK_KIND_MAIN_RESEARCH,
        payload={"topic": topic},
        metadata={"enqueued_at_unix": 0.0},
    )
    repository.enqueue(task)
    return task


def build_example_task_id(*, turn: int) -> str:
    return f"example:main:{turn}:{uuid.uuid4().hex}"


@asynccontextmanager
async def _build_main_agent_graph(
    builder: ResearchRuntimeBuilder,
) -> AsyncIterator[CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]]:
    model_name = os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME)
    if not _is_real_agent_enabled():
        yield builder.mock_main_graph()
        return
    async with build_main_deep_agent_graph(
        model_name=model_name,
        tools=builder.worker_tools(),
        artifact_dir=resolve_deepagent_artifact_dir(_DEEPAGENT_ARTIFACT_DIR_ENV),
    ) as graph:
        yield graph


def _build_worker_agent_graph() -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
    model_name = os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME)
    return build_worker_agent_graph(
        use_real_agent=_is_real_agent_enabled(),
        backend=_resolve_real_agent_backend(),
        model_name=model_name,
        artifact_dir=resolve_deepagent_artifact_dir(_DEEPAGENT_ARTIFACT_DIR_ENV),
    )


def _is_real_agent_enabled() -> bool:
    return os.getenv(_REAL_AGENT_ENV, "0") == "1"


def _resolve_real_agent_backend() -> str:
    selected = os.getenv(_BACKEND_ENV, _BACKEND_LANGCHAIN).strip().lower()
    if selected == _BACKEND_DEEPAGENT:
        return _BACKEND_DEEPAGENT
    return _BACKEND_LANGCHAIN
