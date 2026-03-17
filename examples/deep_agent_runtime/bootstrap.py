from __future__ import annotations

import os
import uuid
from pathlib import Path
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from langfuse.langchain import CallbackHandler
from langfuse import get_client

from runtime_core.infra.logging_utils import get_logger
from runtime_core.types import Task
from runtime_core.runtime import (
    FileTaskRepository,
    HandlerRegistry,
    RetryPolicy,
    Runtime,
    TaskRepository,
    TaskScheduler,
)
from langgraph.graph.state import CompiledStateGraph, Runnable, RunnableConfig


from runtime_core.types.models import TaskContext
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

langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
langfuse_base_url = os.getenv("LANGFUSE_BASE_URL", "")

logger = get_logger(__name__)


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
    workspace_dir: Path | None = None,
    agent_id: str = "default",
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
    lf = get_client()
    trace_id = lf.create_trace_id(seed=agent_id)
    langfuse_handler = CallbackHandler(trace_context={"trace_id": f"{trace_id}"})

    def config_mapper(ctx: TaskContext) -> RunnableConfig:
        thread_id = ctx.task.metadata.get("thread_id")
        configurable: dict[str, str] = {}
        if isinstance(thread_id, str) and thread_id.strip():
            configurable["thread_id"] = thread_id
        return {
            "configurable": configurable or {"thread_id": "user-1"},
            "callbacks": [langfuse_handler],
            "recursion_limit": 50,
        }

    async with _build_main_agent_graph(
        builder,
        workspace_dir=workspace_dir,
        agent_id=agent_id,
    ) as main_graph:
        builder.register_main(
            registry,
            kind=TASK_KIND_MAIN_RESEARCH,
            runnable=main_graph,
            config_mapper=config_mapper,
        )
        builder.register_worker(
            registry,
            kind=TASK_KIND_WORKER_RESEARCH,
            runnable=_build_worker_agent_graph(workspace_dir=workspace_dir),
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
    *,
    workspace_dir: Path | None,
    agent_id: str = "default",
) -> AsyncIterator[CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]]:
    model_name = os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME)
    resolved_workspace = workspace_dir or resolve_deepagent_artifact_dir(
        _DEEPAGENT_ARTIFACT_DIR_ENV
    )
    if not _is_real_agent_enabled():
        yield builder.mock_main_graph()
        return
    async with build_main_deep_agent_graph(
        model_name=model_name,
        tools=builder.worker_tools(),
        workspace_dir=resolved_workspace,
        agent_id=agent_id,
    ) as graph:
        yield graph


def _build_worker_agent_graph(
    *, workspace_dir: Path | None
) -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
    model_name = os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME)
    resolved_workspace = workspace_dir or resolve_deepagent_artifact_dir(
        _DEEPAGENT_ARTIFACT_DIR_ENV
    )
    return build_worker_agent_graph(
        use_real_agent=_is_real_agent_enabled(),
        backend=_resolve_real_agent_backend(),
        model_name=model_name,
        artifact_dir=resolved_workspace,
    )


def _is_real_agent_enabled() -> bool:
    return os.getenv(_REAL_AGENT_ENV, "0") == "1"


def _resolve_real_agent_backend() -> str:
    selected = os.getenv(_BACKEND_ENV, _BACKEND_LANGCHAIN).strip().lower()
    if selected == _BACKEND_DEEPAGENT:
        return _BACKEND_DEEPAGENT
    return _BACKEND_LANGCHAIN
