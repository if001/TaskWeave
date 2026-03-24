from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from langchain_core.callbacks.base import BaseCallbackHandler
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph, RunnableConfig

from .main_agent_runnables import build_main_deep_agent_graph
from .memory_reflection import LangMemMemoryHooks
from .worker_agent_runnables import (
    build_worker_agent_graph,
    resolve_deepagent_artifact_dir,
)
from runtime_core.infra.logging_utils import get_logger
from runtime_core.notifications import NotificationSender
from runtime_core.runtime import (
    FileTaskRepository,
    HandlerRegistry,
    RetryPolicy,
    Runtime,
    TaskRepository,
    TaskScheduler,
)
from runtime_core.tasks import TaskResultConfig
from runtime_core.types import Task
from runtime_core.types.models import TaskContext
from runtime_langchain.runtime_builder import ResearchRuntimeBuilder
from runtime_langchain.task_context_config import build_langgraph_configurable
from runtime_langchain.task_orchestrator import GraphInput

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
_MEMORY_REFLECTION_DELAY_SECONDS = 10 * 60

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
    resolved_repository = repository or FileTaskRepository("./.state/task.json")
    resolved_workspace_dir = _resolve_workspace_dir(workspace_dir)
    registry = HandlerRegistry()
    runtime = Runtime(
        repository=resolved_repository,
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
    memory_hooks = LangMemMemoryHooks(
        workspace_dir=resolved_workspace_dir,
        model_name=os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME),
        delay_seconds=_MEMORY_REFLECTION_DELAY_SECONDS,
    )

    async with (
        _build_main_agent_graph(
            builder,
            workspace_dir=resolved_workspace_dir,
            agent_id=agent_id,
        ) as main_graph,
    ):
        builder.register_main(
            registry,
            kind=TASK_KIND_MAIN_RESEARCH,
            runnable=main_graph,
            config_mapper=_build_config_mapper(agent_id),
            before_invoke=memory_hooks.before_invoke,
            after_invoke=memory_hooks.after_invoke,
        )
        builder.register_worker(
            registry,
            kind=TASK_KIND_WORKER_RESEARCH,
            runnable=build_worker_agent_graph(
                use_real_agent=_is_real_agent_enabled(),
                backend=_resolve_real_agent_backend(),
                model_name=os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME),
                workspace_dir=resolved_workspace_dir,
            ),
        )
        builder.register_notification(
            registry,
            kind=TASK_KIND_NOTIFICATION,
            sender=notification_sender,
        )
        try:
            yield ExampleRuntimeBundle(runtime=runtime, repository=resolved_repository)
        finally:
            memory_hooks.shutdown()


def seed_example_task(
    repository: TaskRepository,
    topic: str,
    *,
    turn: int = 1,
) -> Task:
    task = Task(
        id=build_example_task_id(turn=turn),
        kind=TASK_KIND_MAIN_RESEARCH,
        payload={"topic": topic},
        metadata={"enqueued_at_unix": 0.0},
    )
    repository.enqueue(task)
    return task


def build_example_task_id(*, turn: int) -> str:
    return f"example:main:{turn}:{uuid.uuid4().hex}"


def _build_config_mapper(agent_id: str):
    def config_mapper(ctx: TaskContext) -> RunnableConfig:
        config: RunnableConfig = {
            "configurable": build_langgraph_configurable(ctx),
            "recursion_limit": 50,
        }
        callbacks = _build_callbacks(ctx, agent_id)
        if callbacks:
            config["callbacks"] = callbacks
        return config

    return config_mapper


def _build_callbacks(ctx: TaskContext, agent_id: str) -> list[BaseCallbackHandler]:
    trace_id = _ensure_root_trace_id(ctx, agent_id)
    if not trace_id:
        return []
    try:
        return [CallbackHandler(trace_context={"trace_id": trace_id})]
    except Exception as exc:
        logger.warning("failed to initialize langfuse handler: %s", exc)
        return []


def _ensure_root_trace_id(ctx: TaskContext, agent_id: str) -> str:
    existing = ctx.task.metadata.get("root_trace_id")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    seed = _trace_seed(ctx, agent_id)
    try:
        trace_id = get_client().create_trace_id(seed=seed)
    except Exception as exc:
        logger.warning("failed to create langfuse trace id: %s", exc)
        return ""
    ctx.task.metadata["root_trace_id"] = trace_id
    return trace_id


def _trace_seed(ctx: TaskContext, agent_id: str) -> str:
    for key in ("conversation_id", "thread_id"):
        value = ctx.task.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return f"{agent_id}:{value.strip()}"
    return f"{agent_id}:{ctx.task.id}"


@asynccontextmanager
async def _build_main_agent_graph(
    builder: ResearchRuntimeBuilder,
    *,
    workspace_dir: Path,
    agent_id: str,
) -> AsyncIterator[CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]]:
    if not _is_real_agent_enabled():
        yield builder.mock_main_graph()
        return

    async with build_main_deep_agent_graph(
        model_name=os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME),
        tools=builder.main_tools(),
        workspace_dir=workspace_dir,
        agent_id=agent_id,
    ) as graph:
        yield graph


def _resolve_workspace_dir(workspace_dir: Path | None) -> Path:
    return workspace_dir or resolve_deepagent_artifact_dir(_DEEPAGENT_ARTIFACT_DIR_ENV)


def _is_real_agent_enabled() -> bool:
    return os.getenv(_REAL_AGENT_ENV, "0") == "1"


def _resolve_real_agent_backend() -> str:
    selected = os.getenv(_BACKEND_ENV, _BACKEND_LANGCHAIN).strip().lower()
    if selected == _BACKEND_DEEPAGENT:
        return _BACKEND_DEEPAGENT
    return _BACKEND_LANGCHAIN
