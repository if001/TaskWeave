from __future__ import annotations

import os
from dataclasses import dataclass

from runtime_core.models import Task
from runtime_core.registry import HandlerRegistry
from runtime_core.repository import FileTaskRepository, TaskRepository
from runtime_core.runtime import Runtime
from runtime_langchain.research_handlers import (
    MainResearchTaskHandler,
    WorkerResearchTaskHandler,
)
from runtime_langchain.runnable_handler import CompiledStateGraphLike
from runtime_core.notifications import (
    NoopNotificationSender,
    NotificationSender,
    NotificationTaskHandler,
)
from runtime_core.task_results import TaskResultConfig
from runtime_langchain.task_orchestrator import TaskOrchestrator
from examples.deep_agent_runtime.main_agent_runnables import build_main_deep_agent_graph
from examples.deep_agent_runtime.worker_agent_runnables import (
    build_worker_agent_graph,
    resolve_deepagent_artifact_dir,
)

TASK_KIND_MAIN_RESEARCH = "main_research"
TASK_KIND_WORKER_RESEARCH = "worker_research"
TASK_KIND_NOTIFICATION = "notification"
EXAMPLE_TASK_ID = "example:main:1"
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


def build_example_runtime(
    notification_sender: NotificationSender | None = None,
) -> ExampleRuntimeBundle:
    repository = FileTaskRepository("./.state/task.json")
    registry = HandlerRegistry()
    runtime = Runtime(repository=repository, registry=registry)
    orchestrator = TaskOrchestrator(
        config=TaskResultConfig(
            worker_task_kind=TASK_KIND_WORKER_RESEARCH,
            notification_task_kind=TASK_KIND_NOTIFICATION,
        ),
        recorder=runtime.recorder,
    )

    registry.register(
        TASK_KIND_MAIN_RESEARCH,
        MainResearchTaskHandler.for_langchain(
            runnable=_build_main_agent_graph(orchestrator),
            orchestrator=orchestrator,
        ),
    )
    registry.register(
        TASK_KIND_WORKER_RESEARCH,
        WorkerResearchTaskHandler.for_langchain(
            runnable=_build_worker_agent_graph(),
            orchestrator=orchestrator,
        ),
    )
    registry.register(
        TASK_KIND_NOTIFICATION,
        NotificationTaskHandler(sender=notification_sender or NoopNotificationSender()),
    )

    return ExampleRuntimeBundle(
        runtime=runtime,
        repository=repository,
    )


def seed_example_task(repository: TaskRepository, topic: str) -> Task:
    task = Task(
        id=EXAMPLE_TASK_ID,
        kind=TASK_KIND_MAIN_RESEARCH,
        payload={"topic": topic},
        metadata={"enqueued_at_unix": 0.0},
    )
    repository.enqueue(task)
    return task


def _build_main_agent_graph(
    orchestrator: TaskOrchestrator,
) -> CompiledStateGraphLike:
    model_name = os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME)
    if not _is_real_agent_enabled():
        return orchestrator.mock_main_graph()
    return build_main_deep_agent_graph(
        model_name=model_name,
        tools=orchestrator.worker_request_tools(),
        artifact_dir=resolve_deepagent_artifact_dir(_DEEPAGENT_ARTIFACT_DIR_ENV),
    )


def _build_worker_agent_graph() -> CompiledStateGraphLike:
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
