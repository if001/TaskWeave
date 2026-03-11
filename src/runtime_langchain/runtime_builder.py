from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from runtime_core.notifications import (
    NoopNotificationSender,
    NotificationSender,
    NotificationTaskHandler,
)
from runtime_core.runtime import HandlerRegistry, Runtime
from runtime_core.tasks import TaskResultConfig
from .research_handlers import MainResearchTaskHandler, WorkerResearchTaskHandler
from .task_orchestrator import (
    GraphInput,
    TaskOrchestrator,
)


@dataclass(slots=True)
class ResearchRuntimeBuilder:
    _orchestrator: TaskOrchestrator

    def __init__(self, runtime: Runtime, *, config: TaskResultConfig) -> None:
        self._orchestrator = TaskOrchestrator(
            config=config,
            recorder=runtime.recorder,
        )

    def worker_tools(self) -> list[BaseTool]:
        return self._orchestrator.worker_request_tools()

    def mock_main_graph(self) -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
        return self._orchestrator.mock_main_graph()

    def register_main(
        self,
        registry: HandlerRegistry,
        *,
        kind: str = "main",
        runnable: CompiledStateGraph[GraphInput, None, GraphInput, GraphInput],
        prompt_builder: Callable[[str], str] | None = None,
    ) -> None:
        registry.register(
            kind,
            MainResearchTaskHandler(
                runnable=runnable,
                orchestrator=self._orchestrator,
                prompt_builder=prompt_builder,
            ),
        )

    def register_worker(
        self,
        registry: HandlerRegistry,
        *,
        kind: str,
        runnable: CompiledStateGraph[GraphInput, None, GraphInput, GraphInput],
        prompt_builder: Callable[[str], str] | None = None,
    ) -> None:
        registry.register(
            kind,
            WorkerResearchTaskHandler(
                runnable=runnable,
                orchestrator=self._orchestrator,
                prompt_builder=prompt_builder,
            ),
        )

    def register_notification(
        self,
        registry: HandlerRegistry,
        *,
        kind: str,
        sender: NotificationSender | None = None,
    ) -> None:
        registry.register(
            kind,
            NotificationTaskHandler(
                sender=sender or NoopNotificationSender()
            ),
        )
