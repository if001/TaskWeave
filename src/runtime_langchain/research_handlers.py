from __future__ import annotations

from typing import Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from runtime_core.tasks import WorkerLaunchRecorder
from runtime_core.tasks import to_delayed_plans, to_periodic_plans
from runtime_core.types import TaskContext, TaskResult
from .runnable_handler import (
    AfterInvoke,
    BeforeInvoke,
    ConfigMapper,
    RunnableTaskHandler,
)
from .task_orchestrator import (
    GraphInput,
    TaskOrchestrator,
)


def _default_main_prompt(topic: str) -> str:
    return f"# user_input\n{topic}"


def _default_worker_prompt(query: str) -> str:
    return f"Perform deep research and summarize: {query}"


def _default_agent_config(ctx: TaskContext) -> RunnableConfig | None:
    thread_id = ctx.task.metadata.get("thread_id")
    if isinstance(thread_id, str) and thread_id.strip():
        return {"configurable": {"thread_id": thread_id}}
    return None


class MainResearchTaskHandler(RunnableTaskHandler):
    def __init__(
        self,
        runnable: CompiledStateGraph[GraphInput, None, GraphInput, GraphInput],
        orchestrator: TaskOrchestrator,
        prompt_builder: Callable[[str], str] | None = None,
        config_mapper: ConfigMapper[RunnableConfig] | None = None,
        before_invoke: BeforeInvoke[GraphInput] | None = None,
        after_invoke: Callable[[TaskContext, GraphInput], GraphInput] | None = None,
    ) -> None:
        prompt_builder = prompt_builder or _default_main_prompt

        def _input(ctx: TaskContext) -> GraphInput:
            topic = str(ctx.task.payload["topic"])
            prompt = prompt_builder(topic)

            return GraphInput(
                messages=[{"role": "user", "content": prompt}],
            )

        def _before(ctx: TaskContext, inp: GraphInput) -> GraphInput:
            recorder = orchestrator.recorder
            recorder.drain()
            _record_scheduled_workers(ctx, recorder)
            return inp

        def _output(ctx: TaskContext, raw: object) -> TaskResult:
            _ = ctx
            normalized = _coerce_graph_output(raw)
            return orchestrator.build_main_result(ctx, normalized)

        def _after(ctx: TaskContext, raw: object) -> object:
            normalized = _coerce_graph_output(raw)
            if after_invoke is None:
                return normalized
            return after_invoke(ctx, normalized)

        super().__init__(
            ainvoke=runnable.ainvoke,
            input_mapper=_input,
            output_mapper=_output,
            config_mapper=config_mapper or _default_agent_config,
            before_invoke=before_invoke or _before,
            after_invoke=_after,
        )


class WorkerResearchTaskHandler(RunnableTaskHandler):
    def __init__(
        self,
        runnable: CompiledStateGraph[GraphInput, None, GraphInput, GraphInput],
        orchestrator: TaskOrchestrator,
        prompt_builder: Callable[[str], str] | None = None,
        config_mapper: ConfigMapper[RunnableConfig] | None = None,
        before_invoke: BeforeInvoke[GraphInput] | None = None,
        after_invoke: Callable[[TaskContext, GraphInput], GraphInput] | None = None,
    ) -> None:
        prompt_builder = prompt_builder or _default_worker_prompt

        def _input(ctx: TaskContext) -> GraphInput:
            query = str(ctx.task.payload["query"])
            return GraphInput(
                messages=[{"role": "user", "content": prompt_builder(query)}],
            )

        def _output(ctx: TaskContext, raw: object) -> TaskResult:
            _ = ctx
            normalized = _coerce_graph_output(raw)
            return orchestrator.build_worker_result(ctx, normalized)

        super().__init__(
            ainvoke=runnable.ainvoke,
            input_mapper=_input,
            output_mapper=_output,
            config_mapper=config_mapper or _default_agent_config,
            before_invoke=before_invoke,
            after_invoke=_wrap_after_invoke(after_invoke),
        )


def _record_scheduled_workers(ctx: TaskContext, recorder: WorkerLaunchRecorder) -> None:
    for delayed in to_delayed_plans(ctx.task.payload.get("delayed_jobs", [])):
        recorder.request_worker_at(delayed["query"], delayed["delay_seconds"])
    for periodic in to_periodic_plans(ctx.task.payload.get("periodic_jobs", [])):
        recorder.request_worker_periodic(
            periodic["query"],
            periodic["start_in_seconds"],
            periodic["interval_seconds"],
            periodic["repeat_count"],
        )


def _coerce_graph_output(raw: object) -> GraphInput:
    if isinstance(raw, dict):
        messages = raw.get("messages")
        if isinstance(messages, list):
            return GraphInput(messages=messages)
    return GraphInput(messages=[{"role": "assistant", "content": str(raw).strip()}])


def _wrap_after_invoke(
    after_invoke: Callable[[TaskContext, GraphInput], GraphInput] | None,
) -> AfterInvoke[object] | None:
    if after_invoke is None:
        return None

    def _wrapped(ctx: TaskContext, raw: object) -> object:
        normalized = _coerce_graph_output(raw)
        return after_invoke(ctx, normalized)

    return _wrapped
