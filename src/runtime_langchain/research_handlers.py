from __future__ import annotations

from typing import Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from runtime_core.tasks import WorkerLaunchRecorder, to_delayed_plans, to_periodic_plans
from runtime_core.types import TaskContext, TaskResult

from .runnable_handler import BeforeInvoke, ConfigMapper, RunnableTaskHandler
from .task_orchestrator import GraphInput, TaskOrchestrator


def _default_main_prompt(topic: str) -> str:
    return f"# user_input\n{topic}"


def _default_worker_prompt(query: str) -> str:
    return f"Perform deep research and summarize: {query}"


def _default_agent_config(ctx: TaskContext) -> RunnableConfig | None:
    thread_id = ctx.task.metadata.get("thread_id")
    if isinstance(thread_id, str) and thread_id.strip():
        return {"configurable": {"thread_id": thread_id}}
    return None


def _speaker_type(ctx: TaskContext) -> str:
    return str(ctx.task.metadata.get("speaker_type", "unknown"))


def _user_message(prompt: str) -> GraphInput:
    return GraphInput(messages=[{"role": "user", "content": prompt}])


def _build_main_input(
    ctx: TaskContext,
    prompt_builder: Callable[[str], str],
) -> GraphInput:
    topic = str(ctx.task.payload["topic"])
    prompt = f"[speaker_type={_speaker_type(ctx)}]\n{prompt_builder(topic)}"
    return _user_message(prompt)


def _build_worker_input(
    ctx: TaskContext,
    prompt_builder: Callable[[str], str],
) -> GraphInput:
    query = str(ctx.task.payload["query"])
    prompt = f"[speaker_type={_speaker_type(ctx)}]\n{prompt_builder(query)}"
    return _user_message(prompt)


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


def _prepare_main_input(
    ctx: TaskContext,
    inp: GraphInput,
    recorder: WorkerLaunchRecorder,
) -> GraphInput:
    recorder.drain()
    _record_scheduled_workers(ctx, recorder)
    return inp


def _coerce_graph_output(raw: object) -> GraphInput:
    if isinstance(raw, dict):
        messages = raw.get("messages")
        if isinstance(messages, list):
            return GraphInput(messages=messages)
    return GraphInput(messages=[{"role": "assistant", "content": str(raw).strip()}])


def _apply_after_invoke(
    ctx: TaskContext,
    raw: object,
    after_invoke: Callable[[TaskContext, GraphInput], GraphInput] | None,
) -> object:
    normalized = _coerce_graph_output(raw)
    if after_invoke is None:
        return normalized
    return after_invoke(ctx, normalized)


def _build_main_result(
    ctx: TaskContext,
    raw: object,
    orchestrator: TaskOrchestrator,
) -> TaskResult:
    return orchestrator.build_main_result(ctx, _coerce_graph_output(raw))


def _build_worker_result(
    ctx: TaskContext,
    raw: object,
    orchestrator: TaskOrchestrator,
) -> TaskResult:
    return orchestrator.build_worker_result(ctx, _coerce_graph_output(raw))


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
        build_prompt = prompt_builder or _default_main_prompt
        prepare_input = before_invoke or (
            lambda ctx, inp: _prepare_main_input(ctx, inp, orchestrator.recorder)
        )
        super().__init__(
            ainvoke=runnable.ainvoke,
            input_mapper=lambda ctx: _build_main_input(ctx, build_prompt),
            output_mapper=lambda ctx, raw: _build_main_result(ctx, raw, orchestrator),
            config_mapper=config_mapper or _default_agent_config,
            before_invoke=prepare_input,
            after_invoke=lambda ctx, raw: _apply_after_invoke(ctx, raw, after_invoke),
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
        build_prompt = prompt_builder or _default_worker_prompt
        super().__init__(
            ainvoke=runnable.ainvoke,
            input_mapper=lambda ctx: _build_worker_input(ctx, build_prompt),
            output_mapper=lambda ctx, raw: _build_worker_result(ctx, raw, orchestrator),
            config_mapper=config_mapper or _default_agent_config,
            before_invoke=before_invoke,
            after_invoke=lambda ctx, raw: _apply_after_invoke(ctx, raw, after_invoke),
        )
