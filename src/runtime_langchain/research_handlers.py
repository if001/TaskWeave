from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from runtime_core.tasks import WorkerLaunchRecorder, to_delayed_plans, to_periodic_plans
from runtime_core.types import TaskContext, TaskResult

from .runnable_handler import BeforeInvoke, ConfigMapper, RunnableTaskHandler
from .task_orchestrator import GraphInput, TaskOrchestrator

ResearchKind: TypeAlias = Literal["main", "worker"]
PromptBuilder = Callable[[str], str]
AfterInvokeHook = Callable[[TaskContext, GraphInput], GraphInput]
NormalizedAfterInvoke = Callable[[TaskContext, object], object]


def build_main_task_handler(
    runnable: CompiledStateGraph[GraphInput, None, GraphInput, GraphInput],
    orchestrator: TaskOrchestrator,
    prompt_builder: PromptBuilder | None = None,
    config_mapper: ConfigMapper[RunnableConfig] | None = None,
    before_invoke: BeforeInvoke[GraphInput] | None = None,
    after_invoke: AfterInvokeHook | None = None,
) -> RunnableTaskHandler:
    return ResearchTaskHandler(
        kind="main",
        runnable=runnable,
        orchestrator=orchestrator,
        prompt_builder=prompt_builder,
        config_mapper=config_mapper,
        before_invoke=before_invoke,
        after_invoke=after_invoke,
    )


def build_worker_task_handler(
    runnable: CompiledStateGraph[GraphInput, None, GraphInput, GraphInput],
    orchestrator: TaskOrchestrator,
    prompt_builder: PromptBuilder | None = None,
    config_mapper: ConfigMapper[RunnableConfig] | None = None,
    before_invoke: BeforeInvoke[GraphInput] | None = None,
    after_invoke: AfterInvokeHook | None = None,
) -> RunnableTaskHandler:
    return ResearchTaskHandler(
        kind="worker",
        runnable=runnable,
        orchestrator=orchestrator,
        prompt_builder=prompt_builder,
        config_mapper=config_mapper,
        before_invoke=before_invoke,
        after_invoke=after_invoke,
    )


class ResearchTaskHandler(RunnableTaskHandler):
    def __init__(
        self,
        *,
        kind: ResearchKind,
        runnable: CompiledStateGraph[GraphInput, None, GraphInput, GraphInput],
        orchestrator: TaskOrchestrator,
        prompt_builder: PromptBuilder | None = None,
        config_mapper: ConfigMapper[RunnableConfig] | None = None,
        before_invoke: BeforeInvoke[GraphInput] | None = None,
        after_invoke: AfterInvokeHook | None = None,
    ) -> None:
        normalized_after_invoke = _normalize_after_invoke(after_invoke)
        super().__init__(
            ainvoke=runnable.ainvoke,
            input_mapper=lambda ctx: _build_input(ctx, kind, prompt_builder),
            output_mapper=lambda ctx, raw: _build_result(ctx, kind, raw, orchestrator),
            config_mapper=config_mapper or _default_agent_config,
            before_invoke=before_invoke or _default_before_invoke(kind, orchestrator.recorder),
            after_invoke=normalized_after_invoke,
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


def _default_before_invoke(
    kind: ResearchKind,
    recorder: WorkerLaunchRecorder,
) -> BeforeInvoke[GraphInput]:
    if kind == "worker":
        return _passthrough_input
    return lambda ctx, inp: _prepare_main_input(ctx, inp, recorder)


def _passthrough_input(_: TaskContext, inp: GraphInput) -> GraphInput:
    return inp


def _normalize_after_invoke(
    after_invoke: AfterInvokeHook | None,
) -> NormalizedAfterInvoke:
    if after_invoke is None:
        return _normalize_graph_output
    return lambda ctx, raw: after_invoke(ctx, _coerce_after_invoke_input(raw))


def _build_input(
    ctx: TaskContext,
    kind: ResearchKind,
    prompt_builder: PromptBuilder | None,
) -> GraphInput:
    payload_key = "topic" if kind == "main" else "query"
    prompt_text = str(ctx.task.payload[payload_key])
    prompt = _prompt_builder(kind, prompt_builder)(prompt_text)
    return _user_message(f"[speaker_type={_speaker_type(ctx)}]\n{prompt}")


def _prompt_builder(
    kind: ResearchKind,
    prompt_builder: PromptBuilder | None,
) -> PromptBuilder:
    if prompt_builder is not None:
        return prompt_builder
    if kind == "main":
        return _default_main_prompt
    return _default_worker_prompt


def _speaker_type(ctx: TaskContext) -> str:
    return str(ctx.task.metadata.get("speaker_type", "unknown"))


def _user_message(prompt: str) -> GraphInput:
    return GraphInput(messages=[{"role": "user", "content": prompt}])


def _prepare_main_input(
    ctx: TaskContext,
    inp: GraphInput,
    recorder: WorkerLaunchRecorder,
) -> GraphInput:
    recorder.drain()
    _record_scheduled_workers(ctx, recorder)
    return inp


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


def _build_result(
    ctx: TaskContext,
    kind: ResearchKind,
    raw: object,
    orchestrator: TaskOrchestrator,
) -> TaskResult:
    normalized = _coerce_graph_output(raw)
    if kind == "main":
        return orchestrator.build_main_result(ctx, normalized)
    return orchestrator.build_worker_result(ctx, normalized)


def _normalize_graph_output(_: TaskContext, raw: object) -> object:
    return _coerce_graph_output(raw)


def _coerce_after_invoke_input(raw: object) -> GraphInput:
    return _coerce_graph_output(raw)


def _coerce_graph_output(raw: object) -> GraphInput:
    if isinstance(raw, dict):
        messages = raw.get("messages")
        if isinstance(messages, list):
            return GraphInput(messages=messages)
    return GraphInput(messages=[{"role": "assistant", "content": str(raw).strip()}])
