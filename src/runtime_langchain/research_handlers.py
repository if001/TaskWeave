from __future__ import annotations

from typing import Callable, cast

from runtime_core.agent_types import (
    AgentConfig,
    MainAgentInput,
    MainAgentOutput,
    MainAgentRawResult,
    WorkerAgentInput,
    WorkerAgentOutput,
)
from runtime_core.task_flow import (
    TaskFlowConfig,
    build_main_task_result,
    build_worker_task_result,
    to_delayed_plans,
    to_periodic_plans,
)
from runtime_core.models import TaskContext, TaskResult
from runtime_langchain.runnable_handler import (
    AfterInvoke,
    AsyncRunnable,
    BeforeInvoke,
    ConfigMapper,
    RunnableTaskHandler,
    CompiledStateGraphLike,
    wrap_compiled_state_graph,
)
from runtime_langchain.worker_tools import WorkerLaunchRecorder, collect_worker_requests


def _default_main_prompt(topic: str, needs_worker: bool) -> str:
    return (
        "Handle this user request. "
        "Use worker tools for heavy deep research work if needed. "
        f"topic={topic}, needs_worker={needs_worker}"
    )


def _default_worker_prompt(query: str) -> str:
    return f"Perform deep research and summarize: {query}"


def _default_agent_config(ctx: TaskContext) -> AgentConfig:
    return {"task_id": ctx.task.id, "attempt": ctx.attempt}


class MainResearchTaskHandler(RunnableTaskHandler):
    @classmethod
    def for_langchain(
        cls,
        runnable_factory: Callable[[WorkerLaunchRecorder], CompiledStateGraphLike],
        flow: TaskFlowConfig,
        prompt_builder: Callable[[str, bool], str] | None = None,
        config_mapper: ConfigMapper[AgentConfig] | None = None,
        before_invoke: BeforeInvoke[MainAgentInput] | None = None,
        after_invoke: AfterInvoke[object] | None = None,
    ) -> "MainResearchTaskHandler":
        recorder = WorkerLaunchRecorder()
        runnable = wrap_compiled_state_graph(runnable_factory(recorder))
        return cls(
            runnable=runnable,
            flow=flow,
            recorder=recorder,
            prompt_builder=prompt_builder,
            config_mapper=config_mapper,
            before_invoke=before_invoke,
            after_invoke=after_invoke,
        )

    def __init__(
        self,
        runnable: AsyncRunnable[MainAgentInput, object, AgentConfig],
        flow: TaskFlowConfig,
        recorder: WorkerLaunchRecorder,
        prompt_builder: Callable[[str, bool], str] | None = None,
        config_mapper: ConfigMapper[AgentConfig] | None = None,
        before_invoke: BeforeInvoke[MainAgentInput] | None = None,
        after_invoke: AfterInvoke[object] | None = None,
    ) -> None:
        prompt_builder = prompt_builder or _default_main_prompt

        def _input(ctx: TaskContext) -> MainAgentInput:
            topic = str(ctx.task.payload["topic"])
            needs_worker = bool(ctx.task.payload.get("needs_worker", False))
            prompt = prompt_builder(topic, needs_worker)

            return MainAgentInput(
                messages=[{"role": "user", "content": prompt}],
                topic=topic,
                needs_worker=needs_worker,
                delayed_jobs=to_delayed_plans(ctx.task.payload.get("delayed_jobs", [])),
                periodic_jobs=to_periodic_plans(
                    ctx.task.payload.get("periodic_jobs", [])
                ),
            )

        def _before(ctx: TaskContext, inp: MainAgentInput) -> MainAgentInput:
            _ = ctx
            recorder.drain()
            return inp

        def _output(ctx: TaskContext, raw: object) -> TaskResult:
            _ = ctx
            main_raw = _normalize_main_raw(raw, recorder)
            return build_main_task_result(ctx, main_raw, config=flow)

        def _after(ctx: TaskContext, raw: object) -> object:
            if after_invoke is None:
                return raw
            return after_invoke(ctx, raw)

        super().__init__(
            runnable=runnable,
            input_mapper=_input,
            output_mapper=_output,
            config_mapper=config_mapper or _default_agent_config,
            before_invoke=before_invoke or _before,
            after_invoke=_after,
        )


def build_mock_main_graph(
    recorder: WorkerLaunchRecorder,
) -> CompiledStateGraphLike:
    class _MockMainGraph:
        async def ainvoke(
            self,
            input: MainAgentInput,
            config: object | None = None,
        ) -> MainAgentRawResult:
            _ = config
            return collect_worker_requests(recorder, input)

    return _MockMainGraph()


class WorkerResearchTaskHandler(RunnableTaskHandler):
    @classmethod
    def for_langchain(
        cls,
        runnable_factory: Callable[[], CompiledStateGraphLike],
        flow: TaskFlowConfig,
        prompt_builder: Callable[[str], str] | None = None,
        config_mapper: ConfigMapper[AgentConfig] | None = None,
        before_invoke: BeforeInvoke[WorkerAgentInput] | None = None,
        after_invoke: AfterInvoke[object] | None = None,
    ) -> "WorkerResearchTaskHandler":
        runnable = wrap_compiled_state_graph(runnable_factory())
        return cls(
            runnable=runnable,
            flow=flow,
            prompt_builder=prompt_builder,
            config_mapper=config_mapper,
            before_invoke=before_invoke,
            after_invoke=after_invoke,
        )

    def __init__(
        self,
        runnable: AsyncRunnable[WorkerAgentInput, object, AgentConfig],
        flow: TaskFlowConfig,
        prompt_builder: Callable[[str], str] | None = None,
        config_mapper: ConfigMapper[AgentConfig] | None = None,
        before_invoke: BeforeInvoke[WorkerAgentInput] | None = None,
        after_invoke: AfterInvoke[object] | None = None,
    ) -> None:
        prompt_builder = prompt_builder or _default_worker_prompt

        def _input(ctx: TaskContext) -> WorkerAgentInput:
            query = str(ctx.task.payload["query"])
            return WorkerAgentInput(
                messages=[{"role": "user", "content": prompt_builder(query)}],
                query=query,
            )

        def _output(ctx: TaskContext, raw: object) -> TaskResult:
            _ = ctx
            worker_raw = _normalize_worker_output(raw)
            return build_worker_task_result(ctx, worker_raw, config=flow)

        super().__init__(
            runnable=runnable,
            input_mapper=_input,
            output_mapper=_output,
            config_mapper=config_mapper or _default_agent_config,
            before_invoke=before_invoke,
            after_invoke=after_invoke,
        )


def _normalize_main_raw(
    raw: object, recorder: WorkerLaunchRecorder
) -> MainAgentRawResult:
    if isinstance(raw, dict) and {
        "agent_output",
        "immediate_queries",
        "delayed_queries",
        "periodic_queries",
    }.issubset(raw.keys()):
        return cast(MainAgentRawResult, raw)
    drained = recorder.drain()
    drained["agent_output"] = _normalize_main_output(raw)
    return drained


def _normalize_main_output(raw: object) -> MainAgentOutput:
    if isinstance(raw, dict):
        final_output = str(raw.get("final_output", "")).strip()
        needs_worker = bool(raw.get("needs_worker", False))
        try:
            delayed_count = int(raw.get("delayed_count", 0))
        except (TypeError, ValueError):
            delayed_count = 0
        try:
            periodic_count = int(raw.get("periodic_count", 0))
        except (TypeError, ValueError):
            periodic_count = 0
        return MainAgentOutput(
            final_output=final_output,
            needs_worker=needs_worker,
            delayed_count=delayed_count,
            periodic_count=periodic_count,
        )
    return MainAgentOutput(
        final_output=str(raw).strip(),
        needs_worker=False,
        delayed_count=0,
        periodic_count=0,
    )


def _normalize_worker_output(raw: object) -> WorkerAgentOutput:
    if isinstance(raw, dict):
        final_output = str(raw.get("final_output", "")).strip()
        return WorkerAgentOutput(final_output=final_output)
    return WorkerAgentOutput(final_output=str(raw).strip())
