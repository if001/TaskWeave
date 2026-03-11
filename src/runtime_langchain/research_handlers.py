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


def _default_main_prompt(topic: str) -> str:
    return (
        "Handle this user request. "
        "Use worker tools for heavy deep research work if needed. "
        f"topic={topic}"
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
        prompt_builder: Callable[[str], str] | None = None,
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
        prompt_builder: Callable[[str], str] | None = None,
        config_mapper: ConfigMapper[AgentConfig] | None = None,
        before_invoke: BeforeInvoke[MainAgentInput] | None = None,
        after_invoke: AfterInvoke[object] | None = None,
    ) -> None:
        prompt_builder = prompt_builder or _default_main_prompt

        def _input(ctx: TaskContext) -> MainAgentInput:
            topic = str(ctx.task.payload["topic"])
            prompt = prompt_builder(topic)

            return MainAgentInput(
                messages=[{"role": "user", "content": prompt}],
                topic=topic,
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
        if "final_output" in raw:
            return MainAgentOutput(
                final_output=str(raw.get("final_output", "")).strip(),
            )
        return MainAgentOutput(final_output=_extract_message_output(raw))
    return MainAgentOutput(final_output=str(raw).strip())


def _normalize_worker_output(raw: object) -> WorkerAgentOutput:
    if isinstance(raw, dict):
        if "final_output" in raw:
            return WorkerAgentOutput(final_output=str(raw.get("final_output", "")).strip())
        return WorkerAgentOutput(final_output=_extract_message_output(raw))
    return WorkerAgentOutput(final_output=str(raw).strip())


def _extract_message_output(raw: dict[str, object]) -> str:
    messages = raw.get("messages")
    if not isinstance(messages, list):
        return str(raw).strip()
    for message in reversed(messages):
        content = _extract_message_content(message)
        if content:
            return content
    return ""


def _extract_message_content(message: object) -> str:
    if isinstance(message, dict):
        role = str(message.get("role", "")).lower()
        if role and role not in {"assistant", "ai"}:
            return ""
        return str(message.get("content", "")).strip()
    if _is_ai_message(message):
        content = getattr(message, "content", "")
        return str(content).strip()
    return ""


def _is_ai_message(message: object) -> bool:
    try:
        from langchain_core.messages import AIMessage
    except Exception:
        return False
    return isinstance(message, AIMessage)
