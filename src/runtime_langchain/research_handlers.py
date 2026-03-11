from __future__ import annotations

from typing import Callable

from runtime_core.agent_types import AgentConfig, MainAgentInput, WorkerAgentInput
from runtime_core.task_plans import to_delayed_plans, to_periodic_plans
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
from runtime_langchain.task_orchestrator import TaskOrchestrator


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
        runnable: CompiledStateGraphLike,
        orchestrator: TaskOrchestrator,
        prompt_builder: Callable[[str], str] | None = None,
        config_mapper: ConfigMapper[AgentConfig] | None = None,
        before_invoke: BeforeInvoke[MainAgentInput] | None = None,
        after_invoke: AfterInvoke[object] | None = None,
    ) -> "MainResearchTaskHandler":
        return cls(
            runnable=wrap_compiled_state_graph(runnable),
            orchestrator=orchestrator,
            prompt_builder=prompt_builder,
            config_mapper=config_mapper,
            before_invoke=before_invoke,
            after_invoke=after_invoke,
        )

    def __init__(
        self,
        runnable: AsyncRunnable[MainAgentInput, object, AgentConfig],
        orchestrator: TaskOrchestrator,
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
            orchestrator.recorder.drain()
            return inp

        def _output(ctx: TaskContext, raw: object) -> TaskResult:
            _ = ctx
            return orchestrator.build_main_result(ctx, raw)

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


class WorkerResearchTaskHandler(RunnableTaskHandler):
    @classmethod
    def for_langchain(
        cls,
        runnable: CompiledStateGraphLike,
        orchestrator: TaskOrchestrator,
        prompt_builder: Callable[[str], str] | None = None,
        config_mapper: ConfigMapper[AgentConfig] | None = None,
        before_invoke: BeforeInvoke[WorkerAgentInput] | None = None,
        after_invoke: AfterInvoke[object] | None = None,
    ) -> "WorkerResearchTaskHandler":
        return cls(
            runnable=wrap_compiled_state_graph(runnable),
            orchestrator=orchestrator,
            prompt_builder=prompt_builder,
            config_mapper=config_mapper,
            before_invoke=before_invoke,
            after_invoke=after_invoke,
        )

    def __init__(
        self,
        runnable: AsyncRunnable[WorkerAgentInput, object, AgentConfig],
        orchestrator: TaskOrchestrator,
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
            return orchestrator.build_worker_result(ctx, raw)

        super().__init__(
            runnable=runnable,
            input_mapper=_input,
            output_mapper=_output,
            config_mapper=config_mapper or _default_agent_config,
            before_invoke=before_invoke,
            after_invoke=after_invoke,
        )


 
