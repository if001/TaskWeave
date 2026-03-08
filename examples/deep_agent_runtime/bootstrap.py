from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from runtime_core.models import Task, TaskContext, TaskResult
from runtime_core.registry import HandlerRegistry
from runtime_core.repository import InMemoryTaskRepository
from runtime_core.runtime import Runtime
from runtime_langchain.runnable_handler import RunnableTaskHandler

TASK_KIND_DEEP_RESEARCH = "deep_research"
EXAMPLE_TASK_ID = "example:deep:1"
DEFAULT_MODEL_NAME = "gpt-4o-mini"
_REAL_AGENT_ENV = "EXAMPLE_USE_REAL_DEEP_AGENT"
_MODEL_ENV = "EXAMPLE_MODEL"

AgentRequest = dict[str, object]
AgentConfig = dict[str, str | int]


class DeepAgentRunnable(Protocol):
    async def ainvoke(self, inp: AgentRequest | str, config: AgentConfig | None = None) -> object: ...


@dataclass(slots=True)
class ExampleRuntimeBundle:
    runtime: Runtime
    repository: InMemoryTaskRepository


class _EchoDeepAgentRunnable:
    """Fallback runnable for local execution without external dependencies."""

    async def ainvoke(self, inp: AgentRequest | str, config: AgentConfig | None = None) -> AgentRequest:
        request = _to_agent_request(inp)
        topic = str(request.get("topic", ""))
        return {
            "final_output": f"[mock deep-agent] researched: {topic}",
            "metadata": {"config": config or {}},
        }


class _LangChainDeepAgentRunnable:
    def __init__(self, model_name: str) -> None:
        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=model_name)
        self._agent = create_agent(
            model=model,
            tools=[],
            system_prompt="You are a concise deep-research assistant.",
        )

    async def ainvoke(self, inp: AgentRequest | str, config: AgentConfig | None = None) -> object:
        return await self._agent.ainvoke(_to_agent_request(inp), config=config)


def build_example_runtime() -> ExampleRuntimeBundle:
    repository = InMemoryTaskRepository()
    registry = HandlerRegistry()

    registry.register(
        TASK_KIND_DEEP_RESEARCH,
        RunnableTaskHandler(
            runnable=_build_deep_agent_runnable(),
            input_mapper=_build_agent_input,
            config_mapper=_build_agent_config,
            output_mapper=_build_task_result,
        ),
    )

    runtime = Runtime(repository=repository, registry=registry)
    return ExampleRuntimeBundle(runtime=runtime, repository=repository)


def seed_example_task(repository: InMemoryTaskRepository, topic: str) -> Task:
    task = Task(id=EXAMPLE_TASK_ID, kind=TASK_KIND_DEEP_RESEARCH, payload={"topic": topic})
    repository.enqueue(task)
    return task


def _build_deep_agent_runnable() -> DeepAgentRunnable:
    if _is_real_agent_enabled():
        return _LangChainDeepAgentRunnable(model_name=os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME))
    return _EchoDeepAgentRunnable()


def _is_real_agent_enabled() -> bool:
    return os.getenv(_REAL_AGENT_ENV, "0") == "1"


def _build_agent_input(ctx: TaskContext) -> AgentRequest:
    topic = str(ctx.task.payload["topic"])
    return {
        "messages": [{"role": "user", "content": f"Research this topic and summarize key points: {topic}"}],
        "topic": topic,
    }


def _build_agent_config(ctx: TaskContext) -> AgentConfig:
    return {"task_id": ctx.task.id, "attempt": ctx.attempt}


def _build_task_result(_ctx: TaskContext, raw: object) -> TaskResult:
    return TaskResult(status="succeeded", output={"agent_output": raw})


def _to_agent_request(inp: AgentRequest | str) -> AgentRequest:
    if isinstance(inp, dict):
        return inp
    return {"messages": [{"role": "user", "content": inp}]}
