from __future__ import annotations

from typing import TypeAlias, TypedDict, TypeGuard

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.tools import BaseTool

from runtime_core.tasks import (
    TaskResultConfig,
    WorkerLaunchRecorder,
    build_main_task_result,
    build_worker_task_result,
)
from runtime_core.types import (
    MainAgentOutput,
    MainAgentRawResult,
    Message,
    TaskContext,
    TaskResult,
    WorkerAgentOutput,
)

from .worker_tools import build_worker_request_tools


class GraphInput(TypedDict):
    messages: list[Message | AnyMessage]


MainAgentRunOutput: TypeAlias = MainAgentRawResult | MainAgentOutput | GraphInput
WorkerAgentRunOutput: TypeAlias = WorkerAgentOutput | GraphInput


class TaskOrchestrator:
    def __init__(self, config: TaskResultConfig, recorder: WorkerLaunchRecorder) -> None:
        self._config = config
        self._recorder = recorder

    @property
    def recorder(self) -> WorkerLaunchRecorder:
        return self._recorder

    def worker_request_tools(self) -> list[BaseTool]:
        return build_worker_request_tools(self._recorder)

    def build_main_result(self, ctx: TaskContext, raw: MainAgentRunOutput) -> TaskResult:
        main_raw = _normalize_main_result(raw, self._recorder)
        return build_main_task_result(ctx, main_raw, config=self._config)

    def build_worker_result(self, ctx: TaskContext, raw: WorkerAgentRunOutput) -> TaskResult:
        worker_output = WorkerAgentOutput(final_output=_extract_output_text(raw))
        return build_worker_task_result(ctx, worker_output, config=self._config)


def _normalize_main_result(
    raw: MainAgentRunOutput,
    recorder: WorkerLaunchRecorder,
) -> MainAgentRawResult:
    if _is_main_agent_raw_result(raw):
        return raw
    recorded = recorder.drain()
    recorded["agent_output"] = MainAgentOutput(final_output=_extract_output_text(raw))
    return recorded


def _extract_output_text(
    raw: MainAgentRawResult | MainAgentOutput | WorkerAgentOutput | GraphInput,
) -> str:
    if _is_main_agent_raw_result(raw):
        return _extract_output_text(raw["agent_output"])
    if _is_graph_input(raw):
        return _extract_message_output(raw["messages"])
    return str(raw.get("final_output", "")).strip()


def _is_graph_input(raw: object) -> TypeGuard[GraphInput]:
    return isinstance(raw, dict) and isinstance(raw.get("messages"), list)


def _extract_message_output(messages: list[Message | AnyMessage]) -> str:
    for message in reversed(messages):
        content = _extract_message_content(message)
        if content:
            return content
    return ""


def _extract_message_content(message: Message | AnyMessage) -> str:
    if isinstance(message, dict):
        role = str(message.get("role", "")).lower()
        if role and role not in {"assistant", "ai"}:
            return ""
        return str(message.get("content", "")).strip()
    if isinstance(message, AIMessage):
        return str(message.content).strip()
    return ""


def _is_main_agent_raw_result(raw: MainAgentRunOutput | GraphInput) -> TypeGuard[MainAgentRawResult]:
    if not isinstance(raw, dict):
        return False
    return {
        "agent_output",
        "immediate_queries",
        "delayed_queries",
        "periodic_queries",
    }.issubset(raw.keys())
