from __future__ import annotations

from typing import TypeAlias, TypedDict, TypeGuard

from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool

from langgraph.graph.state import CompiledStateGraph, StateGraph, END

from runtime_core.types import (
    MainAgentOutput,
    MainAgentRawResult,
    Message,
    TaskContext,
    TaskResult,
    WorkerAgentOutput,
)
from runtime_core.tasks import (
    TaskResultConfig,
    WorkerLaunchRecorder,
    build_main_task_result,
    build_worker_task_result,
)
from .worker_tools import build_worker_request_tools


class GraphInput(TypedDict):
    messages: list[Message | AnyMessage]


MainAgentRunOutput: TypeAlias = MainAgentRawResult | MainAgentOutput | GraphInput
WorkerAgentRunOutput: TypeAlias = WorkerAgentOutput | GraphInput



class TaskOrchestrator:
    def __init__(
        self, config: TaskResultConfig, recorder: WorkerLaunchRecorder
    ) -> None:
        self._config = config
        self._recorder = recorder

    @property
    def recorder(self) -> WorkerLaunchRecorder:
        return self._recorder

    def worker_request_tools(self) -> list[BaseTool]:
        return build_worker_request_tools(self._recorder)

    def build_main_result(self, ctx: TaskContext, raw: MainAgentRunOutput) -> TaskResult:
        main_raw = _normalize_main_raw(raw, self._recorder)
        return build_main_task_result(ctx, main_raw, config=self._config)

    def build_worker_result(
        self, ctx: TaskContext, raw: WorkerAgentRunOutput
    ) -> TaskResult:
        worker_raw = _normalize_worker_output(raw)
        return build_worker_task_result(ctx, worker_raw, config=self._config)

    def mock_main_graph(self) -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
        def _echo(state: GraphInput) -> GraphInput:
            return state

        graph = StateGraph(GraphInput)
        graph.add_node("main", _echo)
        graph.set_entry_point("main")
        graph.add_edge("main", END)
        return graph.compile()


def _normalize_main_raw(
    raw: MainAgentRunOutput, recorder: WorkerLaunchRecorder
) -> MainAgentRawResult:
    if _is_main_agent_raw_result(raw):
        return raw
    drained = recorder.drain()
    drained["agent_output"] = _normalize_main_output(raw)
    return drained


def _normalize_main_output(raw: MainAgentRunOutput) -> MainAgentOutput:
    if _is_main_agent_raw_result(raw):
        return raw["agent_output"]
    return MainAgentOutput(
        final_output=_extract_output_text(
            raw,
        )
    )


def _normalize_worker_output(raw: WorkerAgentRunOutput) -> WorkerAgentOutput:
    return WorkerAgentOutput(final_output=_extract_output_text(raw))


def _extract_output_text(
    raw: MainAgentRawResult | GraphInput | MainAgentOutput | WorkerAgentOutput,
) -> str:
    if _is_main_agent_raw_result(raw):
        return _extract_output_text(raw["agent_output"])
    if isinstance(raw, dict):
        return _extract_message_output(raw.get("messages", []))
    return str(raw).strip()


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
    if _is_ai_message(message):
        content = getattr(message, "content", "")
        return str(content).strip()
    return ""


def _is_ai_message(message: Message | AnyMessage) -> bool:
    try:
        from langchain_core.messages import AIMessage
    except Exception:
        return False
    return isinstance(message, AIMessage)


def _is_main_agent_raw_result(
    raw: MainAgentRunOutput | GraphInput,
) -> TypeGuard[MainAgentRawResult]:
    if not isinstance(raw, dict):
        return False
    return {
        "agent_output",
        "immediate_queries",
        "delayed_queries",
        "periodic_queries",
    }.issubset(raw.keys())
