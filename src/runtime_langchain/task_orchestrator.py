from __future__ import annotations

from langchain_core.tools import BaseTool

from runtime_core.agent_types import MainAgentInput, MainAgentOutput, MainAgentRawResult, WorkerAgentOutput
from runtime_core.models import TaskContext, TaskResult
from runtime_core.task_results import TaskResultConfig, build_main_task_result, build_worker_task_result
from runtime_core.worker_recorder import WorkerLaunchRecorder, collect_worker_requests
from runtime_langchain.worker_tools import build_worker_request_tools


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

    def build_main_result(self, ctx: TaskContext, raw: object) -> TaskResult:
        main_raw = _normalize_main_raw(raw, self._recorder)
        return build_main_task_result(ctx, main_raw, config=self._config)

    def build_worker_result(self, ctx: TaskContext, raw: object) -> TaskResult:
        worker_raw = _normalize_worker_output(raw)
        return build_worker_task_result(ctx, worker_raw, config=self._config)

    def mock_main_graph(self):
        recorder = self._recorder

        class _MockMainGraph:
            async def ainvoke(
                self,
                input: MainAgentInput,
                config: object | None = None,
            ) -> MainAgentRawResult:
                _ = config
                return collect_worker_requests(recorder, input)

        return _MockMainGraph()


def _normalize_main_raw(
    raw: object, recorder: WorkerLaunchRecorder
) -> MainAgentRawResult:
    if isinstance(raw, dict) and {
        "agent_output",
        "immediate_queries",
        "delayed_queries",
        "periodic_queries",
    }.issubset(raw.keys()):
        return raw  # type: ignore[return-value]
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
