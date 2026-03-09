from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, TypedDict

from examples.deep_agent_runtime.common import normalize_text

AgentRequest = dict[str, object]
AgentConfig = dict[str, str | int]


class DelayedWorkerPlan(TypedDict):
    query: str
    delay_seconds: float


class PeriodicWorkerPlan(TypedDict):
    query: str
    start_in_seconds: float
    interval_seconds: float
    repeat_count: int


class MainAgentRawResult(TypedDict):
    agent_output: object
    immediate_queries: list[str]
    delayed_queries: list[DelayedWorkerPlan]
    periodic_queries: list[PeriodicWorkerPlan]


class MainAgentRunnable(Protocol):
    async def ainvoke(self, inp: AgentRequest | str, config: AgentConfig | None = None) -> MainAgentRawResult: ...


@dataclass(slots=True)
class WorkerLaunchRecorder:
    immediate_queries: list[str] = field(default_factory=list)
    delayed_queries: list[DelayedWorkerPlan] = field(default_factory=list)
    periodic_queries: list[PeriodicWorkerPlan] = field(default_factory=list)

    def request_worker_now(self, query: str) -> str:
        normalized_query = normalize_text(query)
        if normalized_query:
            self.immediate_queries.append(normalized_query)
        return f"queued-worker-now:{normalized_query}"

    def request_worker_at(self, query: str, delay_seconds: float) -> str:
        normalized_query = normalize_text(query)
        if normalized_query:
            self.delayed_queries.append(DelayedWorkerPlan(query=normalized_query, delay_seconds=max(delay_seconds, 0.0)))
        return f"queued-worker-at:{normalized_query}:{delay_seconds}"

    def request_worker_periodic(self, query: str, start_in_seconds: float, interval_seconds: float, repeat_count: int) -> str:
        normalized_query = normalize_text(query)
        if normalized_query:
            self.periodic_queries.append(
                PeriodicWorkerPlan(
                    query=normalized_query,
                    start_in_seconds=max(start_in_seconds, 0.0),
                    interval_seconds=max(interval_seconds, 1.0),
                    repeat_count=max(repeat_count, 1),
                )
            )
        return f"queued-worker-periodic:{normalized_query}:{interval_seconds}"

    def drain(self) -> MainAgentRawResult:
        drained = MainAgentRawResult(
            agent_output={"message": "worker requests collected"},
            immediate_queries=list(self.immediate_queries),
            delayed_queries=list(self.delayed_queries),
            periodic_queries=list(self.periodic_queries),
        )
        self.immediate_queries.clear()
        self.delayed_queries.clear()
        self.periodic_queries.clear()
        return drained


class _MainRunnableBase(MainAgentRunnable):
    def __init__(self, recorder: WorkerLaunchRecorder) -> None:
        self._recorder = recorder

    def _collect_requests(self, request: AgentRequest) -> MainAgentRawResult:
        topic = str(request.get("topic", ""))
        needs_worker = bool(request.get("needs_worker", False))
        if needs_worker:
            self._recorder.request_worker_now(topic)

        for delayed in _to_delayed_plans(request.get("delayed_jobs", [])):
            self._recorder.request_worker_at(delayed["query"], delayed["delay_seconds"])

        for periodic in _to_periodic_plans(request.get("periodic_jobs", [])):
            self._recorder.request_worker_periodic(
                periodic["query"],
                periodic["start_in_seconds"],
                periodic["interval_seconds"],
                periodic["repeat_count"],
            )

        drained = self._recorder.drain()
        drained["agent_output"] = {
            "final_output": f"[mock main-agent] accepted: {topic}",
            "needs_worker": needs_worker,
            "delayed_count": len(drained["delayed_queries"]),
            "periodic_count": len(drained["periodic_queries"]),
        }
        return drained


class EchoMainAgentRunnable(_MainRunnableBase):
    async def ainvoke(self, inp: AgentRequest | str, config: AgentConfig | None = None) -> MainAgentRawResult:
        _ = config
        return self._collect_requests(_to_agent_request(inp))


class LangChainMainAgentRunnable(_MainRunnableBase):
    def __init__(self, model_name: str, recorder: WorkerLaunchRecorder) -> None:
        super().__init__(recorder)
        from langchain.agents import create_agent
        from langchain.tools import tool
        from langchain_openai import ChatOpenAI

        @tool("request_worker_now")
        def request_worker_now(query: str) -> str:
            return self._recorder.request_worker_now(query)

        @tool("request_worker_at")
        def request_worker_at(query: str, delay_seconds: float) -> str:
            return self._recorder.request_worker_at(query, delay_seconds)

        @tool("request_worker_periodic")
        def request_worker_periodic(query: str, start_in_seconds: float, interval_seconds: float, repeat_count: int) -> str:
            return self._recorder.request_worker_periodic(query, start_in_seconds, interval_seconds, repeat_count)

        self._agent = create_agent(
            model=ChatOpenAI(model=model_name),
            tools=[request_worker_now, request_worker_at, request_worker_periodic],
            system_prompt=(
                "You are a main research agent. "
                "Use worker tools for heavy deep-research tasks: immediate, delayed one-time, or periodic."
            ),
        )

    async def ainvoke(self, inp: AgentRequest | str, config: AgentConfig | None = None) -> MainAgentRawResult:
        self._recorder.drain()
        raw = await self._agent.ainvoke(_to_agent_request(inp), config=config)
        drained = self._recorder.drain()
        drained["agent_output"] = raw
        return drained


def build_main_agent_runnable(use_real_agent: bool, model_name: str, recorder: WorkerLaunchRecorder) -> MainAgentRunnable:
    if use_real_agent:
        return LangChainMainAgentRunnable(model_name=model_name, recorder=recorder)
    return EchoMainAgentRunnable(recorder=recorder)


def _to_agent_request(inp: AgentRequest | str) -> AgentRequest:
    if isinstance(inp, dict):
        return inp
    return {"messages": [{"role": "user", "content": inp}]}


def _to_delayed_plans(value: object) -> list[DelayedWorkerPlan]:
    plans: list[DelayedWorkerPlan] = []
    if not isinstance(value, list):
        return plans
    for item in value:
        if not isinstance(item, dict):
            continue
        query = normalize_text(item.get("query", ""))
        if not query:
            continue
        plans.append(DelayedWorkerPlan(query=query, delay_seconds=max(_to_float(item.get("delay_seconds"), 0.0), 0.0)))
    return plans


def _to_periodic_plans(value: object) -> list[PeriodicWorkerPlan]:
    plans: list[PeriodicWorkerPlan] = []
    if not isinstance(value, list):
        return plans
    for item in value:
        if not isinstance(item, dict):
            continue
        query = normalize_text(item.get("query", ""))
        if not query:
            continue
        plans.append(
            PeriodicWorkerPlan(
                query=query,
                start_in_seconds=max(_to_float(item.get("start_in_seconds"), 0.0), 0.0),
                interval_seconds=max(_to_float(item.get("interval_seconds"), 60.0), 1.0),
                repeat_count=max(_to_int(item.get("repeat_count"), 1), 1),
            )
        )
    return plans


def _to_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _to_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default
