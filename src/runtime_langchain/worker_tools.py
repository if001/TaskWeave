from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.tools import BaseTool, tool

from runtime_core.agent_types import (
    DelayedWorkerPlan,
    MainAgentInput,
    MainAgentOutput,
    MainAgentRawResult,
    PeriodicWorkerPlan,
    normalize_main_input,
)
from runtime_core.research_flow import to_delayed_plans, to_periodic_plans


@dataclass(slots=True)
class WorkerLaunchRecorder:
    immediate_queries: list[str] = field(default_factory=list)
    delayed_queries: list[DelayedWorkerPlan] = field(default_factory=list)
    periodic_queries: list[PeriodicWorkerPlan] = field(default_factory=list)

    def request_worker_now(self, query: str) -> str:
        normalized = str(query).strip()
        if normalized:
            self.immediate_queries.append(normalized)
        return f"queued-worker-now:{normalized}"

    def request_worker_at(self, query: str, delay_seconds: float) -> str:
        normalized = str(query).strip()
        if normalized:
            self.delayed_queries.append(
                DelayedWorkerPlan(
                    query=normalized,
                    delay_seconds=max(float(delay_seconds), 0.0),
                )
            )
        return f"queued-worker-at:{normalized}:{delay_seconds}"

    def request_worker_periodic(
        self,
        query: str,
        start_in_seconds: float,
        interval_seconds: float,
        repeat_count: int,
    ) -> str:
        normalized = str(query).strip()
        if normalized:
            self.periodic_queries.append(
                PeriodicWorkerPlan(
                    query=normalized,
                    start_in_seconds=max(float(start_in_seconds), 0.0),
                    interval_seconds=max(float(interval_seconds), 1.0),
                    repeat_count=max(int(repeat_count), 1),
                )
            )
        return f"queued-worker-periodic:{normalized}:{interval_seconds}"

    def drain(self) -> MainAgentRawResult:
        drained = MainAgentRawResult(
            agent_output=MainAgentOutput(
                final_output="worker requests collected",
                needs_worker=False,
                delayed_count=0,
                periodic_count=0,
            ),
            immediate_queries=list(self.immediate_queries),
            delayed_queries=list(self.delayed_queries),
            periodic_queries=list(self.periodic_queries),
        )
        self.immediate_queries.clear()
        self.delayed_queries.clear()
        self.periodic_queries.clear()
        return drained


def collect_worker_requests(
    recorder: WorkerLaunchRecorder, request: MainAgentInput
) -> MainAgentRawResult:
    normalized = normalize_main_input(request)
    topic = normalized["topic"]
    needs_worker = normalized["needs_worker"]
    if needs_worker:
        recorder.request_worker_now(topic)

    for delayed in to_delayed_plans(normalized["delayed_jobs"]):
        recorder.request_worker_at(delayed["query"], delayed["delay_seconds"])

    for periodic in to_periodic_plans(normalized["periodic_jobs"]):
        recorder.request_worker_periodic(
            periodic["query"],
            periodic["start_in_seconds"],
            periodic["interval_seconds"],
            periodic["repeat_count"],
        )

    drained = recorder.drain()
    drained["agent_output"] = MainAgentOutput(
        final_output=f"[mock main-agent] accepted: {topic}",
        needs_worker=needs_worker,
        delayed_count=len(drained["delayed_queries"]),
        periodic_count=len(drained["periodic_queries"]),
    )
    return drained


def build_worker_request_tools(recorder: WorkerLaunchRecorder) -> list[BaseTool]:
    @tool("request_worker_now")
    def request_worker_now(query: str) -> str:
        """Queue an immediate deep-research worker task.

        Use when the query needs background research right away.
        Args:
            query: Research topic or question to hand off to the worker.
        Returns:
            A status string indicating the request was queued.
        Side effects:
            Records the request in the worker launch recorder.
        """
        return recorder.request_worker_now(query)

    @tool("request_worker_at")
    def request_worker_at(query: str, delay_seconds: float) -> str:
        """Queue a one-time worker task after a delay (seconds).

        Use when the work should start later (e.g., cooldown or scheduled check).
        Args:
            query: Research topic or question for the worker.
            delay_seconds: Seconds from now to start the task (will be clamped to >= 0).
        Returns:
            A status string indicating the delayed request was queued.
        Side effects:
            Records the request in the worker launch recorder.
        """
        return recorder.request_worker_at(query, delay_seconds)

    @tool("request_worker_periodic")
    def request_worker_periodic(
        query: str,
        start_in_seconds: float,
        interval_seconds: float,
        repeat_count: int,
    ) -> str:
        """Queue a periodic worker task with a start delay and repeat interval.

        Use for recurring research (e.g., monitoring a topic).
        Args:
            query: Research topic or question for the worker.
            start_in_seconds: Seconds from now to the first run (clamped to >= 0).
            interval_seconds: Seconds between runs (clamped to >= 1).
            repeat_count: Number of runs (clamped to >= 1).
        Returns:
            A status string indicating the periodic request was queued.
        Side effects:
            Records the request in the worker launch recorder.
        """
        return recorder.request_worker_periodic(
            query, start_in_seconds, interval_seconds, repeat_count
        )

    return [request_worker_now, request_worker_at, request_worker_periodic]
