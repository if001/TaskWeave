from __future__ import annotations

from dataclasses import dataclass, field

from ..types import (
    DelayedWorkerPlan,
    JsonValue,
    MainAgentInput,
    MainAgentOutput,
    MainAgentRawResult,
    PeriodicWorkerPlan,
)
from .task_plans import to_delayed_plans, to_periodic_plans


def _empty_str_list() -> list[str]:
    return []


def _empty_delayed_list() -> list[DelayedWorkerPlan]:
    return []


def _empty_periodic_list() -> list[PeriodicWorkerPlan]:
    return []


@dataclass(slots=True)
class WorkerLaunchRecorder:
    immediate_queries: list[str] = field(default_factory=_empty_str_list)
    delayed_queries: list[DelayedWorkerPlan] = field(default_factory=_empty_delayed_list)
    periodic_queries: list[PeriodicWorkerPlan] = field(default_factory=_empty_periodic_list)

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
    topic = str(request.get("topic", "")).strip()

    delayed_jobs: list[JsonValue] = [
        {
            "query": item["query"],
            "delay_seconds": item["delay_seconds"],
        }
        for item in request.get("delayed_jobs", [])
    ]
    for delayed in to_delayed_plans(delayed_jobs):
        recorder.request_worker_at(delayed["query"], delayed["delay_seconds"])

    periodic_jobs: list[JsonValue] = [
        {
            "query": item["query"],
            "start_in_seconds": item["start_in_seconds"],
            "interval_seconds": item["interval_seconds"],
            "repeat_count": item["repeat_count"],
        }
        for item in request.get("periodic_jobs", [])
    ]
    for periodic in to_periodic_plans(periodic_jobs):
        recorder.request_worker_periodic(
            periodic["query"],
            periodic["start_in_seconds"],
            periodic["interval_seconds"],
            periodic["repeat_count"],
        )

    drained = recorder.drain()
    drained["agent_output"] = MainAgentOutput(
        final_output=f"[mock main-agent] accepted: {topic}",
    )
    return drained
