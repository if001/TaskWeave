from typing import TypedDict
AgentConfig = dict[str, str | int]


class Message(TypedDict):
    role: str
    content: str


class MainAgentRawResult(TypedDict):
    agent_output: "MainAgentOutput"
    immediate_queries: list[str]
    delayed_queries: list["DelayedWorkerPlan"]
    periodic_queries: list["PeriodicWorkerPlan"]


class MainAgentOutput(TypedDict):
    final_output: str


class WorkerAgentOutput(TypedDict):
    final_output: str


class DelayedWorkerPlan(TypedDict):
    query: str
    delay_seconds: float


class PeriodicWorkerPlan(TypedDict):
    query: str
    start_in_seconds: float
    interval_seconds: float
    repeat_count: int


class MainAgentInput(TypedDict):
    topic: str
    delayed_jobs: list[DelayedWorkerPlan]
    periodic_jobs: list[PeriodicWorkerPlan]


class WorkerAgentInput(TypedDict):
    query: str
