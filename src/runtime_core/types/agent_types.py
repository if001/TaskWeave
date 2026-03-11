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
    messages: list[Message]
    topic: str
    delayed_jobs: list[DelayedWorkerPlan]
    periodic_jobs: list[PeriodicWorkerPlan]


class WorkerAgentInput(TypedDict):
    messages: list[Message]
    query: str




def normalize_main_input(inp: MainAgentInput | str) -> MainAgentInput:
    if isinstance(inp, dict):
        return MainAgentInput(
            messages=inp.get("messages", []),
            topic=str(inp.get("topic", "")).strip(),
            delayed_jobs=inp.get("delayed_jobs", []),
            periodic_jobs=inp.get("periodic_jobs", []),
        )
    return MainAgentInput(
        messages=[{"role": "user", "content": inp}],
        topic="",
        delayed_jobs=[],
        periodic_jobs=[],
    )


def normalize_worker_input(inp: WorkerAgentInput | str) -> WorkerAgentInput:
    if isinstance(inp, dict):
        return WorkerAgentInput(
            messages=inp.get("messages", []),
            query=str(inp.get("query", "")).strip(),
        )
    return WorkerAgentInput(
        messages=[{"role": "user", "content": inp}],
        query=str(inp).strip(),
    )
