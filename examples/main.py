from __future__ import annotations

import asyncio
from time import time

from runtime_core.models import Task
from runtime_core.repository import InMemoryTaskRepository
from runtime_core.runtime import Runtime

from examples.deep_agent_runtime.bootstrap import TASK_KIND_MAIN_RESEARCH, build_example_runtime

_EXIT_COMMANDS = {"exit", "quit", ":q"}
_WORKER_TRIGGER_KEYWORDS = ("research", "deep", "investigate", "調査", "深掘り")
_USER_ID = "terminal-user-1"


async def run() -> None:
    bundle = build_example_runtime()
    turn = 1

    print("Deep Agent Runtime Chat (type 'exit' to quit)")
    while True:
        user_text = input("you> ").strip()
        if not user_text:
            continue
        if user_text.lower() in _EXIT_COMMANDS:
            print("bye")
            break

        current_unix = time()
        task_id = f"chat:main:{turn}"
        plan = _build_worker_plan(user_text)
        bundle.repository.enqueue(
            Task(
                id=task_id,
                kind=TASK_KIND_MAIN_RESEARCH,
                payload={
                    "topic": user_text,
                    "needs_worker": plan["needs_worker"],
                    "delayed_jobs": plan["delayed_jobs"],
                    "periodic_jobs": plan["periodic_jobs"],
                },
                metadata={"user_id": _USER_ID, "turn": turn, "enqueued_at_unix": current_unix},
            )
        )

        await _run_until_idle(bundle.runtime)
        _print_turn_result(bundle.repository, task_id=task_id)
        turn += 1


def main() -> None:
    asyncio.run(run())


async def _run_until_idle(runtime: Runtime) -> None:
    while await runtime.tick():
        pass


def _build_worker_plan(user_text: str) -> dict[str, object]:
    needs_worker = _should_launch_worker(user_text)
    delayed_jobs: list[dict[str, object]] = []
    periodic_jobs: list[dict[str, object]] = []

    lowered = user_text.lower()
    if "later" in lowered:
        delayed_jobs.append({"query": user_text, "delay_seconds": 10.0})
    if "daily" in lowered or "periodic" in lowered:
        periodic_jobs.append(
            {
                "query": user_text,
                "start_in_seconds": 5.0,
                "interval_seconds": 60.0,
                "repeat_count": 3,
            }
        )

    return {
        "needs_worker": needs_worker,
        "delayed_jobs": delayed_jobs,
        "periodic_jobs": periodic_jobs,
    }


def _should_launch_worker(user_text: str) -> bool:
    normalized = user_text.lower()
    return any(keyword in normalized for keyword in _WORKER_TRIGGER_KEYWORDS)


def _print_turn_result(repository: InMemoryTaskRepository, task_id: str) -> None:
    main_task = _require_task(task=repository.get(task_id), task_id=task_id)
    print(f"agent> main_task status={main_task.status}")

    worker_ids = [
        f"worker:{task_id}:now:1",
        f"worker:{task_id}:delayed:1",
        f"worker:{task_id}:periodic:1:1",
    ]
    for worker_id in worker_ids:
        worker_task = repository.get(worker_id)
        if worker_task is None:
            continue
        print(
            "agent> worker_task "
            f"id={worker_task.id} status={worker_task.status} run_after={worker_task.run_after}"
        )


def _require_task(task: Task | None, task_id: str) -> Task:
    if task is None:
        raise RuntimeError(f"task not found: {task_id}")
    return task


if __name__ == "__main__":
    main()
