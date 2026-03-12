from __future__ import annotations

import asyncio
from time import time
import uuid
from runtime_core.types import Task
from runtime_core.runtime import RunnerPolicy, RuntimeRunner, TaskRepository

from examples.deep_agent_runtime.bootstrap import (
    TASK_KIND_MAIN_RESEARCH,
    TASK_KIND_NOTIFICATION,
    TASK_KIND_WORKER_RESEARCH,
    build_example_runtime,
)

_EXIT_COMMANDS = {"exit", "quit", ":q"}
_USER_ID = "terminal-user-1"


async def run() -> None:
    async with build_example_runtime() as bundle:
        runner = RuntimeRunner(
            runtime=bundle.runtime,
            policy=RunnerPolicy(
                max_concurrency=2,
                main_kinds=[TASK_KIND_MAIN_RESEARCH],
                worker_kinds=[TASK_KIND_WORKER_RESEARCH, TASK_KIND_NOTIFICATION],
            ),
        )
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
            task_id = f"chat:main:{turn}_{uuid.uuid4()}"
            bundle.repository.enqueue(
                Task(
                    id=task_id,
                    kind=TASK_KIND_MAIN_RESEARCH,
                    payload={
                        "topic": user_text,
                        "delayed_jobs": [],
                        "periodic_jobs": [],
                    },
                    metadata={
                        "user_id": _USER_ID,
                        "turn": turn,
                        "enqueued_at_unix": current_unix,
                    },
                )
            )

            await _run_until_idle(runner)
            _print_turn_result(bundle.repository, task_id=task_id)
            turn += 1


def main() -> None:
    asyncio.run(run())


async def _run_until_idle(runner: RuntimeRunner) -> None:
    while await runner.run_once():
        pass


def _print_turn_result(repository: TaskRepository, task_id: str) -> None:
    main_task = _require_task(task=repository.get(task_id), task_id=task_id)
    print(f"agent> main_task status={main_task.status}")
    _print_notification(repository, f"notification:{task_id}:main")

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
        _print_notification(repository, f"notification:{worker_id}:worker_done")


def _print_notification(repository: TaskRepository, notification_id: str) -> None:
    notification = repository.get(notification_id)
    if notification is None:
        return
    print("notification.payload:", notification.payload)
    message = str(notification.payload.get("message", "")).strip()
    if message:
        print(f"agent> notification {message}")


def _require_task(task: Task | None, task_id: str) -> Task:
    if task is None:
        raise RuntimeError(f"task not found: {task_id}")
    return task


if __name__ == "__main__":
    main()
