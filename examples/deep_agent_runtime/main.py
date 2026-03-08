from __future__ import annotations

import asyncio
from runtime_core.models import Task
from runtime_core.runtime import Runtime

from examples.deep_agent_runtime.bootstrap import EXAMPLE_TASK_ID, build_example_runtime, seed_example_task

_DEFAULT_TOPIC = "LLM runtime architecture patterns"


async def run() -> None:
    bundle = build_example_runtime()
    seed_example_task(bundle.repository, topic=_DEFAULT_TOPIC)

    await _run_until_idle(bundle.runtime)
    task = _require_task_status(bundle.repository.get(EXAMPLE_TASK_ID), task_id=EXAMPLE_TASK_ID)

    print(f"status={task}")
    print(bundle.repository.transitions)


def main() -> None:
    asyncio.run(run())


async def _run_until_idle(runtime: Runtime) -> None:
    while await runtime.tick():
        pass


def _require_task_status(task: Task | None, task_id: str) -> str:
    if task is None:
        raise RuntimeError(f"task not found: {task_id}")
    return task.status


if __name__ == "__main__":
    main()
