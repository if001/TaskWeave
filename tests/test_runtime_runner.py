import asyncio
import unittest

from runtime_core.models import Task, TaskContext, TaskResult
from runtime_core.registry import HandlerRegistry
from runtime_core.repository import InMemoryTaskRepository
from runtime_core.runtime import Runtime
from runtime_core.runner import RunnerPolicy, RuntimeRunner


class _RecordingHandler:
    def __init__(self, record: list[str]) -> None:
        self._record = record

    async def run(self, ctx: TaskContext) -> TaskResult:
        self._record.append(ctx.task.id)
        return TaskResult(status="succeeded")


class RuntimeRunnerTests(unittest.TestCase):
    def test_main_priority_with_single_slot(self) -> None:
        repository = InMemoryTaskRepository()
        registry = HandlerRegistry()
        record: list[str] = []
        registry.register("main", _RecordingHandler(record))
        registry.register("worker", _RecordingHandler(record))
        runtime = Runtime(repository, registry)
        runner = RuntimeRunner(
            runtime=runtime,
            policy=RunnerPolicy(
                max_concurrency=1,
                main_kinds=["main"],
                worker_kinds=["worker"],
            ),
        )
        repository.enqueue(Task(id="main:1", kind="main", payload={}))
        repository.enqueue(Task(id="worker:1", kind="worker", payload={}))

        asyncio.run(runner.run_once(now_unix=1.0))

        self.assertEqual(record, ["main:1"])
        self.assertEqual(_require_task(repository.get("main:1"), "main:1").status, "succeeded")
        self.assertEqual(_require_task(repository.get("worker:1"), "worker:1").status, "queued")

    def test_main_and_worker_run_with_two_slots(self) -> None:
        repository = InMemoryTaskRepository()
        registry = HandlerRegistry()
        record: list[str] = []
        registry.register("main", _RecordingHandler(record))
        registry.register("worker", _RecordingHandler(record))
        runtime = Runtime(repository, registry)
        runner = RuntimeRunner(
            runtime=runtime,
            policy=RunnerPolicy(
                max_concurrency=2,
                main_kinds=["main"],
                worker_kinds=["worker"],
            ),
        )
        repository.enqueue(Task(id="main:1", kind="main", payload={}))
        repository.enqueue(Task(id="worker:1", kind="worker", payload={}))

        asyncio.run(runner.run_once(now_unix=1.0))

        self.assertCountEqual(record, ["main:1", "worker:1"])
        self.assertEqual(_require_task(repository.get("main:1"), "main:1").status, "succeeded")
        self.assertEqual(_require_task(repository.get("worker:1"), "worker:1").status, "succeeded")


def _require_task(task: Task | None, task_id: str) -> Task:
    if task is None:
        raise AssertionError(f"task not found: {task_id}")
    return task


if __name__ == "__main__":
    unittest.main()
