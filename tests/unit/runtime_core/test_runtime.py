import asyncio
import unittest

from runtime_core.runtime import HandlerRegistry, InMemoryTaskRepository, Runtime
from runtime_core.types import Task, TaskContext, TaskResult


class _FailingHandler:
    async def run(self, ctx: TaskContext) -> TaskResult:
        _ = ctx
        raise RuntimeError("boom")


class RuntimeTests(unittest.TestCase):
    def test_execute_task_marks_failed_on_exception(self) -> None:
        repository = InMemoryTaskRepository()
        registry = HandlerRegistry()
        registry.register("boom", _FailingHandler())
        runtime = Runtime(repository, registry)
        repository.enqueue(Task(id="task:boom", kind="boom", payload={}))

        task = repository.lease_next_ready(now_unix=0.0)
        assert task is not None
        result = asyncio.run(runtime.execute_task(task, now_unix=0.0))

        self.assertEqual(result.status, "failed")
        stored = repository.get("task:boom")
        assert stored is not None
        self.assertEqual(stored.status, "failed")


if __name__ == "__main__":
    unittest.main()
