import asyncio
import unittest

from runtime_core.runtime import HandlerRegistry, InMemoryTaskRepository, Runtime
from runtime_core.types import Task, TaskContext, TaskResult


class _FailingHandler:
    async def run(self, ctx: TaskContext) -> TaskResult:
        _ = ctx
        raise RuntimeError("boom")




class _SuccessHandler:
    def __init__(self, next_tasks: list[Task] | None = None) -> None:
        self._next_tasks = next_tasks or []

    async def run(self, ctx: TaskContext) -> TaskResult:
        _ = ctx
        return TaskResult(status="succeeded", next_tasks=list(self._next_tasks))


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

    def test_list_tasks_filters_by_status(self) -> None:
        repository = InMemoryTaskRepository()
        registry = HandlerRegistry()
        runtime = Runtime(repository, registry)
        repository.enqueue(Task(id="task:queued", kind="worker", payload={}))
        repository.enqueue(Task(id="task:cancelled", kind="worker", payload={}))
        repository.mark_status("task:cancelled", "cancelled")

        tasks = runtime.list_tasks(statuses=["queued"])

        self.assertEqual([task.id for task in tasks], ["task:queued"])

    def test_cancel_task_marks_queued_task_cancelled(self) -> None:
        repository = InMemoryTaskRepository()
        registry = HandlerRegistry()
        runtime = Runtime(repository, registry)
        repository.enqueue(Task(id="task:queued", kind="worker", payload={}))

        cancelled = runtime.cancel_task("task:queued")

        self.assertTrue(cancelled)
        stored = repository.get("task:queued")
        assert stored is not None
        self.assertEqual(stored.status, "cancelled")
        self.assertTrue(bool(stored.metadata.get("cancellation_requested")))

    def test_cancel_tasks_by_periodic_root_cancels_matching_tasks(self) -> None:
        repository = InMemoryTaskRepository()
        registry = HandlerRegistry()
        runtime = Runtime(repository, registry)
        repository.enqueue(
            Task(
                id="task:periodic:1",
                kind="worker",
                payload={"periodic_root_id": "root:1"},
            )
        )
        repository.enqueue(
            Task(
                id="task:periodic:2",
                kind="worker",
                payload={"periodic_root_id": "root:1"},
            )
        )
        repository.enqueue(
            Task(
                id="task:periodic:3",
                kind="worker",
                payload={"periodic_root_id": "root:2"},
            )
        )

        cancelled = runtime.cancel_tasks_by_periodic_root("root:1")

        self.assertEqual(cancelled, ["task:periodic:1", "task:periodic:2"])
        task1 = repository.get("task:periodic:1")
        task2 = repository.get("task:periodic:2")
        task3 = repository.get("task:periodic:3")
        assert task1 is not None
        assert task2 is not None
        assert task3 is not None
        self.assertEqual(task1.status, "cancelled")
        self.assertEqual(task2.status, "cancelled")
        self.assertEqual(task3.status, "queued")

    def test_execute_task_replaces_pending_task_with_same_dedupe_key(self) -> None:
        repository = InMemoryTaskRepository()
        registry = HandlerRegistry()
        replacement_task = Task(
            id="memory:next",
            kind="memory_reflection",
            payload={"user_input": "next", "assistant_output": "reply"},
            run_after=20.0,
            dedupe_key="memory_reflection:conversation_id:thread-1",
            metadata={"replace_pending": True},
        )
        registry.register("main", _SuccessHandler(next_tasks=[replacement_task]))
        runtime = Runtime(repository, registry)
        repository.enqueue(
            Task(
                id="memory:old",
                kind="memory_reflection",
                payload={"user_input": "old", "assistant_output": "reply"},
                run_after=15.0,
                dedupe_key="memory_reflection:conversation_id:thread-1",
                metadata={"replace_pending": True},
            )
        )
        repository.enqueue(Task(id="task:main", kind="main", payload={}))

        task = repository.lease_next_ready(now_unix=0.0)
        assert task is not None
        asyncio.run(runtime.execute_task(task, now_unix=0.0))

        old_task = repository.get("memory:old")
        new_task = repository.get("memory:next")
        assert old_task is not None
        assert new_task is not None
        self.assertEqual(old_task.status, "cancelled")
        self.assertIsNone(old_task.dedupe_key)
        self.assertEqual(new_task.status, "queued")
        self.assertEqual(new_task.dedupe_key, "memory_reflection:conversation_id:thread-1")


if __name__ == "__main__":
    unittest.main()
