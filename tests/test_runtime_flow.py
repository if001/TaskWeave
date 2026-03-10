import asyncio
import unittest

from runtime_core.testing.runtime_flow_app import build_runtime
from runtime_core.models import Task, TaskContext, TaskResult
from runtime_core.registry import HandlerRegistry
from runtime_core.repository import InMemoryTaskRepository, TransitionPolicy
from runtime_core.runtime import Runtime
from runtime_core.scheduler import PeriodicRule, RetryPolicy, TaskScheduler


class _RetryOnceHandler:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx: TaskContext) -> TaskResult:
        self.calls += 1
        if self.calls == 1:
            return TaskResult(status="retry", error="temporary")
        return TaskResult(status="succeeded")


class _NoopHandler:
    async def run(self, ctx: TaskContext) -> TaskResult:
        _ = ctx
        return TaskResult(status="succeeded")


class _RejectAllTransitions(TransitionPolicy):
    def validate(self, from_status, to_status) -> None:
        raise ValueError(f"blocked: {from_status}->{to_status}")


class _ConstantScheduler(TaskScheduler):
    def __init__(self, next_retry: float) -> None:
        super().__init__()
        self._next_retry = next_retry

    def next_retry_time(self, now_unix: float, attempt: int, retry_policy: RetryPolicy) -> float:
        _ = now_unix
        _ = attempt
        _ = retry_policy
        return self._next_retry


class RuntimeFlowTests(unittest.TestCase):
    def test_user_request_to_notification_flow(self) -> None:
        runtime, repository, notification_service, artifact_service = build_runtime()
        repository.enqueue(Task(id="task:1", kind="user_request", payload={"text": "topic"}))

        asyncio.run(self._drain(runtime))

        self.assertEqual(notification_service.sent_messages, ["Background research done for: topic"])
        self.assertEqual(_require_task(repository.get("task:1"), "task:1").status, "succeeded")
        self.assertEqual(_require_task(repository.get("worker:task:1"), "worker:task:1").status, "succeeded")
        self.assertEqual(
            _require_task(
                repository.get("notification:worker:task:1"),
                "notification:worker:task:1",
            ).status,
            "succeeded",
        )
        self.assertEqual(
            artifact_service.read_text("worker_summary:worker:task:1"),
            "Background research done for: topic",
        )

    def test_retry_requeues_then_succeeds(self) -> None:
        repository = InMemoryTaskRepository()
        registry = HandlerRegistry()
        handler = _RetryOnceHandler()
        registry.register("retry_task", handler)
        runtime = Runtime(repository, registry)

        repository.enqueue(Task(id="retry:1", kind="retry_task", payload={}))

        first_tick = asyncio.run(runtime.tick(now_unix=100.0))
        self.assertTrue(first_tick)
        task = _require_task(repository.get("retry:1"), "retry:1")
        self.assertEqual(task.status, "queued")
        self.assertEqual(task.run_after, 101.0)

        second_tick_early = asyncio.run(runtime.tick(now_unix=100.5))
        self.assertFalse(second_tick_early)

        second_tick = asyncio.run(runtime.tick(now_unix=101.0))
        self.assertTrue(second_tick)
        self.assertEqual(_require_task(repository.get("retry:1"), "retry:1").status, "succeeded")

    def test_scheduler_controls_retry_time(self) -> None:
        repository = InMemoryTaskRepository()
        registry = HandlerRegistry()
        registry.register("retry_task", _RetryOnceHandler())
        runtime = Runtime(
            repository,
            registry,
            retry_policy=RetryPolicy(delay_seconds=9.0),
            scheduler=_ConstantScheduler(next_retry=123.0),
        )
        repository.enqueue(Task(id="retry:scheduler", kind="retry_task", payload={}))

        asyncio.run(runtime.tick(now_unix=0.0))

        self.assertEqual(
            _require_task(repository.get("retry:scheduler"), "retry:scheduler").run_after,
            123.0,
        )

    def test_periodic_rule_generates_tasks_without_duplicates(self) -> None:
        repository = InMemoryTaskRepository()
        registry = HandlerRegistry()
        registry.register("periodic_ping", _NoopHandler())
        runtime = Runtime(
            repository,
            registry,
            periodic_rules=[
                PeriodicRule(
                    rule_id="ping",
                    kind="periodic_ping",
                    interval_seconds=60.0,
                    payload_factory=lambda: {"source": "periodic"},
                )
            ],
        )

        first = asyncio.run(runtime.tick(now_unix=100.0))
        second = asyncio.run(runtime.tick(now_unix=100.0))
        third = asyncio.run(runtime.tick(now_unix=160.0))

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertTrue(third)
        self.assertEqual(
            _require_task(repository.get("periodic:ping:1"), "periodic:ping:1").status,
            "succeeded",
        )
        self.assertEqual(
            _require_task(repository.get("periodic:ping:2"), "periodic:ping:2").status,
            "succeeded",
        )

    def test_unknown_task_kind_fails_task(self) -> None:
        repository = InMemoryTaskRepository()
        runtime = Runtime(repository, HandlerRegistry())

        repository.enqueue(Task(id="missing:1", kind="unknown", payload={}))

        processed = asyncio.run(runtime.tick(now_unix=10.0))
        self.assertTrue(processed)
        self.assertEqual(_require_task(repository.get("missing:1"), "missing:1").status, "failed")

    def test_transition_history_records_reason_on_retry(self) -> None:
        repository = InMemoryTaskRepository()
        registry = HandlerRegistry()
        registry.register("retry_task", _RetryOnceHandler())
        runtime = Runtime(repository, registry)

        repository.enqueue(Task(id="retry:history", kind="retry_task", payload={}))
        asyncio.run(runtime.tick(now_unix=200.0))

        retry_transition = repository.transitions[-1]
        self.assertEqual(retry_transition.from_status, "running")
        self.assertEqual(retry_transition.to_status, "queued")
        self.assertEqual(retry_transition.reason, "temporary")

    def test_cancellation_requested_marks_task_cancelled(self) -> None:
        runtime, repository, _, _ = build_runtime()
        repository.enqueue(
            Task(
                id="cancel:1",
                kind="user_request",
                payload={"text": "topic"},
                metadata={"cancellation_requested": True},
            )
        )

        processed = asyncio.run(runtime.tick(now_unix=1.0))
        self.assertTrue(processed)
        self.assertEqual(_require_task(repository.get("cancel:1"), "cancel:1").status, "cancelled")

    def test_deadline_exceeded_marks_task_failed(self) -> None:
        runtime, repository, _, _ = build_runtime()
        repository.enqueue(
            Task(
                id="deadline:1",
                kind="user_request",
                payload={"text": "topic"},
                metadata={"deadline_unix": 10.0},
            )
        )

        processed = asyncio.run(runtime.tick(now_unix=10.0))
        self.assertTrue(processed)
        self.assertEqual(_require_task(repository.get("deadline:1"), "deadline:1").status, "failed")

    def test_transition_policy_can_be_injected(self) -> None:
        repository = InMemoryTaskRepository(transition_policy=_RejectAllTransitions())
        repository.enqueue(Task(id="x", kind="user_request", payload={}))

        with self.assertRaises(ValueError):
            repository.mark_status("x", "leased")

    def test_dedupe_policy_raise_is_default(self) -> None:
        repository = InMemoryTaskRepository()
        repository.enqueue(Task(id="t1", kind="a", payload={}, dedupe_key="k1"))

        with self.assertRaises(ValueError):
            repository.enqueue(Task(id="t2", kind="a", payload={}, dedupe_key="k1"))

    def test_dedupe_policy_drop_skips_duplicate_enqueue(self) -> None:
        repository = InMemoryTaskRepository(dedupe_policy="drop")
        repository.enqueue(Task(id="t1", kind="a", payload={}, dedupe_key="k1"))
        repository.enqueue(Task(id="t2", kind="a", payload={}, dedupe_key="k1"))

        self.assertIsNotNone(repository.get("t1"))
        self.assertIsNone(repository.get("t2"))

    async def _drain(self, runtime: Runtime) -> None:
        while await runtime.tick():
            pass


def _require_task(task: Task | None, task_id: str) -> Task:
    if task is None:
        raise AssertionError(f"task not found: {task_id}")
    return task


if __name__ == "__main__":
    unittest.main()
