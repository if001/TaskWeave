from __future__ import annotations

import asyncio
from time import time

from runtime_core.models import Task, TaskContext, TaskResult
from runtime_core.registry import HandlerRegistry
from runtime_core.repository import TaskRepository
from runtime_core.scheduler import PeriodicRule, RetryPolicy, TaskScheduler


class Runtime:
    def __init__(
        self,
        repository: TaskRepository,
        registry: HandlerRegistry,
        retry_policy: RetryPolicy | None = None,
        scheduler: TaskScheduler | None = None,
        periodic_rules: list[PeriodicRule] | None = None,
    ) -> None:
        self._repository = repository
        self._registry = registry
        self._retry_policy = retry_policy or RetryPolicy()
        self._scheduler = scheduler or TaskScheduler()
        self._periodic_rules = periodic_rules or []

    async def tick(self, now_unix: float | None = None) -> bool:
        now = now_unix if now_unix is not None else time()
        self._enqueue_periodic_tasks(now)

        task = self._repository.lease_next_ready(now)
        if task is None:
            return False

        if self._is_cancelled(task):
            self._repository.mark_status(task.id, "cancelled", reason="cancellation requested")
            return True

        deadline = self._resolve_deadline(task)
        if self._is_deadline_exceeded(deadline, now):
            self._repository.mark_status(task.id, "failed", reason="deadline exceeded")
            return True

        attempt = self._start_task(task)
        result = await self._run_handler(task, attempt, deadline, now)
        self._commit(task, result, now, attempt)
        return True

    def _enqueue_periodic_tasks(self, now_unix: float) -> None:
        if not self._periodic_rules:
            return
        periodic_tasks = self._scheduler.generate_periodic_tasks(now_unix, self._periodic_rules)
        self._repository.enqueue_many(periodic_tasks)

    def _start_task(self, task: Task) -> int:
        self._repository.mark_status(task.id, "running")
        return self._repository.increment_attempt(task.id)

    async def _run_handler(
        self,
        task: Task,
        attempt: int,
        deadline_unix: float | None,
        now_unix: float,
    ) -> TaskResult:
        ctx = TaskContext(
            task=task,
            attempt=attempt,
            deadline_unix=deadline_unix,
            cancellation_requested=self._is_cancelled(task),
        )
        try:
            handler = self._registry.resolve(task.kind)
            if deadline_unix is None:
                return await handler.run(ctx)

            timeout_seconds = max(deadline_unix - now_unix, 0.0)
            return await asyncio.wait_for(handler.run(ctx), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return TaskResult(status="failed", error="handler timeout")
        except Exception as exc:  # noqa: BLE001
            return TaskResult(status="failed", error=str(exc))

    def _commit(self, task: Task, result: TaskResult, now_unix: float, attempt: int) -> None:
        self._repository.enqueue_many(result.next_tasks)

        if result.status == "succeeded":
            self._repository.set_run_after(task.id, None)
            self._repository.mark_status(task.id, "succeeded")
            return

        if result.status == "failed":
            self._repository.mark_status(task.id, "failed", reason=result.error)
            return

        self._schedule_retry(task.id, now_unix, attempt, result.error)

    def _schedule_retry(self, task_id: str, now_unix: float, attempt: int, reason: str | None) -> None:
        run_after = self._scheduler.next_retry_time(
            now_unix=now_unix,
            attempt=attempt,
            retry_policy=self._retry_policy,
        )
        self._repository.set_run_after(task_id, run_after)
        self._repository.mark_status(task_id, "queued", reason=reason or "retry")

    def _resolve_deadline(self, task: Task) -> float | None:
        value = task.metadata.get("deadline_unix")
        if value is None:
            return None
        return float(value)

    def _is_cancelled(self, task: Task) -> bool:
        return bool(task.metadata.get("cancellation_requested", False))

    def _is_deadline_exceeded(self, deadline_unix: float | None, now_unix: float) -> bool:
        if deadline_unix is None:
            return False
        return now_unix >= deadline_unix
