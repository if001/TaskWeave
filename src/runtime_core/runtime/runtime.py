from __future__ import annotations

import asyncio
from time import time

from ..infra import get_logger
from ..types import Task, TaskContext, TaskResult, TaskStatus
from .registry import HandlerRegistry
from .repository import TaskRepository
from ..tasks.worker_recorder import WorkerLaunchRecorder
from .scheduler import PeriodicRule, RetryPolicy, TaskScheduler


logger = get_logger("taskweave.runtime_core.runtime")


class Runtime:
    def __init__(
        self,
        repository: TaskRepository,
        registry: HandlerRegistry,
        retry_policy: RetryPolicy | None = None,
        scheduler: TaskScheduler | None = None,
        periodic_rules: list[PeriodicRule] | None = None,
        recorder: WorkerLaunchRecorder | None = None,
    ) -> None:
        self._repository = repository
        self._registry = registry
        self._retry_policy = retry_policy or RetryPolicy()
        self._scheduler = scheduler or TaskScheduler()
        self._periodic_rules = periodic_rules or []
        self._recorder = recorder or WorkerLaunchRecorder()

    @property
    def repository(self) -> TaskRepository:
        return self._repository

    @property
    def recorder(self) -> WorkerLaunchRecorder:
        return self._recorder

    def enqueue_periodic_tasks(self, now_unix: float) -> None:
        self._enqueue_periodic_tasks(now_unix)

    def list_tasks(
        self,
        *,
        statuses: list[TaskStatus] | None = None,
        kinds: list[str] | None = None,
        parent_task_id: str | None = None,
        periodic_root_id: str | None = None,
    ) -> list[Task]:
        return self._repository.list_tasks(
            statuses=statuses,
            kinds=kinds,
            parent_task_id=parent_task_id,
            periodic_root_id=periodic_root_id,
        )

    def get_attempt(self, task_id: str) -> int:
        return self._repository.get_attempt(task_id)

    def cancel_task(self, task_id: str) -> bool:
        task = self._repository.get(task_id)
        if task is None or task.status in {"succeeded", "failed", "cancelled"}:
            return False
        task.metadata["cancellation_requested"] = True
        if task.status == "queued":
            self._repository.mark_status(task.id, "cancelled", reason="cancellation requested")
        return True

    def cancel_tasks_by_parent(self, parent_task_id: str) -> list[str]:
        return self._cancel_matching_tasks(parent_task_id=parent_task_id)

    def cancel_tasks_by_periodic_root(self, periodic_root_id: str) -> list[str]:
        return self._cancel_matching_tasks(periodic_root_id=periodic_root_id)

    async def tick(self, now_unix: float | None = None) -> bool:
        now = now_unix if now_unix is not None else time()
        self._enqueue_periodic_tasks(now)

        task = self._repository.lease_next_ready(now)
        if task is None:
            logger.debug("No task leased at now=%s", now)
            return False

        logger.info("Leased task id=%s kind=%s", task.id, task.kind)
        await self.execute_task(task, now_unix=now)
        return True

    async def execute_task(self, task: Task, now_unix: float | None = None) -> TaskResult:
        now = now_unix if now_unix is not None else time()
        deadline, early_result = self._prepare_execution(task, now)
        if early_result is not None:
            return early_result

        attempt = self._start_task(task)
        try:
            result = await self._run_handler(task, attempt, deadline, now)
        except asyncio.CancelledError:
            self._repository.mark_status(task.id, "cancelled", reason="runner cancelled")
            raise

        logger.info("Handler completed task id=%s status=%s", task.id, result.status)
        self._commit(task, result, now, attempt)
        return result

    def _cancel_matching_tasks(
        self,
        *,
        parent_task_id: str | None = None,
        periodic_root_id: str | None = None,
        dedupe_key: str | None = None,
        statuses: list[TaskStatus] | None = None,
    ) -> list[str]:
        tasks = self._repository.list_tasks(
            statuses=statuses,
            parent_task_id=parent_task_id,
            periodic_root_id=periodic_root_id,
            dedupe_key=dedupe_key,
        )
        cancelled: list[str] = []
        for task in tasks:
            if self.cancel_task(task.id):
                cancelled.append(task.id)
        return cancelled

    def _enqueue_periodic_tasks(self, now_unix: float) -> None:
        if not self._periodic_rules:
            return
        periodic_tasks = self._scheduler.generate_periodic_tasks(now_unix, self._periodic_rules)
        self._repository.enqueue_many(periodic_tasks)

    def _enqueue_next_tasks(self, tasks: list[Task]) -> None:
        for task in tasks:
            self._replace_pending_task(task)
            self._repository.enqueue(task)

    def _replace_pending_task(self, task: Task) -> None:
        if not _should_replace_pending(task):
            return
        dedupe_key = task.dedupe_key
        if dedupe_key is None:
            return
        pending_task_ids = self._cancel_matching_tasks(
            statuses=["queued", "leased"],
            dedupe_key=dedupe_key,
        )
        for pending_task_id in pending_task_ids:
            self._repository.clear_dedupe_key(pending_task_id)

    def _start_task(self, task: Task) -> int:
        self._repository.mark_status(task.id, "running")
        return self._repository.increment_attempt(task.id)

    def _prepare_execution(
        self, task: Task, now_unix: float
    ) -> tuple[float | None, TaskResult | None]:
        if self._is_cancelled(task):
            self._repository.mark_status(
                task.id, "cancelled", reason="cancellation requested"
            )
            logger.warning("Task cancelled before run id=%s", task.id)
            return None, TaskResult(status="failed", error="cancelled")

        deadline = self._resolve_deadline(task)
        if self._is_deadline_exceeded(deadline, now_unix):
            self._repository.mark_status(task.id, "failed", reason="deadline exceeded")
            logger.error(
                "Task deadline exceeded id=%s deadline=%s now=%s",
                task.id,
                deadline,
                now_unix,
            )
            return deadline, TaskResult(status="failed", error="deadline exceeded")

        return deadline, None

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
            logger.error("Handler timeout task id=%s", task.id)
            return TaskResult(status="failed", error="handler timeout")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Handler raised exception task id=%s", task.id)
            return TaskResult(status="failed", error=str(exc))

    def _commit(self, task: Task, result: TaskResult, now_unix: float, attempt: int) -> None:
        self._enqueue_next_tasks(result.next_tasks)
        if result.next_tasks:
            logger.info("Enqueued next tasks count=%s parent=%s", len(result.next_tasks), task.id)

        if result.status == "succeeded":
            self._repository.set_run_after(task.id, None)
            self._repository.mark_status(task.id, "succeeded")
            logger.info("Task succeeded id=%s", task.id)
            return

        if result.status == "failed":
            self._repository.mark_status(task.id, "failed", reason=result.error)
            logger.error("Task failed id=%s error=%s", task.id, result.error)
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
        logger.warning("Task retry scheduled id=%s attempt=%s run_after=%s reason=%s", task_id, attempt, run_after, reason or "retry")

    def _resolve_deadline(self, task: Task) -> float | None:
        value = task.metadata.get("deadline_unix")
        if value is None:
            return None
        if isinstance(value, (int, float, bool)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def _is_cancelled(self, task: Task) -> bool:
        return bool(task.metadata.get("cancellation_requested", False))

    def _is_deadline_exceeded(self, deadline_unix: float | None, now_unix: float) -> bool:
        if deadline_unix is None:
            return False
        return now_unix >= deadline_unix


def _should_replace_pending(task: Task) -> bool:
    replace_pending = task.metadata.get("replace_pending")
    return isinstance(replace_pending, bool) and replace_pending and task.dedupe_key is not None
