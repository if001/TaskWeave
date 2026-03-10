from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import time

from runtime_core.logging_utils import get_logger
from runtime_core.models import Task
from runtime_core.runtime import Runtime

logger = get_logger("taskweave.runtime_core.runner")


@dataclass(frozen=True, slots=True)
class RunnerPolicy:
    max_concurrency: int
    main_kinds: list[str]
    worker_kinds: list[str]
    main_slots: int = 1


class RuntimeRunner:
    def __init__(
        self,
        runtime: Runtime,
        policy: RunnerPolicy,
    ) -> None:
        self._runtime = runtime
        self._policy = policy

    async def run_once(self, now_unix: float | None = None) -> bool:
        now = now_unix if now_unix is not None else time()
        self._runtime.enqueue_periodic_tasks(now)
        tasks = self._lease_tasks(now)
        if not tasks:
            logger.debug("No tasks leased for runner at now=%s", now)
            return False
        await self._run_tasks(tasks, now)
        return True

    def _lease_tasks(self, now_unix: float) -> list[Task]:
        limit = max(self._policy.max_concurrency, 0)
        if limit == 0:
            return []

        leased: list[Task] = []
        leased.extend(self._lease_main_tasks(now_unix, limit))
        remaining = limit - len(leased)
        if remaining > 0:
            leased.extend(self._lease_worker_tasks(now_unix, remaining))
        return leased

    def _lease_main_tasks(self, now_unix: float, limit: int) -> list[Task]:
        slots = min(self._policy.main_slots, limit)
        return self._lease_by_kinds(now_unix, self._policy.main_kinds, slots)

    def _lease_worker_tasks(self, now_unix: float, limit: int) -> list[Task]:
        return self._lease_by_kinds(now_unix, self._policy.worker_kinds, limit)

    def _lease_by_kinds(
        self, now_unix: float, kinds: list[str], limit: int
    ) -> list[Task]:
        if not kinds or limit <= 0:
            return []
        leased: list[Task] = []
        for _ in range(limit):
            task = self._runtime.repository.lease_next_ready_by_kinds(now_unix, kinds)
            if task is None:
                break
            leased.append(task)
        return leased

    async def _run_tasks(self, tasks: list[Task], now_unix: float) -> None:
        async_tasks = [
            asyncio.create_task(self._runtime.execute_task(task, now_unix=now_unix))
            for task in tasks
        ]
        try:
            await asyncio.gather(*async_tasks)
        except asyncio.CancelledError:
            for task in async_tasks:
                task.cancel()
            await asyncio.gather(*async_tasks, return_exceptions=True)
            raise
