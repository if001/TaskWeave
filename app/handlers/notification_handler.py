from __future__ import annotations

from typing import Protocol

from runtime_core.models import TaskContext, TaskResult


class NotificationService(Protocol):
    async def send(self, payload: dict) -> None: ...


class NotificationHandler:
    def __init__(self, service: NotificationService) -> None:
        self._service = service

    async def run(self, ctx: TaskContext) -> TaskResult:
        await self._service.send(ctx.task.payload)
        return TaskResult(status="succeeded")
