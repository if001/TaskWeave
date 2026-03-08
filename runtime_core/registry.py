from __future__ import annotations

from typing import Protocol

from runtime_core.errors import UnknownTaskKindError
from runtime_core.models import TaskContext, TaskResult


class TaskHandler(Protocol):
    async def run(self, ctx: TaskContext) -> TaskResult: ...


class HandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, TaskHandler] = {}

    def register(self, kind: str, handler: TaskHandler) -> None:
        self._handlers[kind] = handler

    def resolve(self, kind: str) -> TaskHandler:
        try:
            return self._handlers[kind]
        except KeyError as exc:
            raise UnknownTaskKindError(f"No handler registered for task kind: {kind}") from exc
