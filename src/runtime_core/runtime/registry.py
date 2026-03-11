from __future__ import annotations

from typing import Protocol

from ..infra import UnknownTaskKindError, get_logger
from ..types import TaskContext, TaskResult


logger = get_logger("taskweave.runtime_core.registry")


class TaskHandler(Protocol):
    async def run(self, ctx: TaskContext) -> TaskResult: ...


class HandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, TaskHandler] = {}

    def register(self, kind: str, handler: TaskHandler) -> None:
        self._handlers[kind] = handler
        logger.debug("Handler registered kind=%s", kind)

    def resolve(self, kind: str) -> TaskHandler:
        try:
            handler = self._handlers[kind]
            logger.debug("Handler resolved kind=%s", kind)
            return handler
        except KeyError as exc:
            logger.error("Unknown task kind requested=%s", kind)
            raise UnknownTaskKindError(
                f"No handler registered for task kind: {kind}"
            ) from exc
