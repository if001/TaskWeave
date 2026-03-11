from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from runtime_core.types import JsonValue, Task, TaskContext, TaskResult
from runtime_core.runtime import HandlerRegistry, InMemoryTaskRepository, Runtime


class ArtifactService(Protocol):
    def put_text(
        self, namespace: str, path: str, text: str, metadata: dict[str, JsonValue]
    ) -> str: ...

    def read_text(self, artifact_id: str) -> str: ...

    def list_by_task(self, task_id: str) -> list[str]: ...


def _empty_str_dict() -> dict[str, str]:
    return {}


def _empty_str_list_dict() -> dict[str, list[str]]:
    return {}


def _empty_str_list() -> list[str]:
    return []


@dataclass(slots=True)
class InMemoryArtifactService:
    _content_by_id: dict[str, str] = field(default_factory=_empty_str_dict)
    _ids_by_task: dict[str, list[str]] = field(default_factory=_empty_str_list_dict)

    def put_text(
        self, namespace: str, path: str, text: str, metadata: dict[str, JsonValue]
    ) -> str:
        artifact_id = f"{namespace}:{path}"
        self._content_by_id[artifact_id] = text

        task_id = str(metadata.get("task_id", ""))
        if task_id:
            self._ids_by_task.setdefault(task_id, []).append(artifact_id)
        return artifact_id

    def read_text(self, artifact_id: str) -> str:
        return self._content_by_id[artifact_id]

    def list_by_task(self, task_id: str) -> list[str]:
        return list(self._ids_by_task.get(task_id, []))


@dataclass(slots=True)
class InMemoryNotificationService:
    sent_messages: list[str] = field(default_factory=_empty_str_list)

    async def send(self, payload: dict[str, JsonValue]) -> None:
        self.sent_messages.append(str(payload.get("message", "")))


class MainHandler:
    async def run(self, ctx: TaskContext) -> TaskResult:
        text = str(ctx.task.payload.get("text", "")).strip()
        worker_tasks = self._build_worker_tasks(ctx.task.id, text)
        return TaskResult(
            status="succeeded",
            output={"reply_text": f"Received: {text}"},
            next_tasks=worker_tasks,
        )

    def _build_worker_tasks(self, parent_task_id: str, text: str) -> list[Task]:
        if not text:
            return []
        return [
            Task(
                id=f"worker:{parent_task_id}",
                kind="worker_run",
                payload={"query": text},
                parent_task_id=parent_task_id,
            )
        ]


class WorkerHandler:
    def __init__(self, artifact_service: ArtifactService) -> None:
        self._artifact_service = artifact_service

    async def run(self, ctx: TaskContext) -> TaskResult:
        query = str(ctx.task.payload.get("query", "")).strip()
        summary = f"Background research done for: {query}" if query else "No query provided"
        artifact_ref = self._save_summary_artifact(ctx, summary)

        notification_task = Task(
            id=f"notification:{ctx.task.id}",
            kind="notification",
            payload={"message": summary},
            parent_task_id=ctx.task.id,
        )
        return TaskResult(
            status="succeeded",
            output={"artifact_ref": artifact_ref},
            next_tasks=[notification_task],
        )

    def _save_summary_artifact(self, ctx: TaskContext, summary: str) -> str:
        return self._artifact_service.put_text(
            namespace="worker_summary",
            path=ctx.task.id,
            text=summary,
            metadata={"task_id": ctx.task.id},
        )


class NotificationService(Protocol):
    async def send(self, payload: dict[str, JsonValue]) -> None: ...


class NotificationHandler:
    def __init__(self, service: NotificationService) -> None:
        self._service = service

    async def run(self, ctx: TaskContext) -> TaskResult:
        await self._service.send(ctx.task.payload)
        return TaskResult(status="succeeded")


def build_runtime() -> tuple[
    Runtime,
    InMemoryTaskRepository,
    InMemoryNotificationService,
    InMemoryArtifactService,
]:
    repository = InMemoryTaskRepository()
    registry = HandlerRegistry()
    notification_service = InMemoryNotificationService()
    artifact_service = InMemoryArtifactService()

    registry.register("user_request", MainHandler())
    registry.register("worker_run", WorkerHandler(artifact_service=artifact_service))
    registry.register("notification", NotificationHandler(notification_service))

    return Runtime(repository=repository, registry=registry), repository, notification_service, artifact_service
