from __future__ import annotations

from app.handlers.main_handler import MainHandler
from app.handlers.notification_handler import NotificationHandler
from app.handlers.worker_handler import WorkerHandler
from app.services.artifact_service import InMemoryArtifactService
from app.services.notification_service import InMemoryNotificationService
from runtime_core.registry import HandlerRegistry
from runtime_core.repository import InMemoryTaskRepository
from runtime_core.runtime import Runtime


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
