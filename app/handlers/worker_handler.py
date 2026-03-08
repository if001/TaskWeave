from __future__ import annotations

from app.services.artifact_service import ArtifactService
from runtime_core.models import Task, TaskContext, TaskResult


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
