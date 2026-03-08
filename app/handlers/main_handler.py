from __future__ import annotations

from runtime_core.models import Task, TaskContext, TaskResult


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
