import asyncio
import unittest

from tests.support.runtime_core.runtime_flow_app import build_runtime
from runtime_core.types import Task
from runtime_core.runtime import Runtime


class RuntimeFlowTests(unittest.TestCase):
    def test_user_request_to_notification_flow(self) -> None:
        runtime, repository, notification_service, artifact_service = build_runtime()
        repository.enqueue(Task(id="task:1", kind="user_request", payload={"text": "topic"}))

        asyncio.run(_drain(runtime))

        self.assertEqual(notification_service.sent_messages, ["Background research done for: topic"])
        self.assertEqual(_require_task(repository.get("task:1"), "task:1").status, "succeeded")
        self.assertEqual(_require_task(repository.get("worker:task:1"), "worker:task:1").status, "succeeded")
        self.assertEqual(
            _require_task(
                repository.get("notification:worker:task:1"),
                "notification:worker:task:1",
            ).status,
            "succeeded",
        )
        self.assertEqual(
            artifact_service.read_text("worker_summary:worker:task:1"),
            "Background research done for: topic",
        )


async def _drain(runtime: Runtime) -> None:
    while await runtime.tick():
        pass


def _require_task(task: Task | None, task_id: str) -> Task:
    if task is None:
        raise AssertionError(f"task not found: {task_id}")
    return task


if __name__ == "__main__":
    unittest.main()
