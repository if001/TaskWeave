import asyncio

from app.bootstrap import build_runtime
from runtime_core.models import Task


async def run_demo() -> None:
    runtime, repository, notification_service, artifact_service = build_runtime()
    repository.enqueue(Task(id="task:1", kind="user_request", payload={"text": "deep research"}))

    while await runtime.tick():
        pass

    print(notification_service.sent_messages)
    print(artifact_service.list_by_task("worker:task:1"))


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
