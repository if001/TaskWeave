import unittest

from runtime_core.tasks import TaskResultConfig, build_main_task_result, build_worker_task_result
from runtime_core.types import MainAgentOutput, MainAgentRawResult, Task, TaskContext, WorkerAgentOutput


class TaskResultTests(unittest.TestCase):
    def test_build_main_task_result_creates_workers_and_notification(self) -> None:
        ctx = TaskContext(task=Task(id="main:1", kind="main", payload={}), attempt=1)
        raw: MainAgentRawResult = {
            "agent_output": MainAgentOutput(final_output="done"),
            "immediate_queries": ["q1"],
            "delayed_queries": [{"query": "q2", "delay_seconds": 5.0}],
            "periodic_queries": [
                {
                    "query": "q3",
                    "start_in_seconds": 0.0,
                    "interval_seconds": 10.0,
                    "repeat_count": 2,
                }
            ],
        }
        config = TaskResultConfig(
            worker_task_kind="worker",
            notification_task_kind="notification",
        )

        result = build_main_task_result(ctx, raw, config=config)

        task_ids = {task.id for task in result.next_tasks}
        self.assertIn("worker:main:1:now:1", task_ids)
        self.assertIn("worker:main:1:delayed:1", task_ids)
        self.assertIn("worker:main:1:periodic:1:1", task_ids)
        self.assertIn("notification:main:1:main", task_ids)
        self.assertEqual(len(result.next_tasks), 4)

    def test_build_worker_task_result_skips_periodic_when_complete(self) -> None:
        ctx = TaskContext(
            task=Task(
                id="worker:1",
                kind="worker",
                payload={
                    "remaining_runs": 1,
                    "periodic_interval_seconds": 10.0,
                },
            ),
            attempt=1,
        )
        raw: WorkerAgentOutput = {"final_output": "ok"}
        config = TaskResultConfig(
            worker_task_kind="worker",
            notification_task_kind="notification",
        )

        result = build_worker_task_result(ctx, raw, config=config)

        task_ids = {task.id for task in result.next_tasks}
        self.assertIn("notification:worker:1:worker_done", task_ids)
        self.assertEqual(
            [task_id for task_id in task_ids if task_id.startswith("worker:")],
            [],
        )


if __name__ == "__main__":
    unittest.main()
