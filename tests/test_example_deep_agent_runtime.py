import asyncio
import unittest

from runtime_core.models import Task

from examples.deep_agent_runtime.bootstrap import EXAMPLE_TASK_ID, TASK_KIND_MAIN_RESEARCH, build_example_runtime, seed_example_task
from examples.deep_agent_runtime.main import _build_worker_plan, _should_launch_worker


class DeepAgentRuntimeExampleTests(unittest.TestCase):
    def test_example_runtime_completes_seeded_main_and_immediate_worker_tasks(self) -> None:
        bundle = build_example_runtime()
        seed_example_task(bundle.repository, topic="test topic", needs_worker=True)

        async def _run() -> None:
            while await bundle.runtime.tick(now_unix=0.0):
                pass

        asyncio.run(_run())

        main_task = bundle.repository.get(EXAMPLE_TASK_ID)
        self.assertIsNotNone(main_task)
        assert main_task is not None
        self.assertEqual(main_task.status, "succeeded")

        worker_task = bundle.repository.get(f"worker:{EXAMPLE_TASK_ID}:now:1")
        self.assertIsNotNone(worker_task)
        assert worker_task is not None
        self.assertEqual(worker_task.status, "succeeded")

    def test_delayed_worker_runs_after_run_after(self) -> None:
        bundle = build_example_runtime()
        bundle.repository.enqueue(
            Task(
                id="main:delayed:1",
                kind=TASK_KIND_MAIN_RESEARCH,
                payload={
                    "topic": "schedule later",
                    "needs_worker": False,
                    "delayed_jobs": [{"query": "delayed query", "delay_seconds": 10.0}],
                    "periodic_jobs": [],
                },
                metadata={"enqueued_at_unix": 100.0},
            )
        )

        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=100.0)))
        delayed_task = bundle.repository.get("worker:main:delayed:1:delayed:1")
        self.assertIsNotNone(delayed_task)
        assert delayed_task is not None
        self.assertEqual(delayed_task.status, "queued")
        self.assertEqual(delayed_task.run_after, 110.0)

        self.assertFalse(asyncio.run(bundle.runtime.tick(now_unix=109.0)))
        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=110.0)))
        self.assertEqual(bundle.repository.get("worker:main:delayed:1:delayed:1").status, "succeeded")

    def test_periodic_worker_is_reenqueued_until_repeat_count(self) -> None:
        bundle = build_example_runtime()
        bundle.repository.enqueue(
            Task(
                id="main:periodic:1",
                kind=TASK_KIND_MAIN_RESEARCH,
                payload={
                    "topic": "periodic",
                    "needs_worker": False,
                    "delayed_jobs": [],
                    "periodic_jobs": [
                        {
                            "query": "periodic query",
                            "start_in_seconds": 0.0,
                            "interval_seconds": 60.0,
                            "repeat_count": 3,
                        }
                    ],
                },
                metadata={"enqueued_at_unix": 200.0},
            )
        )

        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=200.0)))
        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=200.0)))
        second = bundle.repository.get("worker:main:periodic:1:periodic:1:2")
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second.status, "queued")
        self.assertEqual(second.run_after, 260.0)

        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=260.0)))
        third = bundle.repository.get("worker:main:periodic:1:periodic:1:3")
        self.assertIsNotNone(third)
        assert third is not None
        self.assertEqual(third.status, "queued")
        self.assertEqual(third.run_after, 320.0)

        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=320.0)))
        self.assertEqual(bundle.repository.get("worker:main:periodic:1:periodic:1:3").status, "succeeded")

    def test_example_runtime_skips_worker_when_not_needed(self) -> None:
        bundle = build_example_runtime()
        seed_example_task(bundle.repository, topic="no worker", needs_worker=False)

        async def _run() -> None:
            while await bundle.runtime.tick(now_unix=0.0):
                pass

        asyncio.run(_run())

        self.assertIsNone(bundle.repository.get(f"worker:{EXAMPLE_TASK_ID}:now:1"))

    def test_terminal_worker_trigger_keywords(self) -> None:
        self.assertTrue(_should_launch_worker("please research this"))
        self.assertTrue(_should_launch_worker("この件を調査して"))
        self.assertFalse(_should_launch_worker("hello"))

    def test_terminal_worker_plan_includes_delayed_and_periodic(self) -> None:
        plan = _build_worker_plan("please research this later with periodic follow-up")
        self.assertTrue(bool(plan["needs_worker"]))
        self.assertEqual(len(plan["delayed_jobs"]), 1)
        self.assertEqual(len(plan["periodic_jobs"]), 1)


if __name__ == "__main__":
    unittest.main()
