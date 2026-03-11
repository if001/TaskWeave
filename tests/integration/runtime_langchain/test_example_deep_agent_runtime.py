import asyncio
import unittest

from examples.deep_agent_runtime.bootstrap import EXAMPLE_TASK_ID, build_example_runtime, seed_example_task


class DeepAgentRuntimeExampleTests(unittest.TestCase):
    def test_example_runtime_completes_seeded_main_task(self) -> None:
        bundle = build_example_runtime()
        seed_example_task(bundle.repository, topic="test topic")

        async def _run() -> None:
            while await bundle.runtime.tick(now_unix=0.0):
                pass

        asyncio.run(_run())

        main_task = bundle.repository.get(EXAMPLE_TASK_ID)
        self.assertIsNotNone(main_task)
        assert main_task is not None
        self.assertEqual(main_task.status, "succeeded")

        self.assertIsNone(bundle.repository.get(f"worker:{EXAMPLE_TASK_ID}:now:1"))


if __name__ == "__main__":
    unittest.main()
