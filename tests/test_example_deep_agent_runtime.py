import asyncio
import unittest

from examples.deep_agent_runtime.bootstrap import EXAMPLE_TASK_ID, build_example_runtime, seed_example_task


class DeepAgentRuntimeExampleTests(unittest.TestCase):
    def test_example_runtime_completes_seeded_task(self) -> None:
        bundle = build_example_runtime()
        seed_example_task(bundle.repository, topic="test topic")

        async def _run() -> None:
            while await bundle.runtime.tick():
                pass

        asyncio.run(_run())

        task = bundle.repository.get(EXAMPLE_TASK_ID)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.status, "succeeded")

    def test_example_runtime_records_transitions(self) -> None:
        bundle = build_example_runtime()
        seed_example_task(bundle.repository, topic="transition topic")

        async def _run() -> None:
            while await bundle.runtime.tick():
                pass

        asyncio.run(_run())

        transition_statuses = [transition.to_status for transition in bundle.repository.transitions]
        self.assertEqual(transition_statuses, ["leased", "running", "succeeded"])


if __name__ == "__main__":
    unittest.main()
