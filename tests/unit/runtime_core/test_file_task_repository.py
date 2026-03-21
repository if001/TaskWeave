import tempfile
import unittest
from pathlib import Path

from runtime_core.types import Task
from runtime_core.runtime import FileTaskRepository


class FileTaskRepositoryTests(unittest.TestCase):
    def test_persists_and_restores_task_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "repo-state.json"

            repository = FileTaskRepository(state_file)
            repository.enqueue(Task(id="task:1", kind="user_request", payload={"text": "topic"}))
            leased = repository.lease_next_ready(now_unix=1.0)
            self.assertIsNotNone(leased)
            repository.mark_status("task:1", "running")
            repository.increment_attempt("task:1")
            repository.set_run_after("task:1", 10.0)

            loaded = FileTaskRepository(state_file)
            task = loaded.get("task:1")
            assert task is not None
            self.assertEqual(task.status, "running")
            self.assertEqual(task.run_after, 10.0)

    def test_lease_next_ready_respects_run_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "repo-state.json"
            repository = FileTaskRepository(state_file)

            repository.enqueue(
                Task(
                    id="task:delayed",
                    kind="user_request",
                    payload={},
                    run_after=50.0,
                )
            )

            self.assertIsNone(repository.lease_next_ready(now_unix=10.0))
            self.assertIsNotNone(repository.lease_next_ready(now_unix=50.0))

    def test_dedupe_policy_drop_skips_duplicate_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_file = Path(tmp_dir) / "repo-state.json"
            repository = FileTaskRepository(state_file, dedupe_policy="drop")

            repository.enqueue(Task(id="task:1", kind="user_request", payload={}, dedupe_key="k1"))
            repository.enqueue(Task(id="task:2", kind="user_request", payload={}, dedupe_key="k1"))

            self.assertIsNotNone(repository.get("task:1"))
            self.assertIsNone(repository.get("task:2"))


if __name__ == "__main__":
    unittest.main()
