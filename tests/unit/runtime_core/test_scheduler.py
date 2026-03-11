import unittest

from runtime_core.runtime import PeriodicRule, TaskScheduler


class TaskSchedulerTests(unittest.TestCase):
    def test_generate_periodic_tasks_emits_on_interval(self) -> None:
        scheduler = TaskScheduler()
        rules = [
            PeriodicRule(
                rule_id="ping",
                kind="periodic",
                interval_seconds=10.0,
                payload_factory=lambda: {"ok": True},
            )
        ]

        first = scheduler.generate_periodic_tasks(now_unix=0.0, rules=rules)
        second = scheduler.generate_periodic_tasks(now_unix=5.0, rules=rules)
        third = scheduler.generate_periodic_tasks(now_unix=10.0, rules=rules)

        self.assertEqual([task.id for task in first], ["periodic:ping:1"])
        self.assertEqual(second, [])
        self.assertEqual([task.id for task in third], ["periodic:ping:2"])


if __name__ == "__main__":
    unittest.main()
