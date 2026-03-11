from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..infra import get_logger
from ..types import Task


logger = get_logger("taskweave.runtime_core.scheduler")


@dataclass(slots=True)
class RetryPolicy:
    delay_seconds: float = 1.0

    def next_run_after(self, now_unix: float, attempt: int) -> float:
        _ = attempt
        return now_unix + self.delay_seconds


@dataclass(slots=True)
class PeriodicRule:
    rule_id: str
    kind: str
    interval_seconds: float
    payload_factory: Callable[[], dict]
    metadata_factory: Callable[[], dict] = field(default_factory=lambda: (lambda: {}))


class TaskScheduler:
    def __init__(self) -> None:
        self._next_run_by_rule: dict[str, float] = {}
        self._emitted_count: dict[str, int] = {}

    def is_runnable(self, task: Task, now_unix: float) -> bool:
        if task.status != "queued":
            return False
        if task.run_after is None:
            return True
        return task.run_after <= now_unix

    def next_retry_time(self, now_unix: float, attempt: int, retry_policy: RetryPolicy) -> float:
        next_time = retry_policy.next_run_after(now_unix=now_unix, attempt=attempt)
        logger.info("Next retry time calculated attempt=%s next_run=%s", attempt, next_time)
        return next_time

    def generate_periodic_tasks(self, now_unix: float, rules: list[PeriodicRule]) -> list[Task]:
        generated: list[Task] = []
        for rule in rules:
            next_run = self._next_run_by_rule.setdefault(rule.rule_id, now_unix)
            if now_unix < next_run:
                continue
            sequence = self._emitted_count.get(rule.rule_id, 0) + 1
            self._emitted_count[rule.rule_id] = sequence
            self._next_run_by_rule[rule.rule_id] = now_unix + rule.interval_seconds
            task = Task(
                id=f"periodic:{rule.rule_id}:{sequence}",
                kind=rule.kind,
                payload=rule.payload_factory(),
                metadata=rule.metadata_factory(),
            )
            generated.append(task)
            logger.info("Periodic task generated rule=%s task_id=%s", rule.rule_id, task.id)
        return generated
