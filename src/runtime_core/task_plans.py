from __future__ import annotations

from runtime_core.agent_types import DelayedWorkerPlan, PeriodicWorkerPlan


def to_delayed_plans(value: object) -> list[DelayedWorkerPlan]:
    plans: list[DelayedWorkerPlan] = []
    for item in _iter_dict_items(value):
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        plans.append(
            DelayedWorkerPlan(
                query=query,
                delay_seconds=max(parse_float(item.get("delay_seconds"), 0.0), 0.0),
            )
        )
    return plans


def to_periodic_plans(
    value: object, *, min_interval_seconds: float = 1.0
) -> list[PeriodicWorkerPlan]:
    plans: list[PeriodicWorkerPlan] = []
    for item in _iter_dict_items(value):
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        plans.append(
            PeriodicWorkerPlan(
                query=query,
                start_in_seconds=max(
                    parse_float(item.get("start_in_seconds"), 0.0), 0.0
                ),
                interval_seconds=max(
                    parse_float(item.get("interval_seconds"), 60.0),
                    min_interval_seconds,
                ),
                repeat_count=max(parse_int(item.get("repeat_count"), 1), 1),
            )
        )
    return plans


def parse_float(value: object, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def parse_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _iter_dict_items(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
