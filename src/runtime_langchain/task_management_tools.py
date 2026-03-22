from __future__ import annotations

from runtime_core.runtime import Runtime
from runtime_core.types import JsonValue, Task, TaskStatus

from langchain_core.tools import BaseTool, tool


def build_task_management_tools(runtime: Runtime) -> list[BaseTool]:
    @tool("list_tasks")
    def list_tasks(
        status: str = "active",
        kind: str = "",
    ) -> dict[str, list[dict[str, JsonValue]]]:
        """List runtime tasks. status=active returns queued/leased/running tasks."""
        tasks = runtime.list_tasks(
            statuses=_parse_status_filter(status),
            kinds=[kind] if kind.strip() else None,
        )
        return {"tasks": [_serialize_task(runtime, task) for task in tasks]}

    @tool("cancel_task")
    def cancel_task(task_id: str) -> str:
        """Cancel a task by id."""
        return f"cancelled:{task_id}" if runtime.cancel_task(task_id) else f"not-cancelled:{task_id}"

    @tool("cancel_periodic_tasks")
    def cancel_periodic_tasks(periodic_root_id: str) -> dict[str, list[str]]:
        """Cancel periodic worker tasks by periodic_root_id."""
        return {"task_ids": runtime.cancel_tasks_by_periodic_root(periodic_root_id)}

    @tool("cancel_child_tasks")
    def cancel_child_tasks(parent_task_id: str) -> dict[str, list[str]]:
        """Cancel child tasks created from a parent task id."""
        return {"task_ids": runtime.cancel_tasks_by_parent(parent_task_id)}

    return [list_tasks, cancel_task, cancel_periodic_tasks, cancel_child_tasks]


def _parse_status_filter(status: str) -> list[TaskStatus] | None:
    normalized = status.strip().lower()
    if not normalized or normalized == "all":
        return None
    if normalized == "active":
        return ["queued", "leased", "running"]
    allowed: set[TaskStatus] = {"queued", "leased", "running", "succeeded", "failed", "cancelled"}
    if normalized in allowed:
        return [normalized]
    return None


def _serialize_task(runtime: Runtime, task: Task) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {
        "id": task.id,
        "kind": task.kind,
        "status": task.status,
        "attempt": runtime.get_attempt(task.id),
    }
    _set_if_present(result, "parent_task_id", task.parent_task_id)
    _set_if_present(result, "run_after", task.run_after)
    _set_payload_field(result, task, "periodic_root_id")
    _set_payload_field(result, task, "periodic_iteration")
    _set_payload_field(result, task, "remaining_runs")
    _set_payload_field(result, task, "query")
    _set_payload_field(result, task, "topic")
    _set_metadata_field(result, task, "conversation_id")
    _set_metadata_field(result, task, "agent_id")
    _set_metadata_field(result, task, "speaker_type")
    _set_metadata_field(result, task, "discord_request_task_id")
    _set_metadata_field(result, task, "root_trace_id")
    _set_metadata_field(result, task, "deadline_unix")
    _set_metadata_field(result, task, "cancellation_requested")
    _set_metadata_field(result, task, "enqueued_at_unix")
    return result


def _set_payload_field(result: dict[str, JsonValue], task: Task, key: str) -> None:
    _set_if_present(result, key, task.payload.get(key))


def _set_metadata_field(result: dict[str, JsonValue], task: Task, key: str) -> None:
    _set_if_present(result, key, task.metadata.get(key))


def _set_if_present(result: dict[str, JsonValue], key: str, value: JsonValue | None) -> None:
    if value is None:
        return
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and not value.strip():
            return
        result[key] = value
