from __future__ import annotations

from runtime_core.types import TaskContext


def resolve_owner_id(ctx: TaskContext, *, default: str = "") -> str:
    for key in ("user_id", "discord_requester_id", "conversation_id"):
        value = ctx.task.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return default


def resolve_thread_id(ctx: TaskContext) -> str:
    for key in ("conversation_id", "thread_id", "user_id", "discord_requester_id"):
        value = ctx.task.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return ctx.task.id


def resolve_speaker_type(ctx: TaskContext, *, default: str = "unknown") -> str:
    value = ctx.task.metadata.get("speaker_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def build_langgraph_configurable(
    ctx: TaskContext,
    *,
    owner_default: str = "",
) -> dict[str, str]:
    configurable: dict[str, str] = {"thread_id": resolve_thread_id(ctx)}
    owner_id = resolve_owner_id(ctx, default=owner_default)
    if owner_id:
        configurable["langgraph_user_id"] = owner_id
    return configurable
