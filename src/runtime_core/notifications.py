from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

from .types import JsonValue, MainAgentOutput, TaskContext, TaskResult, WorkerAgentOutput


class NotificationPayload(TypedDict, total=False):
    message: str
    discord_channel_id: int
    discord_requester_id: int
    discord_request_task_id: str
    notification_kind: str
    agent_id: str
    conversation_id: str
    speaker_id: str
    speaker_type: str
    bot_hops: int


class NotificationSender(Protocol):
    async def send(self, payload: NotificationPayload) -> None: ...


class NotificationSenderBase(NotificationSender):
    async def send(self, payload: NotificationPayload) -> None:
        raise NotImplementedError


class NoopNotificationSender(NotificationSenderBase):
    async def send(self, payload: NotificationPayload) -> None:
        _ = payload


@dataclass(slots=True)
class NotificationTaskHandler:
    sender: NotificationSender

    async def run(self, ctx: TaskContext) -> TaskResult:
        await self.sender.send(
            notification_payload_from_task_payload(ctx.task.payload)
        )
        return TaskResult(status="succeeded")


def notification_payload_from_task_payload(
    payload: dict[str, JsonValue],
) -> NotificationPayload:
    result: NotificationPayload = {"message": _normalize_text(payload.get("message", ""))}
    _set_int_if_present(result, "discord_channel_id", payload.get("discord_channel_id"))
    _set_int_if_present(result, "discord_requester_id", payload.get("discord_requester_id"))
    _set_int_if_present(result, "bot_hops", payload.get("bot_hops"))
    _set_str_if_present(result, "discord_request_task_id", payload.get("discord_request_task_id"))
    _set_str_if_present(result, "notification_kind", payload.get("notification_kind"))
    _set_str_if_present(result, "agent_id", payload.get("agent_id"))
    _set_str_if_present(result, "conversation_id", payload.get("conversation_id"))
    _set_str_if_present(result, "speaker_id", payload.get("speaker_id"))
    _set_str_if_present(result, "speaker_type", payload.get("speaker_type"))
    return result


def extract_notification_metadata(metadata: dict[str, JsonValue]) -> NotificationPayload:
    result: NotificationPayload = {}
    _set_int_if_present(result, "discord_channel_id", metadata.get("discord_channel_id"))
    _set_int_if_present(
        result, "discord_requester_id", metadata.get("discord_requester_id")
    )
    _set_int_if_present(result, "bot_hops", metadata.get("bot_hops"))
    _set_str_if_present(
        result, "discord_request_task_id", metadata.get("discord_request_task_id")
    )
    _set_str_if_present(result, "agent_id", metadata.get("agent_id"))
    _set_str_if_present(result, "conversation_id", metadata.get("conversation_id"))
    _set_str_if_present(result, "speaker_id", metadata.get("speaker_id"))
    _set_str_if_present(result, "speaker_type", metadata.get("speaker_type"))
    return result


def render_output_message(raw: MainAgentOutput | WorkerAgentOutput | str) -> str:
    if isinstance(raw, dict):
        final_output = raw.get("final_output", "")
        return _normalize_text(final_output)
    return _normalize_text(str(raw))


def _normalize_text(value: JsonValue) -> str:
    return str(value).strip()


def _set_int_if_present(
    payload: NotificationPayload, key: str, value: JsonValue
) -> None:
    if isinstance(value, int):
        payload[key] = value
    elif isinstance(value, str) and value.isdigit():
        payload[key] = int(value)


def _set_str_if_present(
    payload: NotificationPayload, key: str, value: JsonValue
) -> None:
    if isinstance(value, str) and value.strip():
        payload[key] = value.strip()
