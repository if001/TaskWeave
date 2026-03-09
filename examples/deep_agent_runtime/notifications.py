from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

from runtime_core.models import Task, TaskContext, TaskResult

from examples.deep_agent_runtime.common import normalize_text


class NotificationPayload(TypedDict, total=False):
    message: str
    discord_channel_id: int
    discord_requester_id: int
    discord_request_task_id: str
    notification_kind: str


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
        await self.sender.send(notification_payload_from_task_payload(ctx.task.payload))
        return TaskResult(status="succeeded")


def notification_payload_from_task_payload(payload: dict[str, object]) -> NotificationPayload:
    result: NotificationPayload = {"message": normalize_text(payload.get("message", ""))}
    _set_int_if_present(result, "discord_channel_id", payload.get("discord_channel_id"))
    _set_int_if_present(result, "discord_requester_id", payload.get("discord_requester_id"))
    _set_str_if_present(result, "discord_request_task_id", payload.get("discord_request_task_id"))
    _set_str_if_present(result, "notification_kind", payload.get("notification_kind"))
    return result


def extract_notification_metadata(metadata: dict[str, object]) -> NotificationPayload:
    result: NotificationPayload = {}
    _set_int_if_present(result, "discord_channel_id", metadata.get("discord_channel_id"))
    _set_int_if_present(result, "discord_requester_id", metadata.get("discord_requester_id"))
    _set_str_if_present(result, "discord_request_task_id", metadata.get("discord_request_task_id"))
    return result


def render_output_message(raw: object) -> str:
    if isinstance(raw, dict):
        final_output = raw.get("final_output")
        if final_output is not None:
            return normalize_text(final_output)
    return normalize_text(raw)


def _set_int_if_present(payload: NotificationPayload, key: str, value: object) -> None:
    if isinstance(value, int):
        payload[key] = value
    elif isinstance(value, str) and value.isdigit():
        payload[key] = int(value)


def _set_str_if_present(payload: NotificationPayload, key: str, value: object) -> None:
    if isinstance(value, str) and value.strip():
        payload[key] = value.strip()
