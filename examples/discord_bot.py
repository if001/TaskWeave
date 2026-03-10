from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import dataclass, field
from time import time

import discord
from dotenv import load_dotenv

from runtime_core.logging_utils import get_logger
from runtime_core.models import Task

from examples.deep_agent_runtime.bootstrap import (
    TASK_KIND_MAIN_RESEARCH,
    build_example_runtime,
)
from examples.deep_agent_runtime.notifications import (
    NotificationPayload,
    NotificationSenderBase,
)

load_dotenv()
_DISCORD_BOT_TOKEN = "DISCORD_BOT_TOKEN"
_EXIT_NOTE = "Discord bot started. Mention this bot in a channel to create tasks."
_IDLE_SLEEP_SECONDS = 0.5
_TYPING_REFRESH_SECONDS = 8.0
_FALLBACK_PROMPT = "Please help with this request."


logger = get_logger("taskweave.examples.discord_bot")


@dataclass(slots=True)
class TypingTaskController:
    tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)

    def start(self, request_task_id: str, channel: discord.TextChannel) -> None:
        self.stop(request_task_id)
        self.tasks[request_task_id] = asyncio.create_task(self._typing_loop(channel))

    def stop(self, request_task_id: str) -> None:
        running = self.tasks.pop(request_task_id, None)
        if running is not None:
            running.cancel()

    async def stop_all(self) -> None:
        task_ids = list(self.tasks.keys())
        for task_id in task_ids:
            self.stop(task_id)
        await asyncio.sleep(0)

    async def _typing_loop(self, channel: discord.TextChannel) -> None:
        while True:
            async with channel.typing():
                await asyncio.sleep(_TYPING_REFRESH_SECONDS)


class DiscordNotificationSender(NotificationSenderBase):
    def __init__(
        self, client: discord.Client, typing_controller: TypingTaskController
    ) -> None:
        self._client = client
        self._typing_controller = typing_controller

    async def send(self, payload: NotificationPayload) -> None:
        channel_id = payload.get("discord_channel_id")
        message = payload.get("message", "")
        if channel_id is None or not message.strip():
            logger.warning("Skip notification: channel_id or message is invalid")
            return

        channel = self._client.get_channel(channel_id)
        if not isinstance(channel, discord.TextChannel):
            logger.error("Skip notification: unknown text channel id=%s", channel_id)
            return

        await channel.send(message)
        logger.info("Notification sent to channel=%s", channel_id)

        if payload.get("notification_kind") == "main_result":
            request_task_id = payload.get("discord_request_task_id")
            if isinstance(request_task_id, str):
                self._typing_controller.stop(request_task_id)


@dataclass(slots=True)
class MentionTaskBuilder:
    bot_user_id: int

    def build_topic(self, message: discord.Message) -> str:
        topic = message.content.replace(f"<@{self.bot_user_id}>", "")
        topic = topic.replace(f"<@!{self.bot_user_id}>", "").strip()
        return topic or _FALLBACK_PROMPT

    def build_task(self, turn: int, message: discord.Message) -> Task:
        task_id = f"discord:main:{turn}"
        return Task(
            id=task_id,
            kind=TASK_KIND_MAIN_RESEARCH,
            payload={
                "topic": self.build_topic(message),
                "needs_worker": True,
                "delayed_jobs": [],
                "periodic_jobs": [],
            },
            metadata={
                "enqueued_at_unix": time(),
                "discord_channel_id": message.channel.id,
                "discord_requester_id": message.author.id,
                "discord_request_task_id": task_id,
            },
        )


class TaskWeaveDiscordBridge:
    def __init__(self, client: discord.Client) -> None:
        self._client = client
        self._typing_controller = TypingTaskController()
        self._bundle = build_example_runtime(
            notification_sender=DiscordNotificationSender(
                client, typing_controller=self._typing_controller
            )
        )
        self._turn = 1
        self._runtime_worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._runtime_worker is None:
            self._runtime_worker = asyncio.create_task(self._runtime_loop())
            logger.info("Runtime loop started")

    async def stop(self) -> None:
        await self._typing_controller.stop_all()
        if self._runtime_worker is None:
            return
        self._runtime_worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._runtime_worker
        logger.info("Runtime loop stopped")

    async def on_mention(self, message: discord.Message) -> None:
        user = self._client.user
        if user is None:
            logger.error("Client user is unavailable; mention ignored")
            return
        if not isinstance(message.channel, discord.TextChannel):
            logger.warning(
                "Non-text channel mention ignored: channel=%s", message.channel
            )
            return

        builder = MentionTaskBuilder(bot_user_id=user.id)
        task = builder.build_task(self._turn, message)
        self._typing_controller.start(task.id, message.channel)
        self._bundle.repository.enqueue(task)
        logger.info(
            "Task enqueued: id=%s channel=%s author=%s",
            task.id,
            message.channel.id,
            message.author.id,
        )
        self._turn += 1

    async def _runtime_loop(self) -> None:
        while True:
            if not await self._bundle.runtime.tick():
                await asyncio.sleep(_IDLE_SLEEP_SECONDS)


def _require_token() -> str:
    token = os.getenv(_DISCORD_BOT_TOKEN, "").strip()
    if not token:
        logger.error("Environment variable %s is not set", _DISCORD_BOT_TOKEN)
        raise RuntimeError("Set token before running this example.")
    return token


async def _run() -> None:
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    bridge = TaskWeaveDiscordBridge(client=client)

    @client.event
    async def on_ready() -> None:
        await bridge.start()
        logger.info(_EXIT_NOTE)

    @client.event
    async def on_message(message: discord.Message) -> None:
        if message.author.bot or client.user is None:
            return
        if client.user in message.mentions:
            await bridge.on_mention(message)

    try:
        await client.start(_require_token())
    except Exception:
        logger.exception("Discord bot terminated unexpectedly")
        raise
    finally:
        await bridge.stop()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
