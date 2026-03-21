from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from dataclasses import dataclass, field
from time import time
import uuid
import sys

import discord
from dotenv import load_dotenv

from runtime_core.infra import get_logger
from runtime_core.types import Task
from runtime_core.runtime import FileTaskRepository, RunnerPolicy, RuntimeRunner

from examples.deep_agent_runtime.bootstrap import (
    ExampleRuntimeBundle,
    TASK_KIND_MAIN_RESEARCH,
    TASK_KIND_NOTIFICATION,
    TASK_KIND_WORKER_RESEARCH,
    build_example_runtime,
)
from runtime_core.notifications import NotificationPayload, NotificationSenderBase

load_dotenv()
args = sys.argv
bot_key = args[-1]
_DISCORD_BOT_TOKEN = bot_key + "_DISCORD_TOKEN" if bot_key else "DISCORD_BOT_TOKEN"
_AGENT_ID_ENV = bot_key + "_AGENT_ID" if bot_key else "AGENT_ID"
_EXIT_NOTE = "Discord bot started. Mention this bot in a channel to create tasks."
_IDLE_SLEEP_SECONDS = 0.5
_TYPING_REFRESH_SECONDS = 8.0
_FALLBACK_PROMPT = "Please help with this request."


logger = get_logger("taskweave.examples.discord_bot")


def _empty_task_dict() -> dict[str, asyncio.Task[None]]:
    return {}


@dataclass(slots=True)
class TypingTaskController:
    tasks: dict[str, asyncio.Task[None]] = field(default_factory=_empty_task_dict)

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

        for chunk in _split_message(message):
            await channel.send(chunk)
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


class TaskWeaveDiscordBridge:
    def __init__(self, client: discord.Client) -> None:
        self._client = client
        self._typing_controller = TypingTaskController()
        self._agent_id = _resolve_agent_id()
        self._workspace_dir = _resolve_workspace_dir(self._agent_id)
        self._runtime_context: (
            contextlib.AbstractAsyncContextManager[ExampleRuntimeBundle] | None
        ) = None
        self._bundle: ExampleRuntimeBundle | None = None
        self._runner: RuntimeRunner | None = None
        self._turn = 1
        self._runtime_worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._ensure_runtime()
        if self._runtime_worker is None:
            self._runtime_worker = asyncio.create_task(self._runtime_loop())
            logger.info(f"Runtime loop started agent_id={self._agent_id}")

    async def stop(self) -> None:
        await self._typing_controller.stop_all()
        if self._runtime_worker is None:
            return
        self._runtime_worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._runtime_worker
        logger.info("Runtime loop stopped")
        if self._runtime_context is not None:
            await self._runtime_context.__aexit__(None, None, None)
            self._runtime_context = None
            self._bundle = None
            self._runner = None

    async def on_mention(self, message: discord.Message) -> None:
        await self._ensure_runtime()
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
        speaker_id = str(message.author.id)
        speaker_type = "user" if not message.author.bot else self._agent_id
        conversation_id = _conversation_id(message)
        thread_id = f"{self._agent_id}:{conversation_id}"
        task_id = f"discord:main:{self._turn}_{uuid.uuid4()}"
        task = Task(
            id=task_id,
            kind=TASK_KIND_MAIN_RESEARCH,
            payload={
                "topic": builder.build_topic(message),
                "delayed_jobs": [],
                "periodic_jobs": [],
            },
            metadata={
                "enqueued_at_unix": time(),
                "discord_channel_id": message.channel.id,
                "discord_requester_id": message.author.id,
                "discord_request_task_id": task_id,
                "agent_id": self._agent_id,
                "conversation_id": conversation_id,
                "speaker_id": speaker_id,
                "speaker_type": speaker_type,
                "thread_id": thread_id,
                "user_id": speaker_id if speaker_type == "user" else None,
            },
        )
        self._typing_controller.start(task.id, message.channel)
        if self._bundle is None:
            logger.error("Runtime bundle is unavailable; task not enqueued")
            return
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
            if self._runner is None:
                await asyncio.sleep(_IDLE_SLEEP_SECONDS)
                continue
            if not await self._runner.run_once():
                await asyncio.sleep(_IDLE_SLEEP_SECONDS)

    async def _ensure_runtime(self) -> None:
        if self._bundle is not None and self._runner is not None:
            return
        self._runtime_context = build_example_runtime(
            notification_sender=DiscordNotificationSender(
                self._client, typing_controller=self._typing_controller
            ),
            repository=FileTaskRepository(
                str(self._workspace_dir / self._agent_id / "task.json")
            ),
            workspace_dir=self._workspace_dir,
            agent_id=self._agent_id,
        )
        self._bundle = await self._runtime_context.__aenter__()
        self._runner = RuntimeRunner(
            runtime=self._bundle.runtime,
            policy=RunnerPolicy(
                max_concurrency=2,
                main_kinds=[TASK_KIND_MAIN_RESEARCH],
                worker_kinds=[TASK_KIND_WORKER_RESEARCH, TASK_KIND_NOTIFICATION],
            ),
        )


def _require_token() -> str:
    token = os.getenv(_DISCORD_BOT_TOKEN, "").strip()
    if not token:
        logger.error("Environment variable %s is not set", _DISCORD_BOT_TOKEN)
        raise RuntimeError("Set token before running this example.")
    return token


def _resolve_agent_id() -> str:
    return os.getenv(_AGENT_ID_ENV, "agent").strip() or "agent"


def _resolve_workspace_dir(agent_id: str) -> Path:
    base = Path(".state") / agent_id
    base.mkdir(parents=True, exist_ok=True)
    return base


# def _load_system_prompt() -> str | None:
#     path = os.getenv(_AGENT_PROMPT_PATH_ENV, "").strip()
#     if not path:
#         return None
#     prompt_path = Path(path)
#     if not prompt_path.exists():
#         logger.warning("System prompt file not found: %s", prompt_path)
#         return None
#     return prompt_path.read_text(encoding="utf-8")


def _conversation_id(message: discord.Message) -> str:
    if isinstance(message.channel, discord.Thread):
        return str(message.channel.id)
    if isinstance(message.channel, discord.TextChannel):
        return str(message.channel.id)
    return "unknown"


def _split_message(message: str, *, limit: int = 2000) -> list[str]:
    normalized = message.strip()
    if not normalized:
        return [""]
    if len(normalized) <= limit:
        return [normalized]
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + limit, len(normalized))
        chunks.append(normalized[start:end])
        start = end
    return chunks


async def _run() -> None:
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(
        intents=intents,
        allowed_mentions=discord.AllowedMentions(
            users=True, roles=True, everyone=False
        ),
    )
    bridge = TaskWeaveDiscordBridge(client=client)

    @client.event
    async def on_ready() -> None:  # pyright: ignore[reportUnusedFunction]
        await bridge.start()
        logger.info(_EXIT_NOTE)

    @client.event
    async def on_message(message: discord.Message) -> None:  # pyright: ignore[reportUnusedFunction]
        if client.user is None:
            return
        if message.author.bot and message.author.id == client.user.id:
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
