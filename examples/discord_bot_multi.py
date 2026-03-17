from __future__ import annotations

import asyncio
import contextlib
import os
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from time import time

import discord
from dotenv import load_dotenv

from runtime_core.infra import get_logger
from runtime_core.notifications import NotificationPayload, NotificationSenderBase
from runtime_core.runtime import (
    FileTaskRepository,
    HandlerRegistry,
    RunnerPolicy,
    Runtime,
    RuntimeRunner,
)
from runtime_core.tasks import TaskResultConfig
from runtime_core.types import JsonValue, Task
from runtime_langchain.runtime_builder import ResearchRuntimeBuilder
from runtime_langchain.task_orchestrator import GraphInput
from langgraph.graph.state import CompiledStateGraph

from examples.deep_agent_runtime.bootstrap import (
    DEFAULT_MODEL_NAME,
    TASK_KIND_MAIN_RESEARCH,
    TASK_KIND_NOTIFICATION,
    TASK_KIND_WORKER_RESEARCH,
    _BACKEND_ENV,
    _BACKEND_LANGCHAIN,
    _MODEL_ENV,
    _REAL_AGENT_ENV,
)
from examples.deep_agent_runtime.main_agent_runnables import (
    build_main_deep_agent_graph,
)
from examples.deep_agent_runtime.worker_agent_runnables import (
    build_worker_agent_graph,
)

load_dotenv()
logger = get_logger("taskweave.examples.discord_bot_multi")

_DISCORD_BOT_TOKEN = "DISCORD_BOT_TOKEN"
_EXIT_NOTE = "Discord multi-agent bot started. Mention this bot to create tasks."
_IDLE_SLEEP_SECONDS = 0.5
_TYPING_REFRESH_SECONDS = 8.0
_FALLBACK_PROMPT = "Please help with this request."
_MAX_BOT_HOPS = 1


@dataclass(slots=True)
class AgentSpec:
    agent_id: str
    display_name: str
    system_prompt: str | None = None


@dataclass(slots=True)
class AgentRuntimeState:
    agent_id: str
    runtime: Runtime
    repository: FileTaskRepository
    runner: RuntimeRunner
    turn: int = 1

    def next_task_id(self) -> str:
        task_id = f"discord:{self.agent_id}:main:{self.turn}_{uuid.uuid4()}"
        self.turn += 1
        return task_id


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


class MultiAgentNotificationSender(NotificationSenderBase):
    def __init__(
        self,
        client: discord.Client,
        typing_controller: TypingTaskController,
        manager: "MultiAgentManager",
    ) -> None:
        self._client = client
        self._typing_controller = typing_controller
        self._manager = manager

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

        await self._maybe_dispatch_bot_mentions(payload, message)

    async def _maybe_dispatch_bot_mentions(
        self, payload: NotificationPayload, message: str
    ) -> None:
        speaker_type = payload.get("speaker_type")
        bot_hops = payload.get("bot_hops", 0)
        if speaker_type == "bot" and bot_hops >= _MAX_BOT_HOPS:
            return

        source_agent = payload.get("agent_id")
        conversation_id = payload.get("conversation_id")
        if not isinstance(source_agent, str) or not isinstance(conversation_id, str):
            return

        mentions = _extract_agent_mentions(message, self._manager.agent_ids)
        for target_agent in mentions:
            if target_agent == source_agent:
                continue
            self._manager.enqueue_bot_task(
                agent_id=target_agent,
                content=message,
                source_agent_id=source_agent,
                conversation_id=conversation_id,
                bot_hops=bot_hops + 1,
            )


class MultiAgentManager:
    def __init__(self, client: discord.Client, agent_specs: list[AgentSpec]) -> None:
        self._client = client
        self._agent_specs = agent_specs
        self._agent_ids = {spec.agent_id for spec in agent_specs}
        self._typing_controller = TypingTaskController()
        self._runtimes: dict[str, AgentRuntimeState] = {}
        self._runtime_tasks: list[asyncio.Task[None]] = []
        self._contexts: list[contextlib.AbstractAsyncContextManager[AgentRuntimeState]] = []

    @property
    def agent_ids(self) -> set[str]:
        return self._agent_ids

    async def start(self) -> None:
        if self._runtimes:
            return
        sender = MultiAgentNotificationSender(
            client=self._client,
            typing_controller=self._typing_controller,
            manager=self,
        )
        for spec in self._agent_specs:
            ctx = _build_agent_runtime(spec, sender)
            runtime_state = await ctx.__aenter__()
            self._contexts.append(ctx)
            self._runtimes[spec.agent_id] = runtime_state
            self._runtime_tasks.append(asyncio.create_task(self._runtime_loop(runtime_state)))
        logger.info("Multi-agent runtime started")

    async def stop(self) -> None:
        await self._typing_controller.stop_all()
        for task in self._runtime_tasks:
            task.cancel()
        for task in self._runtime_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._runtime_tasks.clear()
        for ctx in self._contexts:
            await ctx.__aexit__(None, None, None)
        self._contexts.clear()
        self._runtimes.clear()
        logger.info("Multi-agent runtime stopped")

    async def handle_user_mention(self, message: discord.Message) -> None:
        user = self._client.user
        if user is None:
            logger.error("Client user is unavailable; mention ignored")
            return
        if not isinstance(message.channel, discord.TextChannel):
            logger.warning("Non-text channel mention ignored: channel=%s", message.channel)
            return

        content = _strip_bot_mention(message.content, user.id)
        agent_id, content = _select_agent_id(content, self._agent_ids)
        runtime_state = self._runtimes.get(agent_id)
        if runtime_state is None:
            logger.error("Unknown agent_id=%s", agent_id)
            return

        task_id = runtime_state.next_task_id()
        metadata = _build_metadata(
            agent_id=agent_id,
            conversation_id=_conversation_id(message),
            speaker_id=str(message.author.id),
            speaker_type="user",
            user_id=str(message.author.id),
            thread_id=_thread_id(agent_id, message),
            bot_hops=0,
            discord_channel_id=message.channel.id,
            discord_requester_id=message.author.id,
            discord_request_task_id=task_id,
        )
        runtime_state.repository.enqueue(
            Task(
                id=task_id,
                kind=TASK_KIND_MAIN_RESEARCH,
                payload={"topic": content or _FALLBACK_PROMPT},
                metadata=metadata,
            )
        )
        self._typing_controller.start(task_id, message.channel)
        logger.info(
            "Task enqueued: id=%s agent=%s channel=%s author=%s",
            task_id,
            agent_id,
            message.channel.id,
            message.author.id,
        )

    def enqueue_bot_task(
        self,
        agent_id: str,
        content: str,
        source_agent_id: str,
        conversation_id: str,
        bot_hops: int,
    ) -> None:
        runtime_state = self._runtimes.get(agent_id)
        if runtime_state is None:
            return
        task_id = runtime_state.next_task_id()
        metadata = _build_metadata(
            agent_id=agent_id,
            conversation_id=conversation_id,
            speaker_id=source_agent_id,
            speaker_type=source_agent_id,
            user_id=None,
            thread_id=f"{agent_id}:{conversation_id}",
            bot_hops=bot_hops,
            discord_channel_id=int(conversation_id),
            discord_requester_id=None,
            discord_request_task_id=task_id,
        )
        runtime_state.repository.enqueue(
            Task(
                id=task_id,
                kind=TASK_KIND_MAIN_RESEARCH,
                payload={"topic": f"[from bot:{source_agent_id}] {content}"},
                metadata=metadata,
            )
        )
        logger.info(
            "Bot task enqueued: id=%s agent=%s from=%s",
            task_id,
            agent_id,
            source_agent_id,
        )

    async def _runtime_loop(self, runtime_state: AgentRuntimeState) -> None:
        while True:
            if not await runtime_state.runner.run_once():
                await asyncio.sleep(_IDLE_SLEEP_SECONDS)


@contextlib.asynccontextmanager
async def _build_agent_runtime(
    spec: AgentSpec, notification_sender: NotificationSenderBase
) -> AsyncIterator[AgentRuntimeState]:
    workspace_dir = _resolve_agent_workspace(spec.agent_id)
    repository = FileTaskRepository(str(workspace_dir / "task.json"))
    registry = HandlerRegistry()
    runtime = Runtime(repository=repository, registry=registry)
    builder = ResearchRuntimeBuilder(
        runtime,
        config=TaskResultConfig(
            worker_task_kind=TASK_KIND_WORKER_RESEARCH,
            notification_task_kind=TASK_KIND_NOTIFICATION,
        ),
    )

    async with build_main_deep_agent_graph(
        model_name=os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME),
        tools=builder.worker_tools(),
        workspace_dir=workspace_dir,
        skills_dir=None,
        system_prompt_override=spec.system_prompt,
    ) as main_graph:
        builder.register_main(
            registry,
            kind=TASK_KIND_MAIN_RESEARCH,
            runnable=main_graph,
        )
        builder.register_worker(
            registry,
            kind=TASK_KIND_WORKER_RESEARCH,
            runnable=_build_worker_agent_graph(workspace_dir),
        )
        builder.register_notification(
            registry,
            kind=TASK_KIND_NOTIFICATION,
            sender=notification_sender,
        )
        runner = RuntimeRunner(
            runtime=runtime,
            policy=RunnerPolicy(
                max_concurrency=2,
                main_kinds=[TASK_KIND_MAIN_RESEARCH],
                worker_kinds=[TASK_KIND_WORKER_RESEARCH, TASK_KIND_NOTIFICATION],
            ),
        )
        yield AgentRuntimeState(
            agent_id=spec.agent_id,
            runtime=runtime,
            repository=repository,
            runner=runner,
        )


def _build_worker_agent_graph(
    workspace_dir: Path,
) -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
    model_name = os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME)
    return build_worker_agent_graph(
        use_real_agent=_is_real_agent_enabled(),
        backend=_resolve_real_agent_backend(),
        model_name=model_name,
        artifact_dir=workspace_dir,
    )


def _resolve_real_agent_backend() -> str:
    selected = os.getenv(_BACKEND_ENV, _BACKEND_LANGCHAIN).strip().lower()
    return selected if selected in {"langchain", "deepagent"} else _BACKEND_LANGCHAIN


def _is_real_agent_enabled() -> bool:
    return os.getenv(_REAL_AGENT_ENV, "0") == "1"


def _resolve_agent_workspace(agent_id: str) -> Path:
    base = Path(".state") / agent_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def _strip_bot_mention(content: str, bot_user_id: int) -> str:
    stripped = content.replace(f"<@{bot_user_id}>", "")
    stripped = stripped.replace(f"<@!{bot_user_id}>", "")
    return stripped.strip()


def _select_agent_id(content: str, agent_ids: set[str]) -> tuple[str, str]:
    lowered = content.strip().lower()
    for agent_id in agent_ids:
        for prefix in (f"{agent_id}:", f"{agent_id} ", f"@{agent_id}"):
            if lowered.startswith(prefix):
                return agent_id, content[len(prefix) :].lstrip()
    return sorted(agent_ids)[0], content


def _extract_agent_mentions(message: str, agent_ids: set[str]) -> list[str]:
    matches = []
    for match in re.findall(r"@([a-zA-Z0-9_-]+)", message):
        if match in agent_ids:
            matches.append(match)
    return matches


def _conversation_id(message: discord.Message) -> str:
    if isinstance(message.channel, discord.Thread):
        return str(message.channel.id)
    if isinstance(message.channel, discord.TextChannel):
        return str(message.channel.id)
    return "unknown"


def _thread_id(agent_id: str, message: discord.Message) -> str:
    return f"{agent_id}:{_conversation_id(message)}"


def _build_metadata(
    *,
    agent_id: str,
    conversation_id: str,
    speaker_id: str,
    speaker_type: str,
    user_id: str | None,
    thread_id: str,
    bot_hops: int,
    discord_channel_id: int,
    discord_requester_id: int | None,
    discord_request_task_id: str,
) -> dict[str, JsonValue]:
    metadata: dict[str, JsonValue] = {
        "enqueued_at_unix": time(),
        "agent_id": agent_id,
        "conversation_id": conversation_id,
        "speaker_id": speaker_id,
        "speaker_type": speaker_type,
        "thread_id": thread_id,
        "bot_hops": bot_hops,
        "discord_channel_id": discord_channel_id,
        "discord_request_task_id": discord_request_task_id,
    }
    if user_id is not None:
        metadata["user_id"] = user_id
    if discord_requester_id is not None:
        metadata["discord_requester_id"] = discord_requester_id
    return metadata


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
    manager = MultiAgentManager(
        client=client,
        agent_specs=[
            AgentSpec(agent_id="agent-a", display_name="Agent A"),
            AgentSpec(agent_id="agent-b", display_name="Agent B"),
        ],
    )

    @client.event
    async def on_ready() -> None:  # pyright: ignore[reportUnusedFunction]
        await manager.start()
        logger.info(_EXIT_NOTE)

    @client.event
    async def on_message(message: discord.Message) -> None:  # pyright: ignore[reportUnusedFunction]
        if message.author.bot or client.user is None:
            return
        if client.user in message.mentions:
            await manager.handle_user_mention(message)

    try:
        await client.start(_require_token())
    except Exception:
        logger.exception("Discord bot terminated unexpectedly")
        raise
    finally:
        await manager.stop()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
