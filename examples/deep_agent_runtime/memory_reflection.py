from __future__ import annotations

from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from inspect import isawaitable
from pathlib import Path
from typing import Protocol
from uuid import UUID

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.store.sqlite import SqliteStore
from langmem.knowledge.extraction import MessagesState, SearchItem

from .memory_store import LANGGRAPH_STORE_FILENAME, search_item_text
from .ollama_client import get_ollama_client
from runtime_core.infra import get_logger
from runtime_core.types import Message, TaskContext
from runtime_langchain.task_context_config import (
    build_langgraph_configurable,
    resolve_speaker_type,
    resolve_thread_id,
)
from runtime_langchain.task_orchestrator import GraphInput

logger = get_logger(__name__)

_PROFILE_INSTRUCTIONS = (
    "Extract stable user profile facts and preferences. "
    "Keep only durable information that should persist across many conversations."
)
_TOPICS_INSTRUCTIONS = (
    "Extract recurring and active user topics. "
    "Keep medium-term interests, repeated subjects, and ongoing themes."
)


class _ManagedReflectionExecutor(Protocol):
    def submit(
        self,
        payload: dict[str, object],
        /,
        config: RunnableConfig | None = None,
        *,
        after_seconds: int = 0,
        thread_id: str | UUID | None = None,
    ) -> Future[object]: ...

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None: ...


Searcher = Callable[[MessagesState, RunnableConfig], Awaitable[list[SearchItem]]]


@dataclass(slots=True)
class _MemoryChannel:
    title: str
    namespace: tuple[str, str, str]
    instructions: str
    executor: _ManagedReflectionExecutor | None = None
    searcher: Searcher | None = None


def _default_channels() -> tuple[_MemoryChannel, ...]:
    return (
        _MemoryChannel(
            title="long_term_profile",
            namespace=("memories", "profile", "{langgraph_user_id}"),
            instructions=_PROFILE_INSTRUCTIONS,
        ),
        _MemoryChannel(
            title="long_term_topics",
            namespace=("memories", "topics", "{langgraph_user_id}"),
            instructions=_TOPICS_INSTRUCTIONS,
        ),
    )


@dataclass(slots=True)
class LangMemMemoryHooks:
    workspace_dir: Path
    model_name: str = "gpt-oss:20b"
    delay_seconds: int = 5
    _store_context: AbstractContextManager[SqliteStore] | None = field(default=None, init=False)
    _store: SqliteStore | None = field(default=None, init=False)
    _channels: tuple[_MemoryChannel, ...] = field(
        default_factory=_default_channels
    )

    async def before_invoke(self, ctx: TaskContext, inp: GraphInput) -> GraphInput:
        if resolve_speaker_type(ctx) != "user" or not inp["messages"]:
            return inp
        query = _query_text(ctx)
        if not query or not self._ensure_resources():
            return inp
        search_input = MessagesState(messages=[HumanMessage(content=query)])
        sections = await self._memory_sections(search_input, _memory_config(ctx))
        memory_block = _memory_block(sections)
        if not memory_block:
            return inp
        return _inject_memory_block(inp, memory_block)

    def after_invoke(self, ctx: TaskContext, output: GraphInput) -> GraphInput:
        if resolve_speaker_type(ctx) != "user":
            return output
        payload = _build_reflection_payload(ctx, output)
        if payload is None or not self._ensure_resources():
            return output
        config = _memory_config(ctx)
        thread_id = resolve_thread_id(ctx)
        for channel in self._channels:
            _submit_reflection(channel.executor, payload, config, self.delay_seconds, thread_id)
        return output

    def shutdown(self) -> None:
        for channel in self._channels:
            executor = channel.executor
            if executor is None:
                continue
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                logger.warning("failed to shut down memory reflection executor: %s", exc)
        if self._store_context is not None:
            self._store_context.__exit__(None, None, None)
            self._store_context = None
            self._store = None

    async def _memory_sections(
        self,
        search_input: MessagesState,
        config: RunnableConfig,
    ) -> list[tuple[str, list[str]]]:
        sections: list[tuple[str, list[str]]] = []
        for channel in self._channels:
            if channel.searcher is None:
                continue
            try:
                results = await channel.searcher(search_input, config)
            except Exception as exc:
                logger.warning("failed to search memories: %s", exc)
                continue
            lines = _memory_lines(results)
            if lines:
                sections.append((channel.title, lines))
        return sections

    def _ensure_resources(self) -> bool:
        if all(channel.executor is not None and channel.searcher is not None for channel in self._channels):
            return True
        try:
            from langmem import ReflectionExecutor, create_memory_searcher, create_memory_store_manager
        except Exception as exc:
            logger.warning("memory reflection disabled: %s", exc)
            return False
        store = self._open_store()
        model = get_ollama_client(self.model_name)
        for channel in self._channels:
            manager = create_memory_store_manager(
                model,
                instructions=channel.instructions,
                namespace=channel.namespace,
                store=store,
            )
            channel.executor = ReflectionExecutor(manager, store=store)
            channel.searcher = _wrap_searcher(
                create_memory_searcher(model, namespace=channel.namespace)
            )
        return True

    def _open_store(self) -> SqliteStore:
        if self._store is not None:
            return self._store
        store_path = self.workspace_dir / LANGGRAPH_STORE_FILENAME
        context = SqliteStore.from_conn_string(str(store_path))
        store = context.__enter__()
        store.setup()
        self._store_context = context
        self._store = store
        return store


def _wrap_searcher(
    searcher: Runnable[MessagesState, Awaitable[list[SearchItem]]],
) -> Searcher:
    async def wrapped(payload: MessagesState, config: RunnableConfig) -> list[SearchItem]:
        result = await searcher.ainvoke(payload, config=config)
        if isawaitable(result):
            return await result
        return result

    return wrapped


def _inject_memory_block(inp: GraphInput, memory_block: str) -> GraphInput:
    messages = list(inp["messages"])
    first = messages[0]
    if not isinstance(first, dict):
        return inp
    content = str(first.get("content", "")).strip()
    messages[0] = Message(
        role=str(first.get("role", "user")),
        content=f"{memory_block}\n\n{content}" if content else memory_block,
    )
    return GraphInput(messages=messages)


def _memory_block(sections: list[tuple[str, list[str]]]) -> str:
    if not sections:
        return ""
    rendered_sections = [f"[{title}]\n" + "\n".join(lines) for title, lines in sections]
    return (
        "Use the following long-term memory only when relevant to the current response.\n\n"
        + "\n\n".join(rendered_sections)
    )


def _memory_lines(results: list[SearchItem]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for item in results:
        text = _memory_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        lines.append(f"- {text}")
    return lines


def _memory_text(item: SearchItem) -> str:
    return search_item_text(item)


def _build_reflection_payload(
    ctx: TaskContext,
    output: GraphInput,
) -> MessagesState | None:
    messages = _turn_messages(ctx, output)
    if not messages:
        return None
    return MessagesState(messages=messages)


def _turn_messages(ctx: TaskContext, output: GraphInput) -> list[AnyMessage]:
    user_input = _query_text(ctx)
    assistant_output = _assistant_output(output)
    if not user_input or not assistant_output:
        return []
    return [
        HumanMessage(content=user_input),
        AIMessage(content=assistant_output),
    ]


def _query_text(ctx: TaskContext) -> str:
    for key in ("topic", "query"):
        value = ctx.task.payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _assistant_output(output: GraphInput) -> str:
    for message in reversed(output["messages"]):
        if isinstance(message, dict):
            role = str(message.get("role", "")).lower()
            if role and role not in {"assistant", "ai"}:
                continue
            return str(message.get("content", "")).strip()
        role = getattr(message, "type", "")
        if isinstance(role, str) and role.lower() == "ai":
            return str(getattr(message, "content", "")).strip()
    return ""


def _memory_config(ctx: TaskContext) -> RunnableConfig:
    return {"configurable": build_langgraph_configurable(ctx, owner_default="default")}


def _submit_reflection(
    executor: _ManagedReflectionExecutor | None,
    payload: MessagesState,
    config: RunnableConfig,
    delay_seconds: int,
    thread_id: str,
) -> None:
    if executor is None:
        return
    try:
        executor.submit(
            {"messages": payload["messages"]},
            config=config,
            after_seconds=delay_seconds,
            thread_id=thread_id,
        )
    except Exception as exc:
        logger.warning("failed to submit memory reflection: %s", exc)
