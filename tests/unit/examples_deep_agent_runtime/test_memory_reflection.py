from __future__ import annotations

import asyncio
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langmem.knowledge.extraction import MessagesState, SearchItem

from examples.deep_agent_runtime.memory_reflection import (
    LangMemMemoryHooks,
    _MemoryChannel,
)
from runtime_core.types import Task, TaskContext
from runtime_langchain.task_orchestrator import GraphInput


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def submit(
        self,
        payload: dict[str, object],
        /,
        config: RunnableConfig | None = None,
        *,
        after_seconds: int = 0,
        thread_id: str | UUID | None = None,
    ) -> Future[object]:
        self.calls.append(
            {
                "payload": payload,
                "config": config,
                "after_seconds": after_seconds,
                "thread_id": thread_id,
            }
        )
        fut: Future[object] = Future()
        fut.set_result(None)
        return fut

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        _ = (wait, cancel_futures)


def _build_ctx() -> TaskContext:
    task = Task(
        id="main:1",
        kind="main_research",
        payload={"topic": "latest AI runtime updates"},
        metadata={
            "speaker_type": "user",
            "conversation_id": "conv-1",
            "user_id": "user-1",
        },
    )
    return TaskContext(task=task, attempt=1)


async def _fake_searcher(_: MessagesState, __: RunnableConfig) -> list[SearchItem]:
    return [
        SearchItem(
            namespace=("memories", "profile", "user-1"),
            key="mem-1",
            value={"content": "User prefers concise technical summaries."},
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            score=0.9,
        )
    ]


def test_before_invoke_injects_memory_block(tmp_path: Path) -> None:
    executor = _RecordingExecutor()
    hooks = LangMemMemoryHooks(
        workspace_dir=tmp_path,
        _channels=(
            _MemoryChannel(
                title="long_term_profile",
                namespace=("memories", "profile", "{langgraph_user_id}"),
                instructions="profile",
                executor=executor,
                searcher=_fake_searcher,
            ),
        ),
    )
    ctx = _build_ctx()
    inp = GraphInput(messages=[{"role": "user", "content": "what should I read next?"}])

    updated = asyncio.run(hooks.before_invoke(ctx, inp))

    first_message = updated["messages"][0]
    assert isinstance(first_message, dict)
    assert "Use the following long-term memory" in str(first_message["content"])
    assert "User prefers concise technical summaries." in str(first_message["content"])


def test_after_invoke_submits_reflection_payload(tmp_path: Path) -> None:
    executor = _RecordingExecutor()
    hooks = LangMemMemoryHooks(
        workspace_dir=tmp_path,
        delay_seconds=7,
        _channels=(
            _MemoryChannel(
                title="long_term_profile",
                namespace=("memories", "profile", "{langgraph_user_id}"),
                instructions="profile",
                executor=executor,
                searcher=_fake_searcher,
            ),
        ),
    )
    ctx = _build_ctx()
    output = GraphInput(messages=[{"role": "assistant", "content": "You should read article A."}])

    hooks.after_invoke(ctx, output)

    assert len(executor.calls) == 1
    call = executor.calls[0]
    assert call["after_seconds"] == 7
    assert call["thread_id"] == "conv-1"

    payload = call["payload"]
    assert isinstance(payload, dict)
    messages = payload.get("messages")
    assert isinstance(messages, list)
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert "latest AI runtime updates" in str(messages[0].content)
    assert "article A" in str(messages[1].content)
