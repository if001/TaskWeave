from __future__ import annotations

from pathlib import Path

from langgraph.store.sqlite import SqliteStore

from examples.deep_agent_runtime.memory_store import (
    LANGGRAPH_STORE_FILENAME,
    search_profile_memories,
    search_topic_memories,
)


def test_search_profile_memories_reads_saved_items(tmp_path: Path) -> None:
    store_path = tmp_path / LANGGRAPH_STORE_FILENAME
    namespace = ("memories", "profile", "user-1")
    with SqliteStore.from_conn_string(str(store_path)) as store:
        store.setup()
        store.put(
            namespace,
            "pref-1",
            {"content": "User likes Python and backend tooling."},
            index=["content"],
        )

    result = search_profile_memories(
        query="Python",
        workspace_dir=tmp_path,
        owner_id="user-1",
        limit=3,
    )

    assert result["status"] == "ok"
    assert result["matches"]
    assert result["matches"][0]["key"] == "pref-1"
    assert "Python" in result["matches"][0]["text"]


def test_search_topic_memories_uses_owner_namespace(tmp_path: Path) -> None:
    store_path = tmp_path / LANGGRAPH_STORE_FILENAME
    with SqliteStore.from_conn_string(str(store_path)) as store:
        store.setup()
        store.put(
            ("memories", "topics", "owner-a"),
            "topic-a",
            {"content": "User tracks AI agent runtime changes."},
            index=["content"],
        )
        store.put(
            ("memories", "topics", "owner-b"),
            "topic-b",
            {"content": "This should not be visible for owner-a."},
            index=["content"],
        )

    result = search_topic_memories(
        query="runtime",
        workspace_dir=tmp_path,
        owner_id="owner-a",
        limit=5,
    )

    assert result["status"] == "ok"
    assert len(result["matches"]) == 1
    assert result["matches"][0]["key"] == "topic-a"

