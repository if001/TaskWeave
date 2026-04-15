from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, TypedDict

from pydantic import BaseModel
from langgraph.store.base import SearchItem
from langgraph.store.sqlite import SqliteStore

LANGGRAPH_STORE_FILENAME = "langgraph_store.sqlite"


class MemorySearchMatch(TypedDict):
    key: str
    text: str
    namespace: str


class MemorySearchResult(TypedDict):
    status: str
    matches: list[MemorySearchMatch]


def search_profile_memories(
    *,
    query: str,
    workspace_dir: Path,
    owner_id: str,
    limit: int = 5,
) -> MemorySearchResult:
    return _search_memories(
        namespace=("memories", "profile", owner_id),
        query=query,
        workspace_dir=workspace_dir,
        limit=limit,
    )


def search_topic_memories(
    *,
    query: str,
    workspace_dir: Path,
    owner_id: str,
    limit: int = 5,
) -> MemorySearchResult:
    return _search_memories(
        namespace=("memories", "topics", owner_id),
        query=query,
        workspace_dir=workspace_dir,
        limit=limit,
    )


class _MemoryItemLike(Protocol):
    key: str
    value: BaseModel | Mapping[str, object]


def search_item_text(item: SearchItem | _MemoryItemLike) -> str:
    content = getattr(item.value, "content", None)
    if isinstance(content, Mapping):
        text = content.get("content")
        if isinstance(text, str):
            return text.strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def _search_memories(
    *,
    namespace: tuple[str, str, str],
    query: str,
    workspace_dir: Path,
    limit: int,
) -> MemorySearchResult:
    normalized_query = query.strip()
    if not normalized_query:
        return {"status": "error", "matches": []}

    store_path = workspace_dir / LANGGRAPH_STORE_FILENAME
    if not store_path.exists():
        return {"status": "ok", "matches": []}

    with SqliteStore.from_conn_string(str(store_path)) as store:
        store.setup()
        items = store.search(namespace, query=normalized_query, limit=max(limit, 1))

    matches: list[MemorySearchMatch] = []
    for item in items:
        text = search_item_text(item)
        if not text:
            continue
        matches.append(
            {
                "key": getattr(item, "key", ""),
                "text": text,
                "namespace": "/".join(namespace),
            }
        )
    return {"status": "ok", "matches": matches}
