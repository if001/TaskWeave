from __future__ import annotations

import json
from pathlib import Path

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool

from .artifact_payloads import parse_article_artifact, parse_web_list_artifact
from .artifact_tools import SavedArtifact, artifact_index_path, artifact_search, save_article_artifact, save_web_list_artifact
from .content_description import ContentDescription
from .memory_store import (
    MemorySearchResult,
    search_profile_memories,
    search_topic_memories,
)
from .web_tools import (
    SearchToolResult,
    missing_search_service_result,
    resolve_simple_client_base_url,
    web_list_and_store_artifact,
    web_page_and_store_artifact,
)
from runtime_core.infra import get_logger
from runtime_core.types import JsonValue, ensure_json_value

logger = get_logger(__name__)


def build_research_tools(
    *,
    artifact_dir: Path,
    log_missing_base_url: bool = False,
) -> list[BaseTool]:
    base_url = resolve_simple_client_base_url()

    @tool("web_list")
    def web_list(query: str, k: int = 5) -> SearchToolResult:
        """Search the web and save the result list as an artifact."""
        if base_url is None:
            if log_missing_base_url:
                logger.error("base url not set")
            return missing_search_service_result()
        return web_list_and_store_artifact(
            query=query,
            k=k,
            base_url=base_url,
            artifact_dir=artifact_dir,
        )

    @tool("web_page")
    def web_page(url: str) -> SearchToolResult:
        """Fetch a web page and save the response as an artifact."""
        if base_url is None:
            if log_missing_base_url:
                logger.error("base url not set")
            return missing_search_service_result()
        return web_page_and_store_artifact(
            url=url,
            base_url=base_url,
            artifact_dir=artifact_dir,
        )

    @tool("artifact_save")
    def artifact_save_tool(kind: str, raw_json: str) -> dict[str, str]:
        """Save a typed artifact payload into /artifacts."""
        raw = parse_raw_json(raw_json)
        saved = _save_typed_artifact(kind=kind, raw=raw, artifact_dir=artifact_dir)
        return {
            "artifact_id": saved["id"],
            "raw_path": saved["raw_path"],
        }

    @tool("artifact_index")
    def artifact_index_tool(
        kind: str,
        raw_path: str,
        title: str,
        summary: str,
        tags_csv: str,
    ) -> dict[str, str]:
        """Index a saved artifact using explicit title, summary, and tags."""
        description: ContentDescription = {
            "title": title.strip(),
            "summary": summary.strip(),
            "tags": _parse_tags_csv(tags_csv),
        }
        meta = artifact_index_path(
            kind=kind,
            raw_path=Path(raw_path),
            description=description,
        )
        return {
            "artifact_id": meta["id"],
            "title": meta["title"],
            "summary": meta["summary"],
            "tags": ", ".join(meta["tags"]),
            "raw_path": meta["raw_path"],
        }

    @tool("artifact_search")
    def artifact_search_tool(
        query: str, limit: int = 5
    ) -> dict[str, list[dict[str, str]]]:
        """Search artifact metadata and return top matches."""
        matches = artifact_search(query=query, limit=limit)
        return {
            "matches": [
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    "title": item["title"],
                    "summary": item["summary"],
                    "raw_path": item["raw_path"],
                }
                for item in matches
            ]
        }

    @tool("memory_profile_search")
    def memory_profile_search(
        query: str,
        limit: int = 5,
        *,
        runtime: ToolRuntime,
    ) -> MemorySearchResult:
        """Search long-term profile memories for the current user."""
        owner_id = _runtime_owner_id(runtime)
        if not owner_id:
            return {"status": "error", "matches": []}
        return search_profile_memories(
            query=query,
            limit=limit,
            workspace_dir=artifact_dir.parent,
            owner_id=owner_id,
        )

    @tool("memory_topics_search")
    def memory_topics_search(
        query: str,
        limit: int = 5,
        *,
        runtime: ToolRuntime,
    ) -> MemorySearchResult:
        """Search long-term topic memories for the current user."""
        owner_id = _runtime_owner_id(runtime)
        if not owner_id:
            return {"status": "error", "matches": []}
        return search_topic_memories(
            query=query,
            limit=limit,
            workspace_dir=artifact_dir.parent,
            owner_id=owner_id,
        )

    return [
        web_list,
        web_page,
        artifact_save_tool,
        artifact_index_tool,
        artifact_search_tool,
        memory_profile_search,
        memory_topics_search,
    ]


def parse_raw_json(raw_json: str) -> JsonValue:
    try:
        parsed: object = json.loads(raw_json)
    except json.JSONDecodeError:
        parsed = None
    coerced = ensure_json_value(parsed) if parsed is not None else None
    if coerced is not None:
        return coerced
    return {"raw_text": raw_json}


def _runtime_owner_id(runtime: ToolRuntime) -> str:
    configurable = runtime.config.get("configurable")
    if not isinstance(configurable, dict):
        return ""
    owner_id = configurable.get("langgraph_user_id")
    if isinstance(owner_id, str) and owner_id.strip():
        return owner_id.strip()
    return ""


def _save_typed_artifact(*, kind: str, raw: JsonValue, artifact_dir: Path) -> SavedArtifact:
    if kind == "web_list":
        payload = parse_web_list_artifact(raw)
        if payload is None:
            raise ValueError("web_list artifact payload is invalid")
        return save_web_list_artifact(artifact=payload, artifact_dir=artifact_dir)
    if kind == "url_digest":
        payload = parse_article_artifact(raw)
        if payload is None:
            raise ValueError("url_digest artifact payload is invalid")
        return save_article_artifact(kind="url_digest", artifact=payload, artifact_dir=artifact_dir)
    if kind == "web_page":
        payload = parse_article_artifact(raw)
        if payload is None:
            raise ValueError("web_page artifact payload is invalid")
        return save_article_artifact(kind="web_page", artifact=payload, artifact_dir=artifact_dir)
    raise ValueError(f"unsupported artifact kind: {kind}")


def _parse_tags_csv(tags_csv: str) -> list[str]:
    return [tag.strip() for tag in tags_csv.split(",") if tag.strip()]
