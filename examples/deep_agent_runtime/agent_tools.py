from __future__ import annotations

import json
from pathlib import Path

from langchain.tools import tool
from langchain_core.tools import BaseTool

from .artifact_tools import artifact_save, artifact_search
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
        """Save a raw JSON payload into /artifacts and index its metadata."""
        meta = artifact_save(
            kind=kind,
            raw=parse_raw_json(raw_json),
            artifact_dir=artifact_dir,
        )
        return {
            "artifact_id": meta["id"],
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

    return [web_list, web_page, artifact_save_tool, artifact_search_tool]


def parse_raw_json(raw_json: str) -> JsonValue:
    try:
        parsed: object = json.loads(raw_json)
    except json.JSONDecodeError:
        parsed = None
    coerced = ensure_json_value(parsed) if parsed is not None else None
    if coerced is not None:
        return coerced
    return {"raw_text": raw_json}
