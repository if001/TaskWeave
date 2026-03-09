from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import gettempdir
from typing import Callable, TypeAlias, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from examples.deep_agent_runtime.common import normalize_text

_SIMPLE_CLIENT_BASE_URL_ENV = "SIMPLE_CLIENT_BASE_URL"
_WEB_SEARCH_DIR_ENV = "EXAMPLE_WEB_SEARCH_DIR"
_WEB_SEARCH_TIMEOUT_SECONDS = 15.0
_WEB_SEARCH_SUMMARY_MAX_CHARS = 500

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | dict[str, "JsonValue"] | list["JsonValue"]
ArtifactWriter = Callable[[str], str]


class SearchToolResult(TypedDict):
    status: str
    artifact_path: str
    summary: str


class _ArtifactPayload(TypedDict):
    kind: str
    request: dict[str, JsonValue]
    response: JsonValue


def resolve_simple_client_base_url() -> str | None:
    base_url = os.getenv(_SIMPLE_CLIENT_BASE_URL_ENV, "").strip().rstrip("/")
    return base_url or None


def missing_search_service_result() -> SearchToolResult:
    return _error_result(
        "web search service base URL is not configured. "
        f"Set {_SIMPLE_CLIENT_BASE_URL_ENV} to enable web_list and web_page tools."
    )


def web_list_and_store_artifact(
    query: str,
    k: int,
    base_url: str,
    artifact_dir: Path | None = None,
    artifact_writer: ArtifactWriter | None = None,
) -> SearchToolResult:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return _error_result("query must not be empty")

    request_payload: dict[str, JsonValue] = {"q": normalized_query, "k": max(k, 1)}
    response = _post_json(url=f"{base_url}/list", payload=request_payload)
    if response is None:
        return _error_result("web_list request failed")

    artifact_path = _write_search_artifact(
        artifact_payload=_ArtifactPayload(kind="web_list", request=request_payload, response=response),
        artifact_dir=artifact_dir,
        artifact_writer=artifact_writer,
    )
    return SearchToolResult(status="ok", artifact_path=artifact_path, summary=_summarize_list_payload(response))


def web_page_and_store_artifact(
    url: str,
    base_url: str,
    artifact_dir: Path | None = None,
    artifact_writer: ArtifactWriter | None = None,
) -> SearchToolResult:
    normalized_url = normalize_text(url)
    if not normalized_url:
        return _error_result("url must not be empty")

    request_payload: dict[str, JsonValue] = {"url": normalized_url}
    response = _post_json(url=f"{base_url}/page", payload=request_payload)
    if response is None:
        return _error_result("web_page request failed")

    artifact_path = _write_search_artifact(
        artifact_payload=_ArtifactPayload(kind="web_page", request=request_payload, response=response),
        artifact_dir=artifact_dir,
        artifact_writer=artifact_writer,
    )
    return SearchToolResult(status="ok", artifact_path=artifact_path, summary=_summarize_page_payload(response))


def _error_result(summary: str) -> SearchToolResult:
    return SearchToolResult(status="error", artifact_path="", summary=summary)


def _post_json(url: str, payload: dict[str, JsonValue]) -> JsonValue | None:
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=_WEB_SEARCH_TIMEOUT_SECONDS) as response:
            raw_text = response.read().decode("utf-8", errors="replace")
    except (URLError, HTTPError, TimeoutError):
        return None
    return _to_json_or_text(raw_text)


def _to_json_or_text(raw_text: str) -> JsonValue:
    try:
        parsed = json.loads(raw_text)
        return parsed
    except json.JSONDecodeError:
        return {"raw_text": raw_text}


def _write_search_artifact(
    artifact_payload: _ArtifactPayload,
    artifact_dir: Path | None,
    artifact_writer: ArtifactWriter | None,
) -> str:
    payload_text = json.dumps(artifact_payload, ensure_ascii=False, indent=2)
    if artifact_writer is not None:
        return artifact_writer(payload_text)

    base_dir = artifact_dir or Path(os.getenv(_WEB_SEARCH_DIR_ENV, str(Path(gettempdir()) / "taskweave_web_search_artifacts")))
    base_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{artifact_payload['kind']}_{int.from_bytes(os.urandom(6), 'big')}.json"
    artifact = base_dir / filename
    artifact.write_text(payload_text, encoding="utf-8")
    return str(artifact)


def _summarize_list_payload(payload: JsonValue) -> str:
    if not isinstance(payload, dict):
        return _summarize_payload(payload)

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return _summarize_payload(payload)

    titles = [
        normalize_text(item.get("title", ""))
        for item in raw_results[:3]
        if isinstance(item, dict) and normalize_text(item.get("title", ""))
    ]
    query = normalize_text(payload.get("query", ""))
    summary = f"web_list query='{query}' results={len(raw_results)} top_titles={', '.join(titles) if titles else '(no titles)'}"
    return summary[:_WEB_SEARCH_SUMMARY_MAX_CHARS]


def _summarize_page_payload(payload: JsonValue) -> str:
    if not isinstance(payload, dict):
        return _summarize_payload(payload)

    raw_docs = payload.get("docs")
    if not isinstance(raw_docs, list):
        return _summarize_payload(payload)
    if not raw_docs:
        return "web_page docs=0"

    first = raw_docs[0]
    if not isinstance(first, dict):
        return f"web_page docs={len(raw_docs)}"

    title = normalize_text(first.get("title", ""))
    url = normalize_text(first.get("url", ""))
    markdown = normalize_text(first.get("markdown", ""))
    summary = f"web_page docs={len(raw_docs)} first_title='{title}' first_url='{url}' markdown_chars={len(markdown)}"
    return summary[:_WEB_SEARCH_SUMMARY_MAX_CHARS]


def _summarize_payload(payload: JsonValue) -> str:
    rendered = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return " ".join(rendered.split())[:_WEB_SEARCH_SUMMARY_MAX_CHARS]


