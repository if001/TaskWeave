from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import gettempdir
from typing import TypedDict
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from examples.deep_agent_runtime.common import normalize_text
from examples.deep_agent_runtime.artifact_tools import artifact_save
from runtime_core.types import JsonValue, ensure_json_value

_SIMPLE_CLIENT_BASE_URL_ENV = "SIMPLE_CLIENT_BASE_URL"
_WEB_SEARCH_DIR_ENV = "EXAMPLE_WEB_SEARCH_DIR"
_WEB_SEARCH_TIMEOUT_SECONDS = 15.0
_WEB_SEARCH_SUMMARY_MAX_CHARS = 500

class SearchToolResult(TypedDict):
    status: str
    artifact_path: str
    summary: str


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
) -> SearchToolResult:
    normalized_query = normalize_text(query)
    if not normalized_query:
        return _error_result("query must not be empty")

    request_payload: dict[str, JsonValue] = {"q": normalized_query, "k": max(k, 1)}
    response, error = _post_json(url=f"{base_url}/list", payload=request_payload)
    if response is None:
        return _error_result(error or "web_list request failed")

    artifact_root = _resolve_artifact_dir(artifact_dir)
    raw_payload: dict[str, JsonValue] = {
        "kind": "web_list",
        "request": request_payload,
        "response": response,
    }
    meta = artifact_save(
        kind="web_list",
        raw=raw_payload,
        artifact_dir=artifact_root,
    )
    return SearchToolResult(
        status="ok",
        artifact_path=meta["raw_path"],
        summary=meta["summary"],
    )


def web_page_and_store_artifact(
    url: str,
    base_url: str,
    artifact_dir: Path | None = None,
) -> SearchToolResult:
    normalized_url = normalize_text(url)
    if not normalized_url:
        return _error_result("url must not be empty")

    request_payload: dict[str, JsonValue] = {"urls": normalized_url}
    response, error = _post_json(url=f"{base_url}/page", payload=request_payload)
    if response is None:
        return _error_result(error or "web_page request failed")

    artifact_root = _resolve_artifact_dir(artifact_dir)
    raw_payload: dict[str, JsonValue] = {
        "kind": "web_page",
        "request": request_payload,
        "response": response,
    }
    meta = artifact_save(
        kind="web_page",
        raw=raw_payload,
        artifact_dir=artifact_root,
    )
    return SearchToolResult(
        status="ok",
        artifact_path=meta["raw_path"],
        summary=meta["summary"],
    )


def _error_result(summary: str) -> SearchToolResult:
    return SearchToolResult(status="error", artifact_path="", summary=summary)


def _post_json(
    url: str, payload: dict[str, JsonValue]
) -> tuple[JsonValue | None, str | None]:
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=_WEB_SEARCH_TIMEOUT_SECONDS) as response:
            raw_text = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        return None, _format_http_error(exc)
    except (TimeoutError, SocketTimeout):
        return None, "web request timed out"
    except URLError as exc:
        if isinstance(exc.reason, (TimeoutError, SocketTimeout)):
            return None, "web request timed out"
        return None, f"web request failed: {exc.reason}"
    return _to_json_or_text(raw_text), None


def _format_http_error(exc: HTTPError) -> str:
    status = exc.code
    body = exc.read().decode("utf-8", errors="replace")
    summary = _summarize_payload(_to_json_or_text(body)) if body else ""
    if 400 <= status < 500:
        details = f": {summary}" if summary else ""
        return f"web request failed (client error {status}){details}"
    if 500 <= status < 600:
        return f"web request failed (server error {status})"
    return f"web request failed (status {status})"


def _to_json_or_text(raw_text: str) -> JsonValue:
    try:
        parsed: object = json.loads(raw_text)
        coerced = ensure_json_value(parsed)
        if coerced is not None:
            return coerced
    except json.JSONDecodeError:
        return {"raw_text": raw_text}
    return {"raw_text": raw_text}


def _resolve_artifact_dir(artifact_dir: Path | None) -> Path:
    base_dir = artifact_dir or Path(
        os.getenv(
            _WEB_SEARCH_DIR_ENV,
            str(Path(gettempdir()) / "taskweave_web_search_artifacts"),
        )
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def _summarize_payload(payload: JsonValue) -> str:
    rendered = (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )
    return " ".join(rendered.split())[:_WEB_SEARCH_SUMMARY_MAX_CHARS]


if __name__ == "__main__":
    url = "https://developers.openai.com/codex/learn/best-practices"
    request_payload: dict[str, JsonValue] = {"urls": url}
    base_url = "http://172.22.1.15:8000"
    print("base_url", base_url)
    response = _post_json(url=f"{base_url}/page", payload=request_payload)
    print(response)
