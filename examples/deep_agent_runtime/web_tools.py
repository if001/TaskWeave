from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import gettempdir
from typing import TypedDict
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .common import normalize_text
from .artifact_payloads import ArticleArtifact, WebListArtifact, WebListItem, article_description_text, web_list_description_text
from .artifact_tools import artifact_index, save_article_artifact, save_web_list_artifact
from .content_description import describe_content
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
    payload = _build_web_list_artifact(normalized_query, response)
    saved = save_web_list_artifact(artifact=payload, artifact_dir=artifact_root)
    description = describe_content(
        content=web_list_description_text(payload),
        fallback_title=normalized_query,
        default_tags=["web_list"],
    )
    meta = artifact_index(saved=saved, description=description)
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
    article = _build_web_page_artifact(normalized_url, response)
    saved = save_article_artifact(
        kind="web_page",
        artifact=article,
        artifact_dir=artifact_root,
    )
    description = describe_content(
        content=article_description_text(article),
        fallback_title=article["source"]["title"],
        default_tags=["web_page"],
    )
    meta = artifact_index(saved=saved, description=description)
    return SearchToolResult(
        status="ok",
        artifact_path=meta["raw_path"],
        summary=meta["summary"],
    )



def _build_web_list_artifact(query: str, response: JsonValue) -> WebListArtifact:
    if not isinstance(response, list):
        raise RuntimeError("web_list response must be a list")
    results: list[WebListItem] = []
    for item in response:
        if not isinstance(item, dict):
            continue
        title = _string_value(item.get("title"))
        url = _string_value(item.get("url"))
        snippet = _string_value(item.get("snippet"))
        if not title or not url:
            continue
        results.append({"title": title, "url": url, "snippet": snippet})
    if not results:
        raise RuntimeError("web_list response does not include results")
    return {"query": query, "results": results}


def _build_web_page_artifact(url: str, response: JsonValue) -> ArticleArtifact:
    if not isinstance(response, dict):
        raise RuntimeError("web_page response must be an object")
    docs = response.get("docs")
    if not isinstance(docs, list) or not docs:
        raise RuntimeError("web_page response does not include docs")
    first = docs[0]
    if not isinstance(first, dict):
        raise RuntimeError("web_page response returned invalid doc format")
    title = _string_value(first.get("title")) or "(untitled)"
    content = _string_value(first.get("markdown")) or _string_value(first.get("text"))
    if not content:
        raise RuntimeError("web_page response does not include content")
    return {
        "source": {"url": url, "title": title},
        "content": content,
        "content_char_count": len(content),
    }


def _string_value(value: JsonValue) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""

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
