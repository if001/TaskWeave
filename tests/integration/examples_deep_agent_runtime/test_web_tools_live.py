from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from examples.deep_agent_runtime import web_tools
from runtime_core.types import JsonValue
from dotenv import load_dotenv

load_dotenv()


def _base_url_or_skip() -> str:
    base_url = os.getenv("SIMPLE_CLIENT_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        pytest.skip("Set SIMPLE_CLIENT_BASE_URL to run live web integration tests.")
    return base_url


def _fixed_description(
    *,
    content: str,
    fallback_title: str,
    default_tags: list[str] | None = None,
) -> dict[str, object]:
    _ = (content, fallback_title, default_tags)
    return {"title": "live-title", "summary": "live-summary", "tags": ["live", "test"]}


def test_live_list_endpoint_contract() -> None:
    base_url = _base_url_or_skip()
    payload: dict[str, JsonValue] = {"q": "python", "k": 3}

    response, error = web_tools._post_json(url=f"{base_url}/list", payload=payload)  # pyright: ignore[reportPrivateUsage]
    assert error is None
    assert isinstance(response, dict)
    assert isinstance(response.get("query"), str)
    assert isinstance(response.get("k"), int)
    results = response.get("results")
    assert isinstance(results, list)
    if results:
        first = results[0]
        assert isinstance(first, dict)
        assert isinstance(first.get("rank"), int)
        assert isinstance(first.get("title"), str)
        assert isinstance(first.get("url"), str)


def test_live_web_list_and_store_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = _base_url_or_skip()
    monkeypatch.setattr(web_tools, "describe_content", _fixed_description)
    response, error = web_tools._post_json(  # pyright: ignore[reportPrivateUsage]
        url=f"{base_url}/list",
        payload={"q": "python runtime", "k": 3},
    )
    assert error is None
    if isinstance(response, dict):
        raw_results = response.get("results")
        assert isinstance(raw_results, list)
        assert raw_results, "live /list returned empty results for query=python runtime"

    result = web_tools.web_list_and_store_artifact(
        query="python runtime",
        k=3,
        base_url=base_url,
        artifact_dir=tmp_path,
    )
    assert result["status"] == "ok"
    assert result["summary"] == "live-summary"
    raw_path = Path(result["artifact_path"])
    assert raw_path.exists()
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    assert isinstance(payload.get("query"), str)
    assert isinstance(payload.get("k"), int)
    assert isinstance(payload.get("results"), list)


def test_live_web_page_and_store_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = _base_url_or_skip()
    test_url = os.getenv("WEB_TOOLS_TEST_URL", "https://example.com").strip()
    monkeypatch.setattr(web_tools, "describe_content", _fixed_description)

    result = web_tools.web_page_and_store_artifact(
        url=test_url,
        base_url=base_url,
        artifact_dir=tmp_path,
    )

    assert result["status"] == "ok"
    assert result["summary"] == "live-summary"
    raw_path = Path(result["artifact_path"])
    assert raw_path.exists()
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    source = payload.get("source")
    assert isinstance(source, dict)
    assert isinstance(source.get("url"), str)
    assert isinstance(source.get("title"), str)
    assert isinstance(payload.get("content"), str)
    assert isinstance(payload.get("content_char_count"), int)
