from __future__ import annotations

import json
from pathlib import Path

from examples.deep_agent_runtime import web_tools


def _fixed_description(*, content: str, fallback_title: str, default_tags: list[str] | None = None) -> dict[str, object]:
    _ = (content, fallback_title, default_tags)
    return {"title": "fixed-title", "summary": "fixed-summary", "tags": ["tag-a", "tag-b"]}


def test_web_list_and_store_artifact(tmp_path: Path, monkeypatch) -> None:
    def fake_post_json(url: str, payload: dict[str, object]) -> tuple[object, str | None]:
        _ = (url, payload)
        return (
            {
                "query": "python memory",
                "k": 5,
                "results": [
                    {
                        "rank": 1,
                        "title": "Result 1",
                        "url": "https://example.com/1",
                        "snippet": "snippet 1",
                        "published_date": "2026-04-01",
                    },
                    {
                        "rank": 2,
                        "title": "Result 2",
                        "url": "https://example.com/2",
                        "snippet": "snippet 2",
                    },
                ],
            },
            None,
        )

    monkeypatch.setattr(web_tools, "_post_json", fake_post_json)
    monkeypatch.setattr(web_tools, "describe_content", _fixed_description)

    result = web_tools.web_list_and_store_artifact(
        query="python memory",
        k=5,
        base_url="http://dummy",
        artifact_dir=tmp_path,
    )

    assert result["status"] == "ok"
    assert result["summary"] == "fixed-summary"

    raw_path = Path(result["artifact_path"])
    assert raw_path.exists()
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    assert payload["query"] == "python memory"
    assert payload["k"] == 5
    assert len(payload["results"]) == 2
    assert payload["results"][0]["rank"] == 1
    assert payload["results"][0]["published_date"] == "2026-04-01"


def test_web_page_and_store_artifact(tmp_path: Path, monkeypatch) -> None:
    def fake_post_json(url: str, payload: dict[str, object]) -> tuple[object, str | None]:
        _ = (url, payload)
        return (
            {
                "docs": [
                    {
                        "title": "Doc title",
                        "url": "https://example.com/doc",
                        "markdown": "first paragraph\nsecond paragraph",
                    }
                ]
            },
            None,
        )

    monkeypatch.setattr(web_tools, "_post_json", fake_post_json)
    monkeypatch.setattr(web_tools, "describe_content", _fixed_description)

    result = web_tools.web_page_and_store_artifact(
        url="https://example.com/doc",
        base_url="http://dummy",
        artifact_dir=tmp_path,
    )

    assert result["status"] == "ok"
    assert result["summary"] == "fixed-summary"

    raw_path = Path(result["artifact_path"])
    assert raw_path.exists()
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    assert payload["source"]["url"] == "https://example.com/doc"
    assert payload["source"]["title"] == "Doc title"
    assert "paragraph" in payload["content"]
