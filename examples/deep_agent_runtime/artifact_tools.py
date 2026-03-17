from __future__ import annotations

import json
import os
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from runtime_core.types import JsonValue
from examples.deep_agent_runtime.ollama_client import get_ollama_client

_ARTIFACT_MODEL_ENV = "ARTIFACT_OLLAMA_MODEL"
_ARTIFACT_OLLAMA_BASE_URL = "ARTIFACT_OLLAMA_BASE_URL"
_ARTIFACT_MAX_INPUT_CHARS = 6000


class ArtifactMeta(TypedDict):
    id: str
    kind: str
    title: str
    summary: str
    tags: list[str]
    raw_path: str


@dataclass(slots=True)
class ArtifactSearchResult:
    meta: ArtifactMeta
    score: int


def artifact_save(
    *,
    kind: str,
    raw: JsonValue,
    artifact_dir: Path,
    title: str | None = None,
    summary: str | None = None,
    tags: list[str] | str | None = None,
) -> ArtifactMeta:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = uuid4().hex
    target_dir = artifact_dir / artifact_id
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_path = target_dir / "raw.json"
    raw_path.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    resolved_title = title or _generate_title(kind, raw)
    resolved_summary = summary or _generate_summary(kind, raw)
    resolved_tags = (
        _normalize_tags(tags) if tags is not None else _generate_tags(kind, raw)
    )
    meta: ArtifactMeta = {
        "id": artifact_id,
        "kind": kind,
        "title": resolved_title,
        "summary": resolved_summary,
        "tags": resolved_tags,
        "raw_path": str(raw_path),
    }
    meta_path = target_dir / "meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def artifact_search(
    *,
    query: str,
    artifact_dir: Path,
    limit: int = 5,
) -> list[ArtifactMeta]:
    if not query.strip() or not artifact_dir.exists():
        return []
    tokens = _tokenize(query)
    results: list[ArtifactSearchResult] = []
    for meta_path in artifact_dir.rglob("meta.json"):
        meta = _read_meta(meta_path)
        if meta is None:
            continue
        score = _score_meta(meta, tokens)
        if score > 0:
            results.append(ArtifactSearchResult(meta=meta, score=score))
    results.sort(key=lambda item: item.score, reverse=True)
    return [item.meta for item in results[: max(limit, 1)]]


def _read_meta(path: Path) -> ArtifactMeta | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    required = {"id", "kind", "title", "summary", "tags", "raw_path"}
    if not required.issubset(data.keys()):
        return None
    tags = data.get("tags")
    if not isinstance(tags, list):
        tags = []
    return ArtifactMeta(
        id=str(data.get("id", "")),
        kind=str(data.get("kind", "")),
        title=str(data.get("title", "")),
        summary=str(data.get("summary", "")),
        tags=[str(tag) for tag in tags if str(tag).strip()],
        raw_path=str(data.get("raw_path", "")),
    )


def _score_meta(meta: ArtifactMeta, tokens: list[str]) -> int:
    haystack = " ".join([meta["title"], meta["summary"], " ".join(meta["tags"])])
    haystack_lower = haystack.lower()
    score = 0
    for token in tokens:
        if token and token in haystack_lower:
            score += 1
    return score


def _normalize_tags(tags: list[str] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw_tags = [item.strip() for item in tags.split(",")]
    else:
        raw_tags = [str(item).strip() for item in tags]
    return [tag for tag in raw_tags if tag]


def _summarize_payload(payload: JsonValue) -> str:
    rendered = (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )
    summary = " ".join(rendered.split())
    return summary[:500]


def _generate_title(kind: str, payload: JsonValue) -> str:
    prompt = (
        "Create a concise title (max 8 words) for the following artifact.\n"
        f"Kind: {kind}\n"
        f"Content:\n{_render_payload(payload)}\n"
        "Return title only."
    )
    response = _call_ollama(prompt)
    return response or _fallback_title(kind)


def _generate_summary(kind: str, payload: JsonValue) -> str:
    prompt = (
        "Summarize the following artifact in 1-2 sentences (max 300 chars).\n"
        f"Kind: {kind}\n"
        f"Content:\n{_render_payload(payload)}\n"
        "Return summary only."
    )
    response = _call_ollama(prompt)
    return response or _summarize_payload(payload)


def _generate_tags(kind: str, payload: JsonValue) -> list[str]:
    prompt = (
        "Generate 3-8 short tags for the following artifact.\n"
        f"Kind: {kind}\n"
        f"Content:\n{_render_payload(payload)}\n"
        "Return comma-separated tags only."
    )
    response = _call_ollama(prompt)
    if not response:
        return []
    return _normalize_tags(response)


def _render_payload(payload: JsonValue) -> str:
    rendered = (
        payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    )
    trimmed = " ".join(rendered.split())
    return trimmed[:_ARTIFACT_MAX_INPUT_CHARS]


def _call_ollama(prompt: str) -> str:
    try:
        model = _get_artifact_model()
        message = model.invoke([{"role": "user", "content": prompt}])
    except Exception:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


@lru_cache(maxsize=1)
def _get_artifact_model():
    model_name = os.getenv(_ARTIFACT_MODEL_ENV, "gpt-oss:20b")
    base_url = os.getenv(_ARTIFACT_OLLAMA_BASE_URL, "gpt-oss:20b")
    return get_ollama_client(model_name=model_name, base_url=base_url)


def _fallback_title(kind: str) -> str:
    safe_kind = kind.strip() or "artifact"
    return f"{safe_kind} artifact"


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = []
    current = []
    for ch in lowered:
        if ch.isalnum() or ch in {"-", "_"}:
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens
