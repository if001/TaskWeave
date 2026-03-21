from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore

from examples.deep_agent_runtime.ollama_client import get_ollama_client
from runtime_core.infra import get_logger
from runtime_core.types import JsonValue, ensure_json_value

_ARTIFACT_MODEL_ENV = "ARTIFACT_OLLAMA_MODEL"
_ARTIFACT_EMBED_MODEL_ENV = "ARTIFACT_OLLAMA_EMBED_MODEL"
_ARTIFACT_RERANK_MODEL_ENV = "ARTIFACT_OLLAMA_RERANK_MODEL"
_ARTIFACT_PG_DSN_ENV = "ARTIFACT_PG_DSN"
_ARTIFACT_PG_SCHEMA_ENV = "ARTIFACT_PG_SCHEMA"
_ARTIFACT_PG_TABLE_ENV = "ARTIFACT_PG_TABLE"
_ARTIFACT_MAX_INPUT_CHARS = 6000
_ARTIFACT_RERANK_MAX_CANDIDATES = 20
_ARTIFACT_VECTOR_MULTIPLIER = 5

logger = get_logger(__name__)


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


@dataclass(slots=True)
class _ArtifactCandidate:
    meta: ArtifactMeta
    vector_score: float


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

    _ = title
    _ = summary
    _ = tags
    resolved_title = _generate_title(kind, raw)
    resolved_summary = _generate_summary(kind, raw)
    resolved_tags = _generate_tags(kind, raw)
    meta: ArtifactMeta = {
        "id": artifact_id,
        "kind": kind,
        "title": resolved_title,
        "summary": resolved_summary,
        "tags": resolved_tags,
        "raw_path": str(raw_path),
    }
    _store_meta_in_vectorstore(meta)
    return meta


def artifact_search(
    *,
    query: str,
    artifact_dir: Path,
    limit: int = 5,
) -> list[ArtifactMeta]:
    _ = artifact_dir
    if not query.strip():
        return []

    vectorstore = _get_vectorstore()
    if vectorstore is None:
        return []

    candidates = _search_vectorstore(vectorstore, query, limit)
    if not candidates:
        return []
    reranked = _rerank_candidates(query, candidates)
    return [item.meta for item in reranked[: max(limit, 1)]]


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
    return get_ollama_client(model_name=model_name)


@lru_cache(maxsize=1)
def _get_rerank_model():
    model_name = os.getenv(_ARTIFACT_RERANK_MODEL_ENV, "gpt-oss:20b")
    return get_ollama_client(model_name=model_name)


@lru_cache(maxsize=1)
def _get_embeddings() -> OllamaEmbeddings:
    model_name = os.getenv(_ARTIFACT_EMBED_MODEL_ENV, "nomic-embed-text")
    base_url = os.getenv("OLLAMA_BASE_URL", "")
    return OllamaEmbeddings(model=model_name, base_url=base_url)


@lru_cache(maxsize=1)
def _get_vectorstore():
    dsn = os.getenv(_ARTIFACT_PG_DSN_ENV, "").strip()
    if not dsn:
        return None
    schema = os.getenv(_ARTIFACT_PG_SCHEMA_ENV, "app")
    table = os.getenv(_ARTIFACT_PG_TABLE_ENV, "documents")
    pg_engine = PGEngine.from_connection_string(url=dsn)
    return PGVectorStore.create_sync(
        engine=pg_engine,
        table_name=table,
        schema_name=schema,
        embedding_service=_get_embeddings(),
    )


def _store_meta_in_vectorstore(meta: ArtifactMeta) -> None:
    vectorstore = _get_vectorstore()
    if vectorstore is None:
        return
    document = Document(
        page_content=_render_meta_text(meta),
        metadata={
            "id": meta["id"],
            "kind": meta["kind"],
            "title": meta["title"],
            "summary": meta["summary"],
            "tags": meta["tags"],
            "raw_path": meta["raw_path"],
        },
    )
    try:
        vectorstore.add_documents([document], ids=[meta["id"]])
    except Exception as exc:
        logger.warning("vectorstore upsert failed: %s", exc)


def _render_meta_text(meta: ArtifactMeta) -> str:
    tags = ", ".join(meta["tags"])
    return (
        f"kind: {meta['kind']}\n"
        f"title: {meta['title']}\n"
        f"summary: {meta['summary']}\n"
        f"tags: {tags}\n"
    )


def _search_vectorstore(
    vectorstore, query: str, limit: int
) -> list[_ArtifactCandidate]:
    k = max(limit, 1) * _ARTIFACT_VECTOR_MULTIPLIER
    try:
        results = vectorstore.similarity_search_with_score(query, k=k)
    except Exception as exc:
        logger.warning("vectorstore search failed: %s", exc)
        return []

    candidates: list[_ArtifactCandidate] = []
    for doc, score in results:
        meta = _document_to_meta(doc)
        if meta is None:
            continue
        candidates.append(_ArtifactCandidate(meta=meta, vector_score=float(score)))
    return candidates


def _document_to_meta(doc: Document) -> ArtifactMeta | None:
    data = doc.metadata
    if not isinstance(data, dict):
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


def _rerank_candidates(
    query: str, candidates: list[_ArtifactCandidate]
) -> list[_ArtifactCandidate]:
    if not candidates:
        return []
    limited = candidates[:_ARTIFACT_RERANK_MAX_CANDIDATES]
    scored = _rerank_with_ollama(query, limited)
    if scored is None:
        return _rerank_with_keywords(query, limited)
    return sorted(
        limited,
        key=lambda item: (scored.get(item.meta["id"], 0.0), item.vector_score),
        reverse=True,
    )


def _rerank_with_keywords(
    query: str, candidates: list[_ArtifactCandidate]
) -> list[_ArtifactCandidate]:
    tokens = _tokenize(query)
    return sorted(
        candidates,
        key=lambda item: (_score_meta(item.meta, tokens), item.vector_score),
        reverse=True,
    )


def _rerank_with_ollama(
    query: str, candidates: list[_ArtifactCandidate]
) -> dict[str, float] | None:
    prompt_items = [
        {
            "id": item.meta["id"],
            "title": item.meta["title"],
            "summary": item.meta["summary"],
            "tags": item.meta["tags"],
            "kind": item.meta["kind"],
        }
        for item in candidates
    ]
    payload = json.dumps(prompt_items, ensure_ascii=False)
    prompt = (
        "You are a reranking model. Score each candidate for relevance to the query.\n"
        f"Query: {query}\n"
        f"Candidates: {payload}\n"
        "Return a JSON array of objects: [{\"id\": \"...\", \"score\": 0-10}, ...]."
    )
    response = _call_rerank_model(prompt)
    if not response:
        return None
    try:
        parsed: object = json.loads(response)
    except json.JSONDecodeError:
        return None
    coerced = ensure_json_value(parsed)
    if coerced is None or not isinstance(coerced, list):
        return None
    scores: dict[str, float] = {}
    for item in coerced:
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id")
        raw_score = item.get("score")
        if not isinstance(raw_id, str):
            continue
        if isinstance(raw_score, (int, float)):
            scores[raw_id] = float(raw_score)
    return scores or None


def _call_rerank_model(prompt: str) -> str:
    try:
        model = _get_rerank_model()
        message = model.invoke([{"role": "user", "content": prompt}])
    except Exception:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


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
