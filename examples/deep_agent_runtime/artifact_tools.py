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

from .artifact_payloads import ArticleArtifact, ArticleArtifactKind, WebListArtifact
from .content_description import ContentDescription
from runtime_core.infra import get_logger
from runtime_core.types import ensure_json_value

_ARTIFACT_EMBED_MODEL_ENV = "ARTIFACT_OLLAMA_EMBED_MODEL"
_ARTIFACT_RERANK_MODEL_ENV = "ARTIFACT_OLLAMA_RERANK_MODEL"
_ARTIFACT_PG_DSN_ENV = "ARTIFACT_PG_DSN"
_ARTIFACT_PG_SCHEMA_ENV = "ARTIFACT_PG_SCHEMA"
_ARTIFACT_PG_TABLE_ENV = "ARTIFACT_PG_TABLE"
_ARTIFACT_RERANK_MAX_CANDIDATES = 20
_ARTIFACT_VECTOR_MULTIPLIER = 5

logger = get_logger(__name__)


class SavedArtifact(TypedDict):
    id: str
    kind: str
    raw_path: str


class ArtifactMeta(TypedDict):
    id: str
    kind: str
    title: str
    summary: str
    tags: list[str]
    raw_path: str


@dataclass(slots=True)
class _ArtifactCandidate:
    meta: ArtifactMeta
    vector_score: float


def save_article_artifact(
    *,
    kind: ArticleArtifactKind,
    artifact: ArticleArtifact,
    artifact_dir: Path,
) -> SavedArtifact:
    return _write_artifact(kind=kind, payload=artifact, artifact_dir=artifact_dir)


def save_web_list_artifact(
    *,
    artifact: WebListArtifact,
    artifact_dir: Path,
) -> SavedArtifact:
    return _write_artifact(kind="web_list", payload=artifact, artifact_dir=artifact_dir)


def artifact_index(
    *,
    saved: SavedArtifact,
    description: ContentDescription,
) -> ArtifactMeta:
    meta = ArtifactMeta(
        id=saved["id"],
        kind=saved["kind"],
        title=description["title"],
        summary=description["summary"],
        tags=description["tags"],
        raw_path=saved["raw_path"],
    )
    _store_meta_in_vectorstore(meta)
    return meta


def artifact_index_path(
    *,
    kind: str,
    raw_path: Path,
    description: ContentDescription,
) -> ArtifactMeta:
    saved = SavedArtifact(id=raw_path.stem, kind=kind, raw_path=str(raw_path))
    return artifact_index(saved=saved, description=description)


def artifact_search(*, query: str, limit: int = 5) -> list[ArtifactMeta]:
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


def _write_artifact(*, kind: str, payload: ArticleArtifact | WebListArtifact, artifact_dir: Path) -> SavedArtifact:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = uuid4().hex
    raw_path = artifact_dir / f"{artifact_id}.json"
    raw_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return SavedArtifact(id=artifact_id, kind=kind, raw_path=str(raw_path))


@lru_cache(maxsize=1)
def _get_embeddings() -> OllamaEmbeddings:
    model_name = os.getenv(_ARTIFACT_EMBED_MODEL_ENV, "nomic-embed-text")
    base_url = os.getenv("OLLAMA_BASE_URL_LOCAL", "")
    return OllamaEmbeddings(model=model_name, base_url=base_url)


@lru_cache(maxsize=1)
def _get_vectorstore() -> PGVectorStore | None:
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
    vectorstore: PGVectorStore,
    query: str,
    limit: int,
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
    raw_tags = data.get("tags")
    tags = [str(tag) for tag in raw_tags] if isinstance(raw_tags, list) else []
    return ArtifactMeta(
        id=str(data.get("id", "")),
        kind=str(data.get("kind", "")),
        title=str(data.get("title", "")),
        summary=str(data.get("summary", "")),
        tags=[tag for tag in tags if tag.strip()],
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


def _rerank_candidates(
    query: str,
    candidates: list[_ArtifactCandidate],
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
    query: str,
    candidates: list[_ArtifactCandidate],
) -> list[_ArtifactCandidate]:
    tokens = _tokenize(query)
    return sorted(
        candidates,
        key=lambda item: (_score_meta(item.meta, tokens), item.vector_score),
        reverse=True,
    )


def _rerank_with_ollama(
    query: str,
    candidates: list[_ArtifactCandidate],
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
        'Return a JSON array of objects: [{"id": "...", "score": 0-10}, ...].'
    )
    response = _call_rerank_model(prompt)
    if not response:
        return None
    try:
        parsed = json.loads(response)
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
    return _invoke_model(prompt, model_name=_rerank_model_name())


def _invoke_model(prompt: str, *, model_name: str) -> str:
    try:
        from .ollama_client import get_ollama_client
        model = get_ollama_client(model_name=model_name)
        message = model.invoke([{"role": "user", "content": prompt}])
    except Exception:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


@lru_cache(maxsize=1)
def _rerank_model_name() -> str:
    return os.getenv(_ARTIFACT_RERANK_MODEL_ENV, "gpt-oss:20b")


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    current: list[str] = []
    for ch in lowered:
        if ch.isalnum() or ch in {"-", "_"}:
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens
