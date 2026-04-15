from __future__ import annotations
import os
from functools import lru_cache
from typing import TypedDict
from .ollama_client import get_ollama_client

_CONTENT_DESCRIPTION_MODEL_ENV = "ARTIFACT_OLLAMA_MODEL"
_CONTENT_DESCRIPTION_OLLAMA_BASE_URL = "ARTIFACT_OLLAMA_BASE_URL"
_CONTENT_INPUT_MAX_CHARS = 6000
_CONTENT_SUMMARY_MAX_CHARS = 400


class ContentDescription(TypedDict):
    title: str
    summary: str
    tags: list[str]


def describe_content(
    *,
    content: str,
    fallback_title: str,
    default_tags: list[str] | None = None,
) -> ContentDescription:
    return {
        "title": _generate_title(content, fallback_title=fallback_title),
        "summary": _generate_summary(content, fallback_title=fallback_title),
        "tags": _generate_tags(
            content,
            fallback_title=fallback_title,
            default_tags=default_tags or [],
        ),
    }


def _generate_title(content: str, *, fallback_title: str) -> str:
    prompt = (
        "Create a concise title (max 8 words) for the following content.\n"
        f"Content:\n{_render_content(content)}\n"
        "Return title only."
    )
    response = _invoke_model(prompt)
    return response or fallback_title.strip() or "content"


def _generate_summary(content: str, *, fallback_title: str) -> str:
    prompt = (
        "Summarize the following content in about 4 short lines (max 400 chars).\n"
        "出力は日本語で行うこと。\n\n"
        f"Content:\n{_render_content(content)}\n"
        "Return summary only."
    )
    raw = _invoke_model(prompt)
    response = _clean_generated_summary(raw)
    return response or _fallback_summary(content, fallback_title=fallback_title)


def _generate_tags(
    content: str,
    *,
    fallback_title: str,
    default_tags: list[str],
) -> list[str]:
    prompt = (
        "Generate 3-8 short tags for the following content.\n"
        f"Content:\n{_render_content(content)}\n"
        "Return comma-separated tags only."
    )
    response = _invoke_model(prompt)
    if not response:
        return _fallback_tags(fallback_title=fallback_title, default_tags=default_tags)
    return _normalize_tags(response)


def _render_content(content: str) -> str:
    return " ".join(content.split())[:_CONTENT_INPUT_MAX_CHARS]


def _fallback_summary(content: str, *, fallback_title: str) -> str:
    summary = _clean_generated_summary(content)
    return summary or fallback_title.strip() or "content"


def _clean_generated_summary(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    if not cleaned:
        return ""
    return cleaned[:_CONTENT_SUMMARY_MAX_CHARS]


def _fallback_tags(*, fallback_title: str, default_tags: list[str]) -> list[str]:
    tags = [*default_tags, *_tokenize(fallback_title)[:3]]
    normalized = _normalize_tags(tags)
    return normalized[:8]


def _normalize_tags(tags: list[str] | str | None) -> list[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw_tags = [item.strip() for item in tags.split(",")]
    else:
        raw_tags = [str(item).strip() for item in tags]
    return [tag for tag in raw_tags if tag]


def _invoke_model(prompt: str) -> str:
    try:
        model_name = _model_name()
        base_url = _ollama_base_url()
        model = get_ollama_client(model_name=model_name, base_url=base_url)
        message = model.invoke([{"role": "user", "content": prompt}])
    except Exception as e:
        print("invoke exception", e)
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


@lru_cache(maxsize=1)
def _model_name() -> str:
    configured = os.getenv(_CONTENT_DESCRIPTION_MODEL_ENV, "").strip()
    if configured:
        return configured
    return "gpt-oss:20b"


@lru_cache(maxsize=1)
def _ollama_base_url() -> str:
    configured = os.getenv(_CONTENT_DESCRIPTION_OLLAMA_BASE_URL, "").strip()
    if configured:
        return configured
    raise ValueError("_CONTENT_DESCRIPTION_OLLAMA_BASE_URL not set")


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
