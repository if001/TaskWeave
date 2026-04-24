from __future__ import annotations

from typing import Literal, TypedDict

from runtime_core.types import JsonValue

ArtifactKind = Literal["url_digest", "web_page", "web_list"]
ArticleArtifactKind = Literal["url_digest", "web_page"]


class ArticleSource(TypedDict):
    url: str
    title: str


class ArticleArtifactBase(TypedDict):
    source: ArticleSource
    content: str
    content_char_count: int


class ArticleArtifact(ArticleArtifactBase, total=False):
    created_at: str
    saved_for: list[str]
    discord_channel_id: int
    discord_message_id: int
    discord_author_id: int


class WebListItemBase(TypedDict):
    rank: int
    title: str
    url: str


class WebListItem(WebListItemBase, total=False):
    snippet: str
    published_date: str


class WebListArtifact(TypedDict):
    query: str
    k: int
    results: list[WebListItem]


def parse_article_artifact(raw: JsonValue) -> ArticleArtifact | None:
    if not isinstance(raw, dict):
        return None
    source = raw.get("source")
    content = raw.get("content")
    content_char_count = raw.get("content_char_count")
    if not isinstance(source, dict):
        return None
    title = source.get("title")
    url = source.get("url")
    if not isinstance(title, str) or not title.strip():
        return None
    if not isinstance(url, str) or not url.strip():
        return None
    if not isinstance(content, str) or not content.strip():
        return None
    if not isinstance(content_char_count, int):
        return None

    article: ArticleArtifact = {
        "source": {"title": title.strip(), "url": url.strip()},
        "content": content,
        "content_char_count": content_char_count,
    }
    created_at = raw.get("created_at")
    if isinstance(created_at, str) and created_at.strip():
        article["created_at"] = created_at.strip()
    saved_for = raw.get("saved_for")
    if isinstance(saved_for, list):
        normalized_saved_for = [item.strip() for item in saved_for if isinstance(item, str) and item.strip()]
        if normalized_saved_for:
            article["saved_for"] = normalized_saved_for
    for field in ("discord_channel_id", "discord_message_id", "discord_author_id"):
        value = raw.get(field)
        if isinstance(value, int):
            article[field] = value
    return article


def parse_web_list_artifact(raw: JsonValue) -> WebListArtifact | None:
    if not isinstance(raw, dict):
        return None
    query = raw.get("query")
    raw_k = raw.get("k")
    results = raw.get("results")
    if not isinstance(query, str) or not query.strip():
        return None
    if not isinstance(raw_k, int):
        return None
    if not isinstance(results, list):
        return None

    normalized_results: list[WebListItem] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        rank = item.get("rank")
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("snippet")
        published_date = item.get("published_date")
        if not isinstance(rank, int):
            continue
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.strip():
            continue
        if not isinstance(snippet, str):
            snippet = ""
        normalized_item: WebListItem = {
            "rank": rank,
            "title": title.strip(),
            "url": url.strip(),
            "snippet": snippet.strip(),
        }
        if isinstance(published_date, str) and published_date.strip():
            normalized_item["published_date"] = published_date.strip()
        normalized_results.append(normalized_item)
    if not normalized_results:
        return None
    return {"query": query.strip(), "k": raw_k, "results": normalized_results}


def article_description_text(article: ArticleArtifact) -> str:
    return "\n".join(
        part
        for part in [
            article["source"]["title"],
            article["source"]["url"],
            article["content"],
        ]
        if part
    )


def web_list_description_text(payload: WebListArtifact) -> str:
    sections: list[str] = [payload["query"]]
    for item in payload["results"][:8]:
        published_date = item.get("published_date", "")
        sections.append(
            "\n".join(
                part for part in [item["title"], item.get("snippet", ""), published_date, item["url"]]
                if part
            )
        )
    return "\n\n".join(section for section in sections if section)
