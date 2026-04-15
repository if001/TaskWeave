from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import discord
from dotenv import load_dotenv

from examples.deep_agent_runtime.artifact_payloads import ArticleArtifact, article_description_text
from examples.deep_agent_runtime.artifact_tools import ArtifactMeta, artifact_index, save_article_artifact
from examples.deep_agent_runtime.content_description import describe_content
from examples.deep_agent_runtime.web_tools import resolve_simple_client_base_url
from runtime_core.infra import get_logger

logger = get_logger("taskweave.examples.discord_url_digest_bot")

_DISCORD_BOT_TOKEN_ENV = "DISCORD_BOT_TOKEN"
_DISCORD_CHANNEL_ID_ENV = "DISCORD_WATCH_CHANNEL_ID"
_MAX_PAGE_CHARS = 16_000
_WEB_REQUEST_TIMEOUT_SECONDS = 20.0
_URL_RE = re.compile(r"https?://[^\s<>\]\)\"']+")

_AGENT_ID = "ao"


class UrlDigestService:
    def __init__(self) -> None:
        self._base_url = resolve_simple_client_base_url()
        if self._base_url is None:
            raise RuntimeError(
                "Set SIMPLE_CLIENT_BASE_URL to enable URL fetching via web_tools endpoint."
            )
        self._artifact_dir = self._resolve_artifact_dir()

    def _resolve_artifact_dir(self) -> Path:
        root_dir = Path(".state") / _AGENT_ID / "artifacts"
        root_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Artifact root directory: %s", root_dir)
        return root_dir

    async def process_url(self, url: str, message: discord.Message) -> ArtifactMeta:
        page_title, content_text = await asyncio.to_thread(self._fetch_page_content, url)
        return await asyncio.to_thread(
            self._persist_article,
            url,
            page_title,
            content_text[:_MAX_PAGE_CHARS],
            content_text,
            message,
        )

    def _fetch_page_content(self, url: str) -> tuple[str, str]:
        if self._base_url is None:
            raise RuntimeError("SIMPLE_CLIENT_BASE_URL is not configured")
        endpoint = f"{self._base_url}/page"
        req = Request(
            url=endpoint,
            data=json.dumps({"urls": url}).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=_WEB_REQUEST_TIMEOUT_SECONDS) as resp:
                raw_text = resp.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"page endpoint request failed ({exc.code}): {detail}"
            ) from exc
        except (TimeoutError, SocketTimeout) as exc:
            raise RuntimeError("page endpoint request timed out") from exc
        except URLError as exc:
            raise RuntimeError(f"page endpoint request failed: {exc.reason}") from exc

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("page endpoint returned non-JSON payload") from exc
        docs = payload.get("docs") if isinstance(payload, dict) else None
        if not isinstance(docs, list) or not docs:
            raise RuntimeError("page endpoint returned empty docs")
        first = docs[0]
        if not isinstance(first, dict):
            raise RuntimeError("page endpoint returned invalid doc format")

        title = str(first.get("title", "")).strip() or "(untitled)"
        markdown = str(first.get("markdown", "")).strip()
        if not markdown:
            markdown = str(first.get("text", "")).strip()
        if not markdown:
            raise RuntimeError("page endpoint response does not include content")
        return title, markdown

    def _persist_article(
        self,
        url: str,
        page_title: str,
        clipped_content: str,
        content_text: str,
        message: discord.Message,
    ) -> ArtifactMeta:
        created_at = datetime.now(UTC).isoformat()
        payload: ArticleArtifact = {
            "created_at": created_at,
            "source": {
                "url": url,
                "title": page_title,
            },
            "content": clipped_content,
            "content_char_count": len(content_text),
            "saved_for": [
                "next_article_selection",
                "interest_direction_update",
            ],
            "discord_channel_id": message.channel.id,
            "discord_message_id": message.id,
            "discord_author_id": message.author.id,
        }
        saved = save_article_artifact(
            kind="url_digest",
            artifact=payload,
            artifact_dir=self._artifact_dir,
        )
        description = describe_content(
            content=article_description_text(payload),
            fallback_title=page_title,
            default_tags=["url_digest"],
        )
        return artifact_index(saved=saved, description=description)


class DiscordUrlDigestBot(discord.Client):
    def __init__(self, service: UrlDigestService, channel_id: int | None) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self._service = service
        self._watch_channel_id = channel_id

    async def on_ready(self) -> None:
        logger.info("Discord URL digest bot started. user=%s", self.user)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if self._watch_channel_id and message.channel.id != self._watch_channel_id:
            return
        urls = _URL_RE.findall(message.content)
        if not urls:
            return

        for url in urls:
            started = time.perf_counter()
            try:
                async with message.channel.typing():
                    meta = await self._service.process_url(url, message)
                tag_text = " ".join(f"#{tag}" for tag in meta["tags"])
                await message.channel.send(
                    f"要約\n{meta['summary']}\n\nタグ: {tag_text}"
                )
                elapsed = time.perf_counter() - started
                logger.info("Processed URL in %.2fs: %s", elapsed, url)
            except Exception:
                logger.exception("Failed to process URL: %s", url)
                await message.channel.send(f"URLの処理に失敗しました: {url}")


def _require_token() -> str:
    token = os.getenv(_DISCORD_BOT_TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(f"Set {_DISCORD_BOT_TOKEN_ENV} before running this example.")
    return token


def _resolve_watch_channel_id() -> int | None:
    value = os.getenv(_DISCORD_CHANNEL_ID_ENV, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Ignore invalid %s: %s", _DISCORD_CHANNEL_ID_ENV, value)
        return None


async def _run() -> None:
    load_dotenv()
    service = UrlDigestService()
    client = DiscordUrlDigestBot(
        service=service, channel_id=_resolve_watch_channel_id()
    )
    await client.start(_require_token())


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
