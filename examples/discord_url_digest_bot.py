from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import gettempdir
from socket import timeout as SocketTimeout
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import discord
from dotenv import load_dotenv

from examples.deep_agent_runtime.ollama_client import get_ollama_client
from examples.deep_agent_runtime.web_tools import resolve_simple_client_base_url
from runtime_core.infra import get_logger

logger = get_logger("taskweave.examples.discord_url_digest_bot")

_DISCORD_BOT_TOKEN_ENV = "DISCORD_BOT_TOKEN"
_MODEL_ENV = "MODEL_NAME"
_DEFAULT_MODEL = "gpt-oss:20b"
_ARTIFACT_ROOT_ENV = "DEEPAGENT_ARTIFACT_DIR"
_DISCORD_CHANNEL_ID_ENV = "DISCORD_WATCH_CHANNEL_ID"
_MAX_PAGE_CHARS = 16_000
_WEB_REQUEST_TIMEOUT_SECONDS = 20.0
_URL_RE = re.compile(r"https?://[^\s<>\]\)\"']+")


@dataclass(slots=True)
class ArticleSummary:
    title: str
    summary: str
    tags: list[str]


class UrlDigestService:
    def __init__(self) -> None:
        model_name = os.getenv(_MODEL_ENV, _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
        self._model = get_ollama_client(model_name)
        self._base_url = resolve_simple_client_base_url()
        if self._base_url is None:
            raise RuntimeError(
                "Set SIMPLE_CLIENT_BASE_URL to enable URL fetching via web_tools endpoint."
            )
        self._artifact_backend = self._build_artifact_backend()

    def _build_artifact_backend(self):
        from deepagents.backends import FilesystemBackend

        root = os.getenv(_ARTIFACT_ROOT_ENV, "").strip()
        root_dir = Path(root) if root else Path(gettempdir()) / "taskweave_deepagent_artifacts"
        root_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Artifact root directory: %s", root_dir)
        return FilesystemBackend(root_dir=str(root_dir), virtual_mode=True)

    async def process_url(self, url: str, message: discord.Message) -> ArticleSummary:
        title, content_text = await asyncio.to_thread(self._fetch_page_content, url)
        clipped = content_text[:_MAX_PAGE_CHARS]
        analyzed = await self._summarize_and_tag(url=url, title=title, content=clipped)
        await asyncio.to_thread(
            self._persist_article,
            url,
            analyzed,
            content_text,
            message,
        )
        return analyzed

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
            raise RuntimeError(f"page endpoint request failed ({exc.code}): {detail}") from exc
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

    async def _summarize_and_tag(self, *, url: str, title: str, content: str) -> ArticleSummary:
        prompt = (
            "次の記事を日本語で要約してください。必ずJSONのみ返してください。\\n"
            "要件:\\n"
            '- "summary": 3〜5文の要約\\n'
            '- "tags": 3〜7個の短いタグ配列（日本語、重複禁止）\\n\\n'
            f"URL: {url}\\n"
            f"TITLE: {title}\\n"
            f"CONTENT:\\n{content}"
        )
        result = await self._model.ainvoke([{"role": "user", "content": prompt}])
        raw = str(getattr(result, "content", "")).strip()
        payload = _parse_json(raw)
        summary = str(payload.get("summary", "")).strip()
        tags_raw = payload.get("tags")
        if not isinstance(tags_raw, list):
            tags_raw = []
        tags = [str(t).strip() for t in tags_raw if str(t).strip()]
        if not summary:
            summary = "要約の生成に失敗しました。"
        if not tags:
            tags = ["未分類"]
        return ArticleSummary(title=title, summary=summary, tags=tags)

    def _persist_article(
        self,
        url: str,
        analyzed: ArticleSummary,
        content_text: str,
        message: discord.Message,
    ) -> None:
        created_at = datetime.now(UTC).isoformat()
        date_slug = datetime.now(UTC).strftime("%Y%m%d")
        record_id = f"article_{date_slug}_{uuid.uuid4().hex[:10]}"
        payload = {
            "id": record_id,
            "created_at": created_at,
            "source": {
                "url": url,
                "title": analyzed.title,
                "discord_channel_id": message.channel.id,
                "discord_message_id": message.id,
                "discord_author_id": message.author.id,
            },
            "summary": analyzed.summary,
            "tags": analyzed.tags,
            "content": content_text,
            "content_char_count": len(content_text),
            "saved_for": [
                "next_article_selection",
                "interest_direction_update",
            ],
        }
        virtual_path = f"/artifact/articles/{date_slug}/{record_id}.json"
        write_result = self._artifact_backend.write(
            virtual_path,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )
        if write_result.error:
            raise RuntimeError(write_result.error)


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
                    result = await self._service.process_url(url, message)
                tag_text = " ".join(f"#{tag}" for tag in result.tags)
                await message.channel.send(
                    f"要約: {result.summary}\\nタグ: {tag_text}"
                )
                elapsed = time.perf_counter() - started
                logger.info("Processed URL in %.2fs: %s", elapsed, url)
            except Exception:
                logger.exception("Failed to process URL: %s", url)
                await message.channel.send(
                    f"URLの処理に失敗しました: {url}"
                )


def _parse_json(raw: str) -> dict[str, object]:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()
    try:
        decoded = json.loads(candidate)
        if isinstance(decoded, dict):
            return decoded
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            decoded = json.loads(match.group(0))
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass
    return {}


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
    client = DiscordUrlDigestBot(service=service, channel_id=_resolve_watch_channel_id())
    await client.start(_require_token())


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
