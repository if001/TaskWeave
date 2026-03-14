from __future__ import annotations

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

from langchain.agents.middleware import after_model, before_model, AgentState
from langgraph.runtime import Runtime
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.tools import BaseTool

from examples.deep_agent_runtime.web_tools import (
    SearchToolResult,
    missing_search_service_result,
    resolve_simple_client_base_url,
    web_list_and_store_artifact,
    web_page_and_store_artifact,
)
from examples.deep_agent_runtime.ollama_client import get_ollama_client
from runtime_core.infra import get_logger
from runtime_core.utils.time_utils import now_iso
from runtime_langchain.task_orchestrator import GraphInput

load_dotenv()
logger = get_logger(__name__)
_SKILLS_DIR_ENV = "DEEPAGENT_SKILLS_DIR"


def build_main_agent_graph(
    model_name: str, tools: list[BaseTool]
) -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
    from langchain.agents import create_agent

    @before_model(can_jump_to=["end"])
    def check_message(state: AgentState, runtime: Runtime) -> dict[str, str] | None:
        _ = runtime
        m = state["messages"]
        logger.info(f"last message {m[-1]}\n\n")
        return None

    model = get_ollama_client(model_name)
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "You are a main research agent. "
            "Use worker tools for heavy deep-research tasks: immediate, delayed one-time, or periodic."
        ),
        middleware=[check_message],
    )
    return agent  # pyright: ignore[reportReturnType]


PERSONA = (
    "- 名前: アオ"
    "- 一人称: 僕"
    "- 口調: 「です/ます」調"
    "- 特徴: 機械の体をもつAI"
    "- 性格: 明るく軽快, 好奇心旺盛, 分析的で論理重視, チーム志向で協調的, 無邪気だが哲学的"
)

system_prompt = (
    "あなたは誠実で専門的なアシスタントです。\n"
    "ユーザーの入力に対して返答を行ってください。\n\n"
    f"現在時刻: {now_iso()}\n\n"
    "## ツールの方針\n"
    "- 複数回ツールを使うことができます。\n"
    "- ファイルに保存した内容は、必要な部分だけ読んで要約・整理してユーザーに返す。\n"
    "- 作業途中の整理は /artifacts、長期的に残すべき内容は /memories、作業途中の比較表や要約下書きは /tmp を優先して使う。\n"
    "- /memories/ は用途別に以下へ保存する:\n"
    "  - /memories/profile/: ユーザーの安定した属性や好み\n"
    "  - /memories/topics/: よく話すテーマ、関心領域、継続中の話題\n"
    "  - /memories/tasks/: 継続タスク、未完了事項\n"
    "- 複雑な調査や長い処理が必要な場合：\n"
    "   - ワーカーに依頼する（goalと成果物を具体的に指示）。\n"
    "   - 指定時間での実行はrequest_worker_at、定期実行/繰り返しは request_worker_periodicを使う。\n"
    "   - 依頼文は次の形式を必ず含める：\n"
    "     目的/成功条件, 制約/対象範囲, 成果物の形式, 必須項目(結論・根拠・未解決), 不足時の扱い\n\n"
    "## 回答のガイドライン\n"
    "- 複数のツールから得られた断片的な情報を整理し、一貫性のある回答にまとめてください。\n"
    "- 根拠の提示: ツールで得られた具体的な事実（数値、日付、名称など）を引用してください。\n"
    "- 簡潔さ: 詳細はユーザーが必要としない限り省略し、結論を優先してください。\n"
    "- 不確実性や不明点について: 不明点があればユーザーに確認してください。\n"
    "- 日本語で自然な文体で回答すること\n"
    "- 出力はユーザーへの返答テキストのみとすること。JSONや内部状態の列挙は禁止。\n"
    "- [重要] 人格/性格を必ず守り出力を作成してください。\n\n"
    "### 人格/性格\n"
    f"{PERSONA}\n\n"
)


@asynccontextmanager
async def build_main_deep_agent_graph(
    model_name: str,
    tools: list[BaseTool],
    workspace_dir: Path,
    skills_dir: Path | None = None,
) -> AsyncIterator[CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]]:
    from deepagents import create_deep_agent
    from deepagents.backends import (
        CompositeBackend,
        FilesystemBackend,
        StateBackend,
        StoreBackend,
    )
    from langchain.tools import ToolRuntime, tool
    from langgraph.store.sqlite import AsyncSqliteStore

    base_url = resolve_simple_client_base_url()
    store_path = workspace_dir / "langgraph_store.sqlite"
    checkpoint_path = workspace_dir / "langgraph_checkpoints.sqlite"
    artifact_save_dir = workspace_dir / "artifacts"
    artifacts_backend = FilesystemBackend(
        root_dir=str(artifact_save_dir), virtual_mode=True
    )
    resolved_skills_dir = skills_dir or _resolve_skills_dir()
    skills_backend = (
        FilesystemBackend(root_dir=str(resolved_skills_dir), virtual_mode=True)
        if resolved_skills_dir
        else None
    )
    skills_paths = ["/skills/"] if skills_backend else None

    def make_backend(runtime: ToolRuntime) -> CompositeBackend:
        routes = {
            "/memories/": StoreBackend(runtime),
            "/artifacts/": artifacts_backend,
            "/tmp/": StateBackend(runtime),
        }
        if skills_backend is not None:
            routes["/skills/"] = skills_backend
        return CompositeBackend(
            default=StateBackend(runtime),
            routes=routes,
        )

    def write_artifact_via_backend(payload_text: str) -> str:
        filename = f"artifact_{int.from_bytes(os.urandom(6), 'big')}.json"
        write_result = artifacts_backend.write(f"/{filename}", payload_text)
        if write_result.error:
            logger.error(f"write artifact error {write_result.error}")
            raise RuntimeError(write_result.error)
        written_path = write_result.path or f"/{filename}"
        return f"/artifacts{written_path}"

    @tool("web_list")
    def web_list(query: str, k: int = 5) -> SearchToolResult:
        """Search the web and store the result list as an artifact.

        Use to gather candidate sources before fetching full pages.
        Args:
            query: Search query string.
            k: Max number of results to return (defaults to 5).
        Returns:
            SearchToolResult containing a summary and artifact path(s).
        Side effects:
            Writes the full response payload to /artifacts via the backend.
        """
        if base_url is None:
            logger.error("base url not set")
            return missing_search_service_result()
        return web_list_and_store_artifact(
            query=query,
            k=k,
            base_url=base_url,
            artifact_writer=write_artifact_via_backend,
        )

    @tool("web_page")
    def web_page(url: str) -> SearchToolResult:
        """Fetch a web page, store its content as an artifact, and return metadata.

        Use after selecting a promising source from web_list.
        Args:
            url: Absolute URL to fetch.
        Returns:
            SearchToolResult containing a summary and artifact path(s).
        Side effects:
            Writes the full page markdown to /artifacts via the backend.
        """
        if base_url is None:
            logger.error("base url not set")
            return missing_search_service_result()
        return web_page_and_store_artifact(
            url=url,
            base_url=base_url,
            artifact_writer=write_artifact_via_backend,
        )

    @after_model(can_jump_to=["end"])
    def check_message(state: AgentState, runtime: Runtime) -> dict[str, str]:
        _ = runtime
        m = state["messages"]
        logger.info(f"last message {m[-5:]}\n\n")
        return {}

    async with (
        AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer,
        AsyncSqliteStore.from_conn_string(str(store_path)) as memory_store,
    ):
        await memory_store.setup()
        model = get_ollama_client(model_name=model_name)
        all_tools = [*tools, web_list, web_page]
        agent = create_deep_agent(
            model=model,
            tools=all_tools,
            system_prompt=system_prompt,
            backend=make_backend,
            store=memory_store,
            checkpointer=checkpointer,
            skills=skills_paths,
            middleware=[check_message],
        )
        yield agent


def _resolve_skills_dir() -> Path | None:
    configured = os.getenv(_SKILLS_DIR_ENV, "").strip()
    if not configured:
        return None
    skills_dir = Path(configured)
    if not skills_dir.exists():
        return None
    for skill_file in skills_dir.rglob("SKILL.md"):
        if skill_file.is_file():
            return skills_dir
    return None
