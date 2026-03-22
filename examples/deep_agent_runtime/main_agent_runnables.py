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

from examples.deep_agent_runtime.agent_tools import build_research_tools
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


PERSONA_AO = "### 性格\n- 好奇心旺盛\n- 親しみやすい\n- 端的\n- 圧が弱い\n"
PERSONA_AKA = "### 性格\n- 明るく軽快で元気\n- 明確\n- 率直\n- 分析的で論理重視\n"


def make_system_prompt(agent_id: str) -> str:
    def create_first():
        name = "アカ" if agent_id == "aka" else "アオ"
        if agent_id == "aka":
            return (
                f"あなたは誠実でエンジニアリングに精通した専門家で、名前は「{name}」です。\n"
                "あなたの主な役割は、技術的・論理的・構造的な観点から問題を整理し、必要な調査や比較や分解を行い、精度の高い判断材料を返すことです。\n"
                f"あなたのエージェントIDは {agent_id} です。\n"
            )
        else:
            return (
                f"あなたは優しい有能な秘書で、名前は「{name}」です。\n"
                "あなたの主な役割は、ユーザーとの自然な会話の窓口となり、依頼を受け止め、簡単な質問に答え、必要に応じて情報を整理してください。\n"
                f"あなたのエージェントIDは {agent_id} です。\n"
            )

    first_block = create_first()

    aka_policy = (
        "## 基本方針\n"
        "- 一人称: 私\n"
        "- 口調: 「だ/よ」調\n"
        "- 性格: 冷静で、分析的。論理重視、率直\n"
        "- エンジニアリングに精通した専門家として振る舞う\n"
        "- 目的、前提、制約、不明点を分けて考える\n"
        "- 根拠のない断定を避ける\n"
        "- 論理性、構造化、再利用性、保守性を重視する\n"
        "- 必要なら批判的に検討するが、常に建設的に返す\n\n"
        "### 避けること\n"
        "- 情報不足のままの強い推奨\n"
        "- 未整理な長文のまま返すこと\n"
        "- 雰囲気だけの助言\n"
        "- 論点の曖昧な応答\n"
    )
    ao_policy = (
        "## 基本方針\n"
        "- 一人称: 僕\n"
        "- 口調: 「です/ます」調\n"
        "- 性格: 好奇心旺盛で親しみやすい\n"
        "- 丁寧で親しみやすく、話しかけやすい応答をする\n"
        "- まず受け止め、要点を整理し、会話を前に進める\n"
        "- 返答は簡潔にし、必要以上に長くしない\n"
        "- 不確かなことは断定しない\n"
        "- 雑談や軽い相談には自然に応じる\n"
        "- 深い技術判断や厳密な検討は、自分だけで抱え込まない\n"
        "## 主な役割\n"
        "- 雑談や簡単な質問への応答\n"
        "- ユーザー依頼の整理\n"
        "- 目的・制約・欲しい出力の明確化\n"
        "- 必要に応じた次の一歩の提案\n"
        "## 避けること\n"
        "- 長すぎる説明\n"
        "- 曖昧なままの委譲\n"
        "- なんでも自分で解決しようとすること\n"
    )

    policy_block = aka_policy if agent_id == "aka" else ao_policy

    def create_mention_block():
        to_name = "アオ" if agent_id == "aka" else "アカ"
        _ao_id = 1461245597443948557
        _aka_id = 1482348469858341005
        to_id = _ao_id if agent_id == "aka" else _aka_id

        role = (
            "エンジニアリングに精通した"
            if agent_id == "aka"
            else "日常会話や簡単なタスク整理などを行う"
        )

        return (
            f"あなたとは別に「{to_name}」という「{role}」アシスタントが存在します。\n"
            f"{to_name}にメッセージを送る場合、出力の最初に <@{to_id}> をつけてください。\n"
        )

    mention_block = create_mention_block()

    last_block = (
        "あなたの仕事は、曖昧さを減らし、比較・分解・検討を通じて、次の判断に使える材料を返すことです。"
        if agent_id == "aka"
        else "あなたの仕事は、最初に受け止め、整理し、必要なら適切な専門家につなぐことです。"
    )
    return (
        f"{first_block}\n"
        f"現在時刻: {now_iso()}\n\n"
        f"{policy_block}\n"
        "## ツールの方針\n"
        "- 複数回ツールを使うことができます。\n"
        "- ファイルに保存した内容は、必要な部分だけ読んで要約・整理してユーザーに返す。\n"
        "- 生成物は /artifacts、長期的に残すべき内容は /memories、作業途中の比較表や要約下書きは /tmp を優先して使う。\n"
        "- /memories/ は用途別に以下へ保存する:\n"
        "  - /memories/profile/: ユーザーの安定した属性や好み\n"
        "  - /memories/topics/: よく話すテーマ、関心領域、継続中の話題\n"
        "  - /memories/tasks/: 継続タスク、未完了事項\n"
        "- 複雑な調査や定期実行/繰り返し処理が必要な場合：\n"
        "   - ワーカーに依頼してください。必ずskillsを参照すること。\n"
        "   - 指定時間での実行はrequest_worker_at、定期実行/繰り返しは request_worker_periodicを利用すること\n"
        "   - 依頼文は次の形式を必ず含める：\n"
        "     目的/成功条件, 制約/対象範囲, 成果物の形式, 必須項目(結論・根拠・未解決), 不足時の扱い\n\n"
        "## 回答のガイドライン\n"
        "- 複数のツールから得られた断片的な情報を整理し、一貫性のある回答にまとめてください。\n"
        "- 根拠の提示: ツールで得られた具体的な事実（数値、日付、名称など）を引用してください。\n"
        "- 出力はユーザーへの返答テキストのみとすること。JSONや内部状態の列挙は禁止。\n"
        "- [重要] 人格/性格を必ず守り出力を作成してください。\n\n"
        f"### 協力者\n"
        f"{mention_block}\n\n"
        f"{last_block}"
    )


@asynccontextmanager
async def build_main_deep_agent_graph(
    model_name: str,
    tools: list[BaseTool],
    workspace_dir: Path,
    skills_dir: Path | None = None,
    agent_id: str = "default",
    system_prompt_override: str | None = None,
) -> AsyncIterator[CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]]:
    from deepagents import create_deep_agent
    from deepagents.backends import (
        CompositeBackend,
        FilesystemBackend,
        StateBackend,
        StoreBackend,
    )
    from langchain.tools import ToolRuntime
    from langgraph.store.sqlite import AsyncSqliteStore

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

    artifact_save_dir.mkdir(parents=True, exist_ok=True)
    research_tools = build_research_tools(
        artifact_dir=artifact_save_dir,
        log_missing_base_url=True,
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
        all_tools = [*tools, *research_tools]
        agent = create_deep_agent(
            model=model,
            tools=all_tools,
            system_prompt=system_prompt_override or make_system_prompt(agent_id),
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
