from __future__ import annotations

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false

import os
from pathlib import Path
from tempfile import gettempdir

from langchain_core.messages import AnyMessage
from langgraph.graph.state import CompiledStateGraph, END, StateGraph

from .agent_tools import build_research_tools
from .ollama_client import get_ollama_client
from runtime_core.types import Message
from runtime_langchain.task_orchestrator import GraphInput


def build_worker_agent_graph(
    use_real_agent: bool,
    backend: str,
    model_name: str,
    workspace_dir: Path,
) -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
    if not use_real_agent:
        return _build_echo_worker_graph()
    if backend == "deepagent":
        return _build_deepagent_worker_graph(
            model_name=model_name,
            workspace_dir=workspace_dir,
        )
    return _build_langchain_worker_graph(model_name=model_name)


def resolve_deepagent_artifact_dir(env_name: str) -> Path:
    configured = os.getenv(env_name, "").strip()
    if configured:
        path = Path(configured)
    else:
        path = Path(gettempdir()) / "taskweave_deepagent_artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_echo_worker_graph() -> CompiledStateGraph[
    GraphInput, None, GraphInput, GraphInput
]:
    def _respond(state: GraphInput) -> GraphInput:
        query = _extract_last_message_text(state["messages"])
        return GraphInput(
            messages=[
                {
                    "role": "assistant",
                    "content": f"[mock worker-agent] deep researched: {query}",
                }
            ],
        )

    graph = StateGraph(GraphInput)
    graph.add_node("worker", _respond)
    graph.set_entry_point("worker")
    graph.add_edge("worker", END)
    return graph.compile()


def _build_langchain_worker_graph(
    model_name: str,
) -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
    from langchain.agents import create_agent

    model = get_ollama_client(model_name=model_name)
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a concise worker agent specialized in heavy deep research.",
    )
    return agent  # pyright: ignore[reportReturnType]


def _extract_last_message_text(messages: list[Message | AnyMessage]) -> str:
    if not messages:
        return ""
    message = messages[-1]
    if isinstance(message, dict):
        return str(message.get("content", "")).strip()
    return str(getattr(message, "content", "")).strip()


system_prompt = (
    "あなたはworker agentです。\n"
    "main agentから依頼文が入力されます。依頼に従い適切な出力を行ってください。\n"
    "依頼文には以下が含まれます。\n"
    "成功条件, 制約/対象範囲, 成果物の形式, 必須項目(結論・根拠・未解決), 不足時の扱い\n\n"
    "## 回答のガイドライン\n"
    "- 複数のツールから得られた断片的な情報を整理し、一貫性のある回答にまとめてください。\n"
    "- 根拠の提示: ツールで得られた具体的な事実（数値、日付、名称など）を引用してください。\n"
    "- 出力はユーザーへの返答テキストのみとすること。JSONや内部状態の列挙は禁止。\n"
)


def _build_deepagent_worker_graph(
    model_name: str,
    workspace_dir: Path,
) -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
    from langchain.tools import ToolRuntime
    from langgraph.store.memory import InMemoryStore

    memory_store = InMemoryStore()
    artifact_dir = workspace_dir / "artifacts"
    artifacts_backend = FilesystemBackend(root_dir=str(artifact_dir), virtual_mode=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def make_backend(runtime: ToolRuntime) -> CompositeBackend:
        return CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/artifacts/": artifacts_backend,
            },
        )

    research_tools = build_research_tools(artifact_dir=artifact_dir)

    model = get_ollama_client(model_name=model_name)
    agent = create_deep_agent(
        model=model,
        tools=research_tools,
        system_prompt=system_prompt,
        backend=make_backend,
        store=memory_store,
    )

    return agent  # pyright: ignore[reportReturnType]
