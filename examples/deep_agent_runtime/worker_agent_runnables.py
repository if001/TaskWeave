from __future__ import annotations

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownParameterType=false

import os
import json
from pathlib import Path
from tempfile import gettempdir

from examples.deep_agent_runtime.ollama_client import get_ollama_client
from examples.deep_agent_runtime.web_tools import (
    SearchToolResult,
    missing_search_service_result,
    resolve_simple_client_base_url,
    web_list_and_store_artifact,
    web_page_and_store_artifact,
)
from examples.deep_agent_runtime.artifact_tools import (
    artifact_save,
    artifact_search,
)

from runtime_core.types import JsonValue, Message, ensure_json_value
from langgraph.graph.state import CompiledStateGraph, StateGraph, END
from runtime_langchain.task_orchestrator import GraphInput
from langchain_core.messages import AnyMessage


def build_worker_agent_graph(
    use_real_agent: bool, backend: str, model_name: str, artifact_dir: Path
) -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
    if not use_real_agent:
        return _build_echo_worker_graph()
    if backend == "deepagent":
        return _build_deepagent_worker_graph(
            model_name=model_name, artifact_dir=artifact_dir
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


def _build_echo_worker_graph() -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
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


def _parse_raw_json(raw_json: str) -> JsonValue:
    try:
        parsed: object = json.loads(raw_json)
    except json.JSONDecodeError:
        parsed = None
    coerced = ensure_json_value(parsed) if parsed is not None else None
    if coerced is None:
        fallback: dict[str, JsonValue] = {"raw_text": raw_json}
        return fallback
    return coerced


def _build_deepagent_worker_graph(
    model_name: str, artifact_dir: Path
) -> CompiledStateGraph[GraphInput, None, GraphInput, GraphInput]:
    from deepagents import create_deep_agent
    from deepagents.backends import (
        CompositeBackend,
        FilesystemBackend,
        StateBackend,
        StoreBackend,
    )
    from langchain.tools import ToolRuntime, tool
    from langgraph.store.memory import InMemoryStore

    base_url = resolve_simple_client_base_url()
    memory_store = InMemoryStore()
    artifacts_backend = FilesystemBackend(root_dir=str(artifact_dir), virtual_mode=True)

    def make_backend(runtime: ToolRuntime) -> CompositeBackend:
        return CompositeBackend(
            default=StateBackend(runtime),
            routes={
                "/memories/": StoreBackend(runtime),
                "/artifacts/": artifacts_backend,
            },
        )

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
            return missing_search_service_result()
        return web_list_and_store_artifact(
            query=query,
            k=k,
            base_url=base_url,
            artifact_dir=artifact_dir,
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
            return missing_search_service_result()
        return web_page_and_store_artifact(
            url=url,
            base_url=base_url,
            artifact_dir=artifact_dir,
        )

    @tool("artifact_save")
    def artifact_save_tool(
        kind: str,
        title: str,
        summary: str,
        tags: str,
        raw_json: str,
    ) -> dict[str, str]:
        """Save a raw JSON payload plus metadata into /artifacts."""
        raw_payload = _parse_raw_json(raw_json)
        meta = artifact_save(
            kind=kind,
            raw=raw_payload,
            artifact_dir=artifact_dir,
            title=title,
            summary=summary,
            tags=tags,
        )
        return {
            "meta_path": str(Path(meta["raw_path"]).with_name("meta.json")),
            "raw_path": meta["raw_path"],
        }

    @tool("artifact_search")
    def artifact_search_tool(query: str, limit: int = 5) -> dict[str, list[dict[str, str]]]:
        """Search artifact metadata and return top matches."""
        matches = artifact_search(
            query=query,
            artifact_dir=artifact_dir,
            limit=limit,
        )
        rendered = [
            {
                "id": item["id"],
                "kind": item["kind"],
                "title": item["title"],
                "summary": item["summary"],
                "raw_path": item["raw_path"],
            }
            for item in matches
        ]
        return {"matches": rendered}

    model = get_ollama_client(model_name=model_name)
    agent = create_deep_agent(
        model=model,
        tools=[web_list, web_page, artifact_save_tool, artifact_search_tool],
        system_prompt=(
            "You are a focused deep-research worker agent. "
            "Use web_list to collect candidate sources and web_page to fetch details. "
            "Do not keep full search or page payloads in chat context. "
            "Persist them in artifact files and return concise summaries with artifact paths. "
            "Use /memories/ for long-term memory that should persist across threads."
        ),
        backend=make_backend,
        store=memory_store,
    )

    return agent  # pyright: ignore[reportReturnType]
