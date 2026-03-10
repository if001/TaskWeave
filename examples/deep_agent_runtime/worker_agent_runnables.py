from __future__ import annotations

import os
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

from runtime_core.agent_types import WorkerAgentInput, WorkerAgentOutput
from runtime_langchain.runnable_handler import CompiledStateGraphLike


def build_worker_agent_graph(
    use_real_agent: bool, backend: str, model_name: str, artifact_dir: Path
) -> CompiledStateGraphLike:
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


def _build_echo_worker_graph() -> CompiledStateGraphLike:
    class _EchoWorkerGraph:
        async def ainvoke(
            self, input: WorkerAgentInput, config: object | None = None
        ) -> WorkerAgentOutput:
            _ = config
            query = str(input.get("query", "")).strip()
            return WorkerAgentOutput(
                final_output=f"[mock worker-agent] deep researched: {query}"
            )

    return _EchoWorkerGraph()


def _build_langchain_worker_graph(
    model_name: str,
) -> CompiledStateGraphLike:
    from langchain.agents import create_agent

    model = get_ollama_client(model_name=model_name)
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="You are a concise worker agent specialized in heavy deep research.",
    )
    return agent


def _build_deepagent_worker_graph(
    model_name: str, artifact_dir: Path
) -> CompiledStateGraphLike:
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

    def write_artifact_via_backend(payload_text: str) -> str:
        filename = f"artifact_{int.from_bytes(os.urandom(6), 'big')}.json"
        write_result = artifacts_backend.write(f"/{filename}", payload_text)
        if write_result.error:
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
            return missing_search_service_result()
        return web_page_and_store_artifact(
            url=url,
            base_url=base_url,
            artifact_writer=write_artifact_via_backend,
        )

    model = get_ollama_client(model_name=model_name)
    agent = create_deep_agent(
        model=model,
        tools=[web_list, web_page],
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

    return agent
