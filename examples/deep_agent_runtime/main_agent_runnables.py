from __future__ import annotations

from dotenv import load_dotenv

from langchain.agents.middleware import before_model, AgentState
from langgraph.runtime import Runtime

from runtime_langchain.research_handlers import build_mock_main_graph
from runtime_langchain.runnable_handler import CompiledStateGraphLike
from runtime_langchain.worker_tools import (
    WorkerLaunchRecorder,
    build_worker_request_tools,
)
from examples.deep_agent_runtime.ollama_client import get_ollama_client
from runtime_core.logging_utils import get_logger

load_dotenv()
logger = get_logger(__name__)


def build_main_agent_graph(
    use_real_agent: bool, model_name: str, recorder: WorkerLaunchRecorder
) -> CompiledStateGraphLike:
    if use_real_agent:
        from langchain.agents import create_agent

        @before_model(can_jump_to=["end"])
        def check_message(
            state: AgentState, runtime: Runtime
        ) -> dict[str, object] | None:
            _ = runtime
            m = state["messages"]
            logger.info(f"last message {m[-1]}\n\n")
            return None

        model = get_ollama_client(model_name)
        tools = build_worker_request_tools(recorder)
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=(
                "You are a main research agent. "
                "Use worker tools for heavy deep-research tasks: immediate, delayed one-time, or periodic."
            ),
            middleware=[check_message],
        )
        return agent
    return build_mock_main_graph(recorder)
