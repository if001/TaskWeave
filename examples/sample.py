from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage

from examples.deep_agent_runtime.bootstrap import (
    DEFAULT_MODEL_NAME,
    _DEEPAGENT_ARTIFACT_DIR_ENV,
    _MODEL_ENV,
)
from examples.deep_agent_runtime.main_agent_runnables import (
    build_main_deep_agent_graph,
)
from examples.deep_agent_runtime.worker_agent_runnables import (
    resolve_deepagent_artifact_dir,
)
from runtime_langchain.task_orchestrator import GraphInput


def _last_assistant_message(output: object) -> str:
    normalized = _coerce_graph_output(output)
    messages = normalized.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, dict):
            role = str(message.get("role", "")).lower()
            if role in {"assistant", "ai"}:
                return str(message.get("content", "")).strip()
            continue
        content = getattr(message, "content", "")
        if _is_ai_message(message):
            return str(content).strip()
    return ""


def _is_ai_message(message: AnyMessage) -> bool:
    try:
        from langchain_core.messages import AIMessage
    except Exception:
        return False
    return isinstance(message, AIMessage)


def _coerce_graph_output(raw: object) -> GraphInput:
    if isinstance(raw, dict):
        messages = raw.get("messages")
        if isinstance(messages, list):
            return GraphInput(messages=messages)
    return GraphInput(messages=[{"role": "assistant", "content": str(raw).strip()}])


async def _run() -> None:
    load_dotenv()
    model_name = os.getenv(_MODEL_ENV, DEFAULT_MODEL_NAME)
    artifact_dir = resolve_deepagent_artifact_dir(_DEEPAGENT_ARTIFACT_DIR_ENV)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    async with build_main_deep_agent_graph(
        model_name=model_name,
        tools=[],
        workspace_dir=artifact_dir,
    ) as agent:
        print("agent ready. type 'exit' to quit.")
        while True:
            user_input = input("you> ").strip()
            if not user_input or user_input.lower() in {"exit", "quit"}:
                break
            output = await agent.ainvoke(
                GraphInput(messages=[{"role": "user", "content": user_input}]),
                config={"configurable": {"thread_id": "user-1"}},
            )
            response = _last_assistant_message(output)
            if response:
                print(f"agent> {response}")
            else:
                print("agent> (no response)")


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
