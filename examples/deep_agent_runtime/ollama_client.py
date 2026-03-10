import os
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama


def get_ollama_client(model_name="gpt-oss:20b") -> BaseChatModel:
    headers = {"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")}
    base_url = os.getenv("OLLAMA_BASE_URL", "")
    client = ChatOllama(
        model=model_name,
        base_url=base_url,
        client_kwargs={"headers": headers},
    )
    return client
