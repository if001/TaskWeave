import os
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama


def get_ollama_client(
    model_name: str = "gpt-oss:20b", base_url: str = ""
) -> BaseChatModel:
    if base_url == "":
        base_url = os.getenv("OLLAMA_BASE_URL", base_url)
    # headers = {"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")}
    client = ChatOllama(
        model=model_name,
        base_url=base_url,
        verbose=True,
        reasoning=True,
        num_ctx=8192,
        # client_kwargs={"headers": headers},
    )
    return client


if __name__ == "__main__":
    client = ChatOllama(
        model="qwen3.5:4b",
        base_url="http://172.22.1.15:11434",
        verbose=True,
        # client_kwargs={"headers": headers},
    )
    r = client.invoke([{"role": "user", "content": "hello"}])
    print(r)
