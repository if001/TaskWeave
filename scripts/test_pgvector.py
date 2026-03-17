from __future__ import annotations

import os
import uuid
from dotenv import load_dotenv


from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.vectorstores import PGVectorStore


load_dotenv()


## はじめの１度のみ初期化を行う
def init():
    url = os.getenv("ARTIFACT_PG_DSN_ADMIN", "").strip()
    engine = PGEngine.from_connection_string(url=url)
    engine.init_vectorstore_table(
        table_name="documents",
        schema_name="app",
        vector_size=768,
    )


def main() -> None:
    # init()
    dsn = os.getenv("ARTIFACT_PG_DSN", "").strip()
    if not dsn:
        raise RuntimeError("ARTIFACT_PG_DSN is required")

    embed_model = os.getenv("ARTIFACT_OLLAMA_EMBED_MODEL", "nomic-embed-text")
    base_url = os.getenv("OLLAMA_BASE_URL_LOCAL", "")

    embeddings = OllamaEmbeddings(model=embed_model, base_url=base_url)
    pg_engine = PGEngine.from_connection_string(url=dsn)
    store = PGVectorStore.create_sync(
        engine=pg_engine,
        table_name="documents",
        schema_name="app",
        embedding_service=embeddings,
    )

    doc = Document(
        page_content="pgvector connectivity check", metadata={"id": uuid.uuid4().hex}
    )
    store.add_documents([doc], ids=[uuid.uuid4().hex])
    results = store.similarity_search_with_score("connectivity check", k=1)
    if not results:
        raise RuntimeError("No results returned from PGVector")
    top_doc, score = results[0]
    print("pgvector ok:", top_doc.metadata.get("id", ""), score)


if __name__ == "__main__":
    main()
