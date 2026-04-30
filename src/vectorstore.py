"""Wrapper sobre la integración LangChain ↔ ChromaDB persistente."""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import COLLECTION_NAME


class VectorStore:
    def __init__(self, persist_dir: Path, embeddings: Embeddings) -> None:
        self._persist_dir = persist_dir
        self._embeddings = embeddings
        self._client: Chroma | None = None

    def _get_client(self) -> Chroma:
        if self._client is None:
            self._client = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self._embeddings,
                persist_directory=str(self._persist_dir),
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._client

    def reset(self) -> None:
        client = self._get_client()
        client.delete_collection()
        self._client = None

    def add_documents(self, chunks: list[Document]) -> None:
        ids = [
            f"{c.metadata['source_path']}#{c.metadata.get('chunk_index', i)}"
            for i, c in enumerate(chunks)
        ]
        self._get_client().add_documents(documents=chunks, ids=ids)

    def search(self, query: str, k: int = 5, filter_dict: dict | None = None) -> list[Document]:
        return self._get_client().similarity_search(query, k=k, filter=filter_dict)
