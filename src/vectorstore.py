"""Wrapper sobre la integración LangChain ↔ ChromaDB persistente."""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import COLLECTION_NAME

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Error en operaciones de ChromaDB."""


class VectorStore:
    def __init__(self, persist_dir: Path, embeddings: Embeddings) -> None:
        self._persist_dir = persist_dir
        self._embeddings = embeddings
        self._client: Chroma | None = None

    def _get_client(self) -> Chroma:
        if self._client is None:
            try:
                self._client = Chroma(
                    collection_name=COLLECTION_NAME,
                    embedding_function=self._embeddings,
                    persist_directory=str(self._persist_dir),
                    collection_metadata={"hnsw:space": "cosine"},
                )
            except Exception as e:
                raise VectorStoreError(f"No se pudo conectar a ChromaDB en {self._persist_dir}: {e}") from e
        return self._client

    def reset(self) -> None:
        try:
            client = self._get_client()
            client.delete_collection()
            self._client = None
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Error al borrar la colección: {e}") from e

    def add_documents(self, chunks: list[Document]) -> None:
        if not chunks:
            logger.warning("add_documents llamado con lista vacía, nada que indexar.")
            return
        try:
            ids = [
                f"{c.metadata['source_path']}#{c.metadata.get('chunk_index', i)}"
                for i, c in enumerate(chunks)
            ]
            self._get_client().add_documents(documents=chunks, ids=ids)
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Error al indexar {len(chunks)} chunks: {e}") from e

    def search(self, query: str, k: int = 5, filter_dict: dict | None = None) -> list[Document]:
        try:
            return self._get_client().similarity_search(query, k=k, filter=filter_dict)
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Error en búsqueda semántica: {e}") from e
