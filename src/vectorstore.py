"""Wrapper sobre la integración LangChain ↔ ChromaDB persistente."""
from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import COLLECTION_NAME


def open_store(persist_dir: Path, embeddings: Embeddings) -> Chroma:
    """Abre (o crea) la colección persistente."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def reset_store(persist_dir: Path, embeddings: Embeddings) -> Chroma:
    """Borra el contenido de la colección y devuelve el store recreado."""
    # TODO Fase 1: open_store + delete_collection + open_store
    raise NotImplementedError


def add_documents(store: Chroma, chunks: list[Document]) -> None:
    """Indexa chunks usando ids derivados de source_path#chunk_index (idempotente)."""
    # TODO Fase 1: ids = [f"{c.metadata['source_path']}#{c.metadata['chunk_index']}" ...]
    raise NotImplementedError
