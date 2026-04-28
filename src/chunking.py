"""Chunking sobre Documents de LangChain.

Estrategia: splitter recursivo con separadores que respetan la jerarquía markdown
(`## ` > `### ` > párrafo > frase). Los metadatos del Document fuente se propagan
a cada chunk y se añade `chunk_index`.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )


def split_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Divide cada Document en chunks anotados con `chunk_index`."""
    # TODO Fase 1: usar splitter.split_documents(docs) y enumerar por source_path
    raise NotImplementedError
