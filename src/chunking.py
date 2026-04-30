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
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(docs)
    result = []
    for doc in chunks:
        source = doc.metadata.get("source_path", "unknown")
        existing_idx = doc.metadata.get("chunk_index")
        if existing_idx is None:
            idx = len([c for c in result if c.metadata.get("source_path") == source])
            doc.metadata["chunk_index"] = idx
        result.append(doc)
    return result
