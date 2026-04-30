"""Pipeline de ingesta: ml2_clases/ → ChromaDB.

Uso:
    python -m src.ingest          # incremental
    python -m src.ingest --reset  # limpia y reconstruye
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.progress import track

from .chunking import split_documents
from .config import Settings
from .llm_client import LLMClient
from .loaders import iter_documents
from .vectorstore import VectorStore

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(reset: bool = typer.Option(False, "--reset", help="Borra el índice antes de reconstruir")) -> None:
    settings = Settings.from_env()
    console.print(f"[bold]Apuntes:[/bold] {settings.apuntes_dir}")
    console.print(f"[bold]Chroma :[/bold] {settings.chroma_dir}")

    embeddings = LLMClient(settings).embeddings
    store = VectorStore(settings.chroma_dir, embeddings)

    if reset:
        store.reset()
        console.print("[yellow]Índice borrado.[/yellow]")

    docs = list(iter_documents(settings.apuntes_dir))
    console.print(f"[bold]Documentos cargados:[/bold] {len(docs)}")

    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    console.print(f"[bold]Chunks creados:[/bold] {len(chunks)}")

    store.add_documents(chunks)
    console.print("[green]Índice actualizado.[/green]")


if __name__ == "__main__":
    app()
