"""Pipeline de ingesta: ml2_clases/ → ChromaDB.

Uso:
    python -m src.ingest          # incremental
    python -m src.ingest --reset  # limpia y reconstruye
"""
from __future__ import annotations

import sys

import typer
from rich.console import Console

from .chunking import split_documents
from .config import Settings
from .llm_client import LLMClient
from .loaders import iter_documents
from .vectorstore import VectorStore, VectorStoreError

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(reset: bool = typer.Option(False, "--reset", help="Borra el índice antes de reconstruir")) -> None:
    try:
        settings = Settings.from_env()
    except RuntimeError as e:
        console.print(f"[red]Error de configuración:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"[bold]Apuntes:[/bold] {settings.apuntes_dir}")
    console.print(f"[bold]Chroma :[/bold] {settings.chroma_dir}")

    if not settings.apuntes_dir.exists():
        console.print(f"[red]Directorio de apuntes no encontrado:[/red] {settings.apuntes_dir}")
        raise typer.Exit(1)

    embeddings = LLMClient(settings).embeddings
    store = VectorStore(settings.chroma_dir, embeddings)

    if reset:
        try:
            store.reset()
            console.print("[yellow]Índice borrado.[/yellow]")
        except VectorStoreError as e:
            console.print(f"[red]Error al borrar índice:[/red] {e}")
            raise typer.Exit(1)

    try:
        docs = list(iter_documents(settings.apuntes_dir))
    except Exception as e:
        console.print(f"[red]Error al cargar documentos:[/red] {e}")
        raise typer.Exit(1)

    if not docs:
        console.print("[yellow]No se encontraron documentos (.md/.pdf) en el directorio.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[bold]Documentos cargados:[/bold] {len(docs)}")

    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    console.print(f"[bold]Chunks creados:[/bold] {len(chunks)}")

    try:
        store.add_documents(chunks)
    except VectorStoreError as e:
        console.print(f"[red]Error al indexar:[/red] {e}")
        raise typer.Exit(1)

    console.print("[green]Índice actualizado.[/green]")


if __name__ == "__main__":
    app()
