"""CLI para probar el RAG sin Claude Desktop. Útil para demos y debugging.

Uso:
    python -m src.cli ask "¿Qué es chain-of-thought?"
    python -m src.cli ask "Explica RAG" --unidad 5
    python -m src.cli search "embeddings" --k 3
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

from .config import Settings
from .llm_client import LLMClient
from .rag import RagPipeline
from .vectorstore import VectorStore

app = typer.Typer(add_completion=False)
console = Console()


def _pipeline() -> RagPipeline:
    try:
        s = Settings.from_env()
    except RuntimeError as e:
        console.print(f"[red]Error de configuración:[/red] {e}")
        raise typer.Exit(1)
    llm = LLMClient(s)
    store = VectorStore(s.chroma_dir, llm.embeddings)
    return RagPipeline(s, store, llm.chat)


@app.command()
def ask(question: str, unidad: int = typer.Option(None, "--unidad", "-u")) -> None:
    rag = _pipeline()
    try:
        result = rag.answer(question, unidad=unidad)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    console.print(Panel(result.answer, title="Respuesta", border_style="green"))
    for i, src in enumerate(result.sources, 1):
        console.print(f"[dim]{i}.[/dim] [cyan]{src.metadata.get('source_path')}[/cyan]")


@app.command()
def search(query: str, k: int = 5, unidad: int = typer.Option(None, "--unidad", "-u")) -> None:
    rag = _pipeline()
    try:
        chunks = rag.retrieve(query, k=k, unidad=unidad)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    if not chunks:
        console.print("[yellow]No se encontraron resultados.[/yellow]")
        return
    for i, c in enumerate(chunks, 1):
        console.print(Panel(c.page_content, title=f"{i}. {c.metadata.get('source_path')}"))


if __name__ == "__main__":
    app()
