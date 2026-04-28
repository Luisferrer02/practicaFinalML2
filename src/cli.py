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
    s = Settings.from_env()
    return RagPipeline(s, VectorStore(s.chroma_dir), LLMClient(s))


@app.command()
def ask(question: str, unidad: int = typer.Option(None, "--unidad", "-u")) -> None:
    rag = _pipeline()
    result = rag.answer(question, unidad=unidad)
    console.print(Panel(result.answer, title="Respuesta", border_style="green"))
    for i, src in enumerate(result.sources, 1):
        console.print(f"[dim]{i}.[/dim] [cyan]{src.metadata.get('source_path')}[/cyan]")


@app.command()
def search(query: str, k: int = 5, unidad: int = typer.Option(None, "--unidad", "-u")) -> None:
    rag = _pipeline()
    chunks = rag.retrieve(query, k=k, unidad=unidad)
    for i, c in enumerate(chunks, 1):
        console.print(Panel(c.text, title=f"{i}. {c.metadata.get('source_path')}"))


if __name__ == "__main__":
    app()
