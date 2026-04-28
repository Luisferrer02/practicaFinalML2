"""Servidor MCP que expone el RAG sobre los apuntes de AAU2.

Cubre U6: tools, resources y prompts.
Cliente esperado: Claude Desktop.

Lanzamiento manual:
    python -m src.mcp_server
"""
from __future__ import annotations

from fastmcp import FastMCP

from .config import Settings
from .llm_client import LLMClient
from .rag import RagPipeline
from .vectorstore import VectorStore


mcp = FastMCP("apuntes-aau2")

# Inicialización perezosa para que importar el módulo no requiera API key
_pipeline: RagPipeline | None = None


def _get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        s = Settings.from_env()
        _pipeline = RagPipeline(s, VectorStore(s.chroma_dir), LLMClient(s))
    return _pipeline


# ---------- TOOLS ----------

@mcp.tool()
def buscar_apuntes(query: str, k: int = 5, unidad: int | None = None) -> list[dict]:
    """Busca fragmentos relevantes en los apuntes de AAU2.

    Args:
        query: pregunta o términos a buscar.
        k: número de fragmentos a devolver (1-10).
        unidad: opcional, filtra por unidad (1-6).
    """
    # TODO Fase 2: pipeline.retrieve → serializar a dicts
    raise NotImplementedError


@mcp.tool()
def responder_pregunta(pregunta: str, unidad: int | None = None) -> dict:
    """Responde una pregunta sobre los apuntes citando las fuentes.

    Args:
        pregunta: pregunta en lenguaje natural.
        unidad: opcional, restringe la búsqueda a una unidad concreta.
    """
    # TODO Fase 2: pipeline.answer → {"respuesta": ..., "fuentes": [...]}
    raise NotImplementedError


@mcp.tool()
def listar_unidades() -> list[dict]:
    """Devuelve la estructura del temario (unidades y archivos disponibles)."""
    # TODO Fase 2: recorrer apuntes_dir y agrupar por unidad
    raise NotImplementedError


@mcp.tool()
def obtener_documento(ruta_relativa: str) -> str:
    """Devuelve el contenido íntegro de un archivo de apuntes.

    Args:
        ruta_relativa: ruta dentro de ml2_clases/, ej. "unidad5_sesion1/sesion1_fundamentos_rag_embeddings_vectores.md"
    """
    # TODO Fase 2: leer archivo de apuntes_dir, validar que está dentro de la carpeta
    raise NotImplementedError


# ---------- RESOURCES ----------

@mcp.resource("apuntes://temario")
def temario() -> str:
    """Outline general de las 6 unidades del curso AAU2."""
    # TODO Fase 2: generar markdown con el árbol de archivos
    raise NotImplementedError


@mcp.resource("apuntes://unidad/{numero}")
def unidad_resource(numero: str) -> str:
    """Listado de archivos disponibles para la unidad indicada."""
    # TODO Fase 2: filtrar archivos por unidad y devolver markdown
    raise NotImplementedError


# ---------- PROMPTS ----------

@mcp.prompt()
def estudio_unidad(numero: int) -> str:
    """Plantilla para una sesión de estudio guiada de una unidad."""
    return (
        f"Quiero estudiar la unidad {numero} de AAU2. "
        f"Usa la tool `responder_pregunta` con unidad={numero} para construir un resumen "
        f"de los conceptos clave, después proponme 3 preguntas para autoevaluarme."
    )


@mcp.prompt()
def generar_quiz(tema: str, n_preguntas: int = 5) -> str:
    """Plantilla para generar un test sobre un tema concreto."""
    return (
        f"Genera {n_preguntas} preguntas tipo test (4 opciones, una correcta) sobre '{tema}'. "
        f"Antes de generarlas, usa `buscar_apuntes` para obtener fragmentos relevantes y "
        f"basa cada pregunta en uno de esos fragmentos, citando la fuente."
    )


if __name__ == "__main__":
    mcp.run()
