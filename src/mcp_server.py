"""Servidor MCP que expone el RAG sobre los apuntes de AAU2.

Cubre U6: tools, resources y prompts.
Cliente esperado: Claude Desktop.

Lanzamiento manual:
    python -m src.mcp_server
"""
from __future__ import annotations

import logging

from fastmcp import FastMCP

from .config import Settings
from .llm_client import LLMClient
from .rag import RagPipeline
from .vectorstore import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

mcp = FastMCP("apuntes-aau2")

# Inicialización perezosa para que importar el módulo no requiera API key
_settings: Settings | None = None
_pipeline: RagPipeline | None = None
_store: VectorStore | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        s = _get_settings()
        llm = LLMClient(s)
        _store = VectorStore(s.chroma_dir, llm.embeddings)
    return _store


def _get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        s = _get_settings()
        store = _get_store()
        llm = LLMClient(s)
        _pipeline = RagPipeline(s, store, llm.chat)
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
    k = max(1, min(k, 10))
    if unidad is not None and not 1 <= unidad <= 6:
        return [{"error": f"Unidad debe estar entre 1 y 6, recibido: {unidad}"}]
    try:
        pipeline = _get_pipeline()
        docs = pipeline.retrieve(query, k=k, unidad=unidad)
    except Exception as e:
        logger.exception("Error en buscar_apuntes")
        return [{"error": f"Error al buscar: {e}"}]
    if not docs:
        return [{"info": "No se encontraron fragmentos relevantes."}]
    return [
        {
            "contenido": doc.page_content,
            "unidad": doc.metadata.get("unidad"),
            "archivo": doc.metadata.get("source_path"),
            "tipo": doc.metadata.get("tipo"),
            "titulo": doc.metadata.get("titulo"),
        }
        for doc in docs
    ]


@mcp.tool()
def responder_pregunta(pregunta: str, unidad: int | None = None) -> dict:
    """Responde una pregunta sobre los apuntes citando las fuentes.

    Args:
        pregunta: pregunta en lenguaje natural.
        unidad: opcional, restringe la búsqueda a una unidad concreta.
    """
    if unidad is not None and not 1 <= unidad <= 6:
        return {"error": f"Unidad debe estar entre 1 y 6, recibido: {unidad}"}
    try:
        pipeline = _get_pipeline()
        result = pipeline.answer(pregunta, unidad=unidad)
    except Exception as e:
        logger.exception("Error en responder_pregunta")
        return {"error": f"Error al responder: {e}"}
    return {
        "respuesta": result.answer,
        "fuentes": [
            {
                "archivo": doc.metadata.get("source_path"),
                "unidad": doc.metadata.get("unidad"),
            }
            for doc in result.sources
        ],
    }


@mcp.tool()
def listar_unidades() -> list[dict]:
    """Devuelve la estructura del temario (unidades y archivos disponibles) desde el índice ChromaDB."""
    try:
        store = _get_store()
        sources = store.list_sources()
    except Exception as e:
        logger.exception("Error en listar_unidades")
        return [{"error": f"Error al listar unidades: {e}"}]

    unidades: dict[int, list[str]] = {}
    for src in sources:
        u = src["unidad"]
        if u not in unidades:
            unidades[u] = []
        unidades[u].append(src["source_path"])

    return [
        {"unidad": u, "archivos": sorted(archivos)}
        for u, archivos in sorted(unidades.items())
    ]


@mcp.tool()
def obtener_documento(ruta_relativa: str) -> str:
    """Devuelve el contenido de un documento indexado a partir de su ruta relativa.

    Args:
        ruta_relativa: ruta del archivo, ej. "unidad5_sesion1/sesion1_fundamentos_rag_embeddings_vectores.md"
    """
    try:
        store = _get_store()
        chunks = store.get_by_source(ruta_relativa)
    except Exception as e:
        logger.exception("Error en obtener_documento")
        return f"Error al obtener documento: {e}"

    if not chunks:
        return f"Documento no encontrado en el índice: {ruta_relativa}"

    return "\n\n".join(c.page_content for c in chunks)


# ---------- RESOURCES ----------

@mcp.resource("apuntes://temario")
def temario() -> str:
    """Outline general de las unidades del curso AAU2."""
    unidades = listar_unidades()
    if unidades and "error" in unidades[0]:
        return unidades[0]["error"]
    lines = ["# Temario AAU2\n"]
    for item in unidades:
        u = item["unidad"]
        archivos = item["archivos"]
        if u == -1:
            lines.append("\n## Otros archivos\n")
        else:
            lines.append(f"\n## Unidad {u}\n")
        for a in archivos:
            lines.append(f"- {a}")
    return "\n".join(lines)


@mcp.resource("apuntes://unidad/{numero}")
def unidad_resource(numero: str) -> str:
    """Listado de archivos disponibles para la unidad indicada."""
    try:
        num = int(numero)
    except ValueError:
        return f"Número de unidad inválido: {numero}"

    unidades = listar_unidades()
    if unidades and "error" in unidades[0]:
        return unidades[0]["error"]
    for item in unidades:
        if item["unidad"] == num:
            lines = [f"# Unidad {num}\n"]
            for a in item["archivos"]:
                lines.append(f"- {a}")
            return "\n".join(lines)

    return f"No se encontró la unidad {num}"


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
