"""Servidor MCP que expone el RAG sobre los apuntes de AAU2.

Cubre U6: tools, resources y prompts.
Cliente esperado: Claude Desktop.

Lanzamiento manual:
    python -m src.mcp_server
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from fastmcp import FastMCP

from .config import Settings
from .llm_client import LLMClient
from .rag import RagPipeline
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)

mcp = FastMCP("apuntes-aau2")

# Inicialización perezosa para que importar el módulo no requiera API key
_pipeline: RagPipeline | None = None


def _get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        s = Settings.from_env()
        llm = LLMClient(s)
        _pipeline = RagPipeline(s, VectorStore(s.chroma_dir, llm.embeddings), llm.chat)
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
    """Devuelve la estructura del temario (unidades y archivos disponibles)."""
    try:
        settings = Settings.from_env()
    except RuntimeError as e:
        return [{"error": f"Error de configuración: {e}"}]

    base = settings.apuntes_dir
    if not base.exists():
        return [{"error": f"Directorio de apuntes no encontrado: {base}"}]

    unidades: dict[int, list[str]] = {}
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith((".md", ".pdf")):
                rel = Path(root).relative_to(base)
                parts = rel.parts
                unidad_num = -1
                for p in parts:
                    m = re.match(r"unidad(\d+)", p)
                    if m:
                        unidad_num = int(m.group(1))
                        break
                if unidad_num not in unidades:
                    unidades[unidad_num] = []
                unidades[unidad_num].append(str(rel / f))

    return [
        {"unidad": u, "archivos": sorted(files)}
        for u, files in sorted(unidades.items())
    ]


@mcp.tool()
def obtener_documento(ruta_relativa: str) -> str:
    """Devuelve el contenido íntegro de un archivo de apuntes.

    Args:
        ruta_relativa: ruta dentro de ml2_clases/, ej. "unidad5_sesion1/sesion1_fundamentos_rag_embeddings_vectores.md"
    """
    try:
        settings = Settings.from_env()
    except RuntimeError as e:
        return f"Error de configuración: {e}"

    base = settings.apuntes_dir
    safe_path = os.path.normpath(ruta_relativa)
    full_path = base / safe_path

    if not str(full_path.resolve()).startswith(str(base.resolve())):
        return "Error: ruta fuera del directorio de apuntes."

    if not full_path.exists():
        return f"Archivo no encontrado: {ruta_relativa}"

    try:
        if full_path.suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(full_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return full_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.exception("Error en obtener_documento")
        return f"Error al leer el archivo: {e}"


# ---------- RESOURCES ----------

@mcp.resource("apuntes://temario")
def temario() -> str:
    """Outline general de las 6 unidades del curso AAU2."""
    unidades = listar_unidades()
    if unidades and "error" in unidades[0]:
        return unidades[0]["error"]
    lines = ["# Temario AAU2\n"]
    for item in unidades:
        u = item["unidad"]
        archivos = item["archivos"]
        if u == -1:
            lines.append(f"\n## Otros archivos\n")
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
