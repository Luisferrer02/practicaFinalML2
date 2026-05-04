"""Lectores de archivos: producen `langchain_core.documents.Document` con metadatos.

Recorremos `apuntes_dir` y devolvemos un Document por archivo con:
- page_content: texto íntegro
- metadata: source_path (relativo), unidad (1..6 o -1), tipo, titulo
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from pypdf import PdfReader

_UNIDAD_RE = re.compile(r"unidad(\d+)_")


def infer_unidad(path_str: str) -> int:
    """Extrae el número de unidad de una ruta (ej. 'unidad3_sesion1/...' → 3). Devuelve -1 si no se encuentra."""
    for part in Path(path_str).parts:
        m = _UNIDAD_RE.match(part)
        if m:
            return int(m.group(1))
    return -1


def _infer_metadata(rel_path: Path) -> tuple[int, str]:
    unidad = infer_unidad(str(rel_path))
    tipo = "otro"
    for p in rel_path.parts:
        if _UNIDAD_RE.match(p):
            if "practica" in p:
                tipo = "practica"
            elif "sesion" in p:
                tipo = "sesion"
            break
    if "ejercicios" in rel_path.name.lower():
        tipo = "ejercicios"
    return unidad, tipo


def _first_h1(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def _build_doc(text: str, path: Path, base: Path) -> Document:
    rel = path.relative_to(base)
    unidad, tipo = _infer_metadata(rel)
    return Document(
        page_content=text,
        metadata={
            "source_path": str(rel).replace("\\", "/"),
            "unidad": unidad,
            "tipo": tipo,
            "titulo": _first_h1(text, rel.stem),
        },
    )


def read_file(path: Path) -> str:
    """Lee el contenido de un archivo .md o .pdf como texto."""
    if path.suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def load_file(path: Path, base: Path) -> Document:
    """Carga un archivo .md o .pdf y devuelve un Document con metadatos."""
    text = read_file(path)
    return _build_doc(text, path, base)


def iter_documents(apuntes_dir: Path) -> Iterable[Document]:
    for path in sorted(apuntes_dir.rglob("*.md")):
        yield load_file(path, apuntes_dir)
    for path in sorted(apuntes_dir.rglob("*.pdf")):
        yield load_file(path, apuntes_dir)
