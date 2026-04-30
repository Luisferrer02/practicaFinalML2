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

_UNIDAD_RE = re.compile(r"unidad(\d+)_")


def _infer_metadata(rel_path: Path) -> tuple[int, str]:
    parts = rel_path.parts
    unidad = -1
    tipo = "otro"
    for p in parts:
        m = _UNIDAD_RE.match(p)
        if m:
            unidad = int(m.group(1))
            if "practica" in p:
                tipo = "practica"
            elif "sesion" in p:
                tipo = "sesion"
            break
    name = rel_path.name.lower()
    if "ejercicios" in name:
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


def load_markdown(path: Path, base: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return _build_doc(text, path, base)


def load_pdf(path: Path, base: Path) -> Document:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return _build_doc(text, path, base)


def iter_documents(apuntes_dir: Path) -> Iterable[Document]:
    for path in sorted(apuntes_dir.rglob("*.md")):
        yield load_markdown(path, apuntes_dir)
    for path in sorted(apuntes_dir.rglob("*.pdf")):
        yield load_pdf(path, apuntes_dir)
