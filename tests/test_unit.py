"""Tests unitarios — no requieren API key ni ChromaDB."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


# ---------- loaders ----------

from src.loaders import _infer_metadata, _first_h1


@pytest.mark.parametrize("rel_path, expected_unidad, expected_tipo", [
    (Path("unidad1_sesion1/intro.md"), 1, "sesion"),
    (Path("unidad3_practica2/ejercicio.md"), 3, "practica"),
    (Path("unidad5_sesion1/ejercicios_rag.md"), 5, "ejercicios"),
    (Path("otros/readme.md"), -1, "otro"),
])
def test_infer_metadata(rel_path, expected_unidad, expected_tipo):
    unidad, tipo = _infer_metadata(rel_path)
    assert unidad == expected_unidad
    assert tipo == expected_tipo


def test_first_h1_found():
    assert _first_h1("# Mi Título\ncontenido", "fallback") == "Mi Título"


def test_first_h1_fallback():
    assert _first_h1("sin encabezado", "fallback") == "fallback"


# ---------- chunking ----------

from src.chunking import split_documents


def test_split_documents_propagates_metadata():
    doc = Document(page_content="A " * 500, metadata={"source_path": "test.md", "unidad": 1})
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert c.metadata["source_path"] == "test.md"
        assert c.metadata["unidad"] == 1
        assert "chunk_index" in c.metadata


def test_split_documents_empty():
    assert split_documents([], chunk_size=100, chunk_overlap=20) == []


# ---------- config ----------

from src.config import Settings


def test_settings_missing_api_key():
    with patch.dict("os.environ", {"NV_API_KEY": "", "NV_EMBED_MODEL": "x", "NV_CHAT_MODEL": "x"}, clear=False):
        with pytest.raises(RuntimeError, match="NV_API_KEY"):
            Settings.from_env()


def test_settings_missing_embed_model():
    with patch.dict("os.environ", {"NV_API_KEY": "key", "NV_EMBED_MODEL": "", "NV_CHAT_MODEL": "x"}, clear=False):
        with pytest.raises(RuntimeError, match="NV_EMBED_MODEL"):
            Settings.from_env()


# ---------- rag ----------

from src.rag import _format_context, RagPipeline, RagAnswer


def test_format_context_deduplicates():
    docs = [
        Document(page_content="contenido A", metadata={"source_path": "a.md"}),
        Document(page_content="contenido A bis", metadata={"source_path": "a.md"}),
        Document(page_content="contenido B", metadata={"source_path": "b.md"}),
    ]
    result = _format_context(docs)
    assert result.count("--- a.md ---") == 1
    assert "--- b.md ---" in result


def test_answer_no_docs_returns_message():
    store = MagicMock()
    store.search.return_value = []
    chat = MagicMock()
    settings = MagicMock(top_k=5)
    pipeline = RagPipeline(settings, store, chat)
    result = pipeline.answer("pregunta")
    assert "No se encontraron" in result.answer
    assert result.sources == []
    chat.invoke.assert_not_called()


# ---------- vectorstore ----------

from src.vectorstore import VectorStore, VectorStoreError


def test_add_documents_empty_is_noop():
    store = VectorStore(Path("/tmp/fake"), MagicMock())
    store.add_documents([])  # should not raise


# ---------- mcp input validation ----------

from src.mcp_server import buscar_apuntes, responder_pregunta


def test_buscar_apuntes_invalid_unidad():
    result = buscar_apuntes("test", k=5, unidad=99)
    assert result[0].get("error")
    assert "1 y 6" in result[0]["error"]


def test_responder_pregunta_invalid_unidad():
    result = responder_pregunta("test", unidad=0)
    assert result.get("error")
    assert "1 y 6" in result["error"]
