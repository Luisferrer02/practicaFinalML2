"""Pipeline RAG construido con LCEL (LangChain Expression Language).

Cubre U5: retrieval semántico con filtros por unidad y síntesis con citas.
Cubre U2: system prompt estructurado (rol, reglas, formato).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .config import Settings
from .vectorstore import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class RagError(Exception):
    """Error en el pipeline RAG."""


SYSTEM_PROMPT = """Eres un asistente de estudio para el curso de Aprendizaje Automático 2 (AAU2).
Tu tarea es explicar conceptos de los apuntes del curso de forma clara y completa.

REGLAS:
- Explica el concepto en profundidad, desarrollando la respuesta de forma natural.
- Usa la información de los fragmentos proporcionados para formular tu respuesta.
- Si un fragmento es relevante, inclúyelo en tu explicación.
- Las fuentes se citan al final entre corchetes, ej: [archivo.md].
- No necesitas ser extremadamente breve; proporciona una explicación completa."""

USER_PROMPT_TEMPLATE = """Fragmentos de los apuntes:
{context}

Pregunta: {question}

Responde de forma concisa usando solo la información de los fragmentos."""


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE),
    ])


def _format_context(docs: list[Document]) -> str:
    seen = set()
    blocks = []
    for d in docs:
        source = d.metadata.get("source_path", "unknown")
        if source in seen:
            continue
        seen.add(source)
        blocks.append(f"--- {source} ---\n{d.page_content}")
    return "\n\n".join(blocks)


@dataclass
class RagAnswer:
    answer: str
    sources: list[Document]


class RagPipeline:
    def __init__(self, settings: Settings, store: VectorStore, chat: BaseChatModel) -> None:
        self._settings = settings
        self._store = store
        self._chat = chat
        self._prompt = _build_prompt()

    def retrieve(self, query: str, k: int | None = None, unidad: int | None = None) -> list[Document]:
        k = k or self._settings.top_k
        filter_dict = {"unidad": unidad} if unidad is not None else None
        try:
            return self._store.search(query, k=k, filter_dict=filter_dict)
        except VectorStoreError:
            raise
        except Exception as e:
            raise RagError(f"Error en la recuperación de documentos: {e}") from e

    def answer(self, question: str, unidad: int | None = None) -> RagAnswer:
        docs = self.retrieve(question, unidad=unidad)
        if not docs:
            return RagAnswer(
                answer="No se encontraron fragmentos relevantes en los apuntes.",
                sources=[],
            )
        context = _format_context(docs)
        try:
            response = self._chat.invoke(
                self._prompt.invoke({"question": question, "context": context})
            )
        except Exception as e:
            raise RagError(f"Error al generar respuesta con el LLM: {e}") from e
        return RagAnswer(answer=response.content, sources=docs)
