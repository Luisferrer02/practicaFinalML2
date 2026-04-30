"""Pipeline RAG construido con LCEL (LangChain Expression Language).

Cubre U5: retrieval semántico con filtros por unidad y síntesis con citas.
Cubre U2: system prompt estructurado (rol, reglas, formato).
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from .config import Settings
from .vectorstore import VectorStore


SYSTEM_PROMPT = """Eres un asistente de estudio para "Aprendizaje Automático 2" (AAU2).
Respondes en español, basándote ESTRICTAMENTE en los fragmentos de apuntes que se te proporcionan.

Reglas:
- Si la respuesta no está en los fragmentos, responde: "No encuentro esto en los apuntes."
- Cita siempre las fuentes al final como [unidad N - source_path].
- Sé conciso. Si el usuario pide ejemplos, prioriza los que aparecen en los apuntes.
"""

USER_PROMPT_TEMPLATE = """Pregunta: {question}

Fragmentos recuperados:
{context}

Responde a la pregunta usando solo los fragmentos anteriores."""


def _build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE),
    ])


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        header = f"[{i}] unidad={m.get('unidad')} src={m.get('source_path')}"
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


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
        return self._store.search(query, k=k, filter_dict=filter_dict)

    def answer(self, question: str, unidad: int | None = None) -> RagAnswer:
        docs = self.retrieve(question, unidad=unidad)
        context = _format_context(docs)
        response = self._chat.invoke(self._prompt.invoke({"question": question, "context": context}))
        return RagAnswer(answer=response.content, sources=docs)
