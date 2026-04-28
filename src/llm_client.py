"""Factorías de los componentes LLM/embeddings de LangChain.

Cubre U3: el manejo de errores y reintentos lo proporciona LangChain (`with_retry`)
sobre las clases `ChatOpenAI` y `OpenAIEmbeddings`.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import Settings


def build_embeddings(settings: Settings) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embed_model,
        api_key=settings.openai_api_key,
    )


def build_chat(settings: Settings, temperature: float = 0.2) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=temperature,
        api_key=settings.openai_api_key,
        timeout=30,
    ).with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )
