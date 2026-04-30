"""Factorías de los componentes LLM/embeddings de LangChain para NVIDIA NIM.

Cubre U3: reintentos exponenciales sobre las clases de LangChain.
"""
from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from .config import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def embeddings(self) -> Embeddings:
        return NVIDIAEmbeddings(
            model=self._settings.embed_model,
            nvidia_api_key=self._settings.nv_api_key,
        )

    @property
    def chat(self) -> BaseChatModel:
        return ChatNVIDIA(
            model=self._settings.chat_model,
            nvidia_api_key=self._settings.nv_api_key,
            temperature=0.2,
        ).with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True,
        )
