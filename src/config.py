"""Configuración centralizada del proyecto. Lee variables de entorno con defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    nv_api_key: str
    embed_model: str
    chat_model: str
    apuntes_dir: Path
    chroma_dir: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        nv_api_key = os.environ.get("NV_API_KEY")
        embed_model = os.environ.get("NV_EMBED_MODEL")
        chat_model = os.environ.get("NV_CHAT_MODEL")
        if not nv_api_key:
            raise RuntimeError(
                "Define NV_API_KEY en .env. "
                "Obtén tu clave en https://build.nvidia.com → tu perfil → API Keys."
            )
        if not embed_model:
            raise RuntimeError(
                "Define NV_EMBED_MODEL en .env. "
                "(ej: nvidia/llama-3.2-nv-embedqa-1b-v2)"
            )
        if not chat_model:
            raise RuntimeError(
                "Define NV_CHAT_MODEL en .env. "
                "(ej: nvidia/llama-3.1-nemotron-nano-8b-v1)"
            )

        apuntes = Path(os.getenv("APUNTES_DIR", "../../AAU2Apuntes/ml2_clases"))
        if not apuntes.is_absolute():
            apuntes = (PROJECT_ROOT / apuntes).resolve()

        chroma = Path(os.getenv("CHROMA_DIR", "./data/chroma"))
        if not chroma.is_absolute():
            chroma = (PROJECT_ROOT / chroma).resolve()

        return cls(
            nv_api_key=nv_api_key,
            embed_model=embed_model,
            chat_model=chat_model,
            apuntes_dir=apuntes,
            chroma_dir=chroma,
            chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
            top_k=int(os.getenv("TOP_K", "5")),
        )


COLLECTION_NAME = "apuntes_aau2"
