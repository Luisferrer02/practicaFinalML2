# Apuntes AAU2 — Servidor MCP con RAG

Servidor MCP que expone los apuntes del curso "Aprendizaje Automático 2" como una base de conocimiento consultable. Permite, desde Claude Desktop u otro cliente MCP, hacer preguntas sobre la teoría, prácticas y ejercicios del curso, obteniendo respuestas sintetizadas con citas a las fuentes.

## Unidades del curso aplicadas

- **Unidad 1 — IA Generativa y LLMs:** Selección y justificación del LLM en base a coste, latencia y calidad para el caso de uso Q&A en español.
- **Unidad 3 — Transformers y APIs:** Cliente OpenAI con manejo de errores y reintentos exponenciales para embeddings y chat completions.
- **Unidad 5 — RAG y Bases Vectoriales:** Pipeline RAG completo — ingesta de markdown y PDF, chunking recursivo, embeddings, indexación en ChromaDB persistente y recuperación semántica con filtros por metadatos.
- **Unidad 6 — Model Context Protocol:** Servidor MCP con FastMCP que expone **tools** (`buscar_apuntes`, `responder_pregunta`, `listar_unidades`, `obtener_documento`), **resources** (`apuntes://temario`, `apuntes://unidad/{n}`) y **prompts** (`estudio_unidad`, `generar_quiz`).

## Arquitectura

Ver [docs/arquitectura.md](docs/arquitectura.md) para el detalle. Resumen:

```
ml2_clases/  ──►  ingesta  ──►  ChromaDB
                                   ▲
Claude Desktop  ──stdio──►  MCP server  ──►  OpenAI (síntesis)
```

## Tecnologías utilizadas

| Capa | Librería | Versión |
|------|----------|---------|
| Servidor MCP | `fastmcp` | >=0.2 |
| Vector DB | `chromadb` | >=0.5 |
| LLM y embeddings | `openai` | >=1.40 |
| PDF | `pypdf` | >=4 |
| CLI dev | `typer` + `rich` | — |
| Reintentos | `tenacity` | >=8 |

## Instalación

```bash
git clone <este-repo>
cd practicaFinalML2

python -m venv .venv
source .venv/bin/activate          # Linux/Mac
.venv\Scripts\activate             # Windows

pip install -r requirements.txt

cp .env.example .env
# Editar .env con tu OPENAI_API_KEY
```

Apuntar `APUNTES_DIR` (en `.env`) a la ruta del repo `ml2_clases`.

## Uso

### 1. Indexar los apuntes

```bash
python -m src.ingest --reset
```

### 2. Probar el RAG por CLI (sin Claude Desktop)

```bash
python -m src.cli ask "¿Qué es chain-of-thought?"
python -m src.cli ask "Explica el pipeline RAG" --unidad 5
python -m src.cli search "embeddings" --k 3
```

### 3. Conectar con Claude Desktop

Añadir en `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "apuntes-aau2": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/ruta/absoluta/a/practicaFinalML2"
    }
  }
}
```

Reiniciar Claude Desktop y comprobar que aparecen las tools `buscar_apuntes`, `responder_pregunta`, etc.

## Decisiones técnicas

Ver [docs/arquitectura.md](docs/arquitectura.md). Principales:

- **ChromaDB local** en lugar de Pinecone: simplifica setup, no requiere cuenta cloud.
- **Modelo**: Por decidir
- **Modelo para retrieval en español** Por decidir
- **Splitter recursivo por encabezados** (`##`, `###`): los apuntes están bien estructurados, respetar esa jerarquía mejora la coherencia de los chunks.
- **Filtro por unidad en metadata**: permite consultas dirigidas sin reentrenar.

## Plan por fases

- [x] **Fase 0 — Esqueleto**: estructura, configs, stubs.
- [ ] **Fase 1 — RAG funcional**: loaders, chunking, ingesta, retrieval, CLI.
- [ ] **Fase 2 — Servidor MCP**: implementar tools/resources/prompts y conectar a Claude Desktop.
- [ ] **Fase 3 — Pulido**: errores, capturas, README final.

## Posibles adiciones

- Reranker (cross-encoder) sobre los top-k para mejorar precisión.
- Cache de embeddings por hash de contenido para re-ingestas incrementales.
- Soporte para Q&A multi-turno con historial conversacional.
- Despliegue del servidor MCP como remote MCP server (HTTP) para usarlo desde clientes web.

## Autores

- Luis Ferrer
- Victor Riddlestone
