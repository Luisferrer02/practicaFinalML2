# Arquitectura

## Visión general

El sistema sigue un patrón RAG (Retrieval-Augmented Generation) expuesto como servidor MCP. Los apuntes del curso se indexan en una base vectorial y se consultan desde Claude Desktop mediante herramientas MCP.

```
ml2_clases/          src/ingest.py          ChromaDB
  *.md  *.pdf  ──────►  loaders  ──────►  (data/chroma/)
                         chunking              │
                         embeddings            │ similarity_search
                                               ▼
Claude Desktop  ◄──stdio──  MCP server  ◄──  VectorStore
                               │
                               │ síntesis
                               ▼
                          NVIDIA NIM
                     (llama-3.1-nemotron)
```

## Componentes

### Capa de datos

| Módulo | Responsabilidad |
|--------|----------------|
| `src/loaders.py` | Lee archivos `.md` y `.pdf` del directorio de apuntes. Produce un `Document` por archivo con metadatos inferidos (unidad, tipo, título). |
| `src/chunking.py` | Divide cada documento en fragmentos usando `RecursiveCharacterTextSplitter` con separadores que respetan la jerarquía markdown (`##` > `###` > párrafo). |
| `src/vectorstore.py` | Wrapper sobre ChromaDB (vía `langchain-chroma`). Gestiona la colección persistente, indexación por IDs deterministas y búsqueda semántica con filtros. |

### Capa de inteligencia

| Módulo | Responsabilidad |
|--------|----------------|
| `src/llm_client.py` | Factoría de embeddings (`NVIDIAEmbeddings`) y chat (`ChatNVIDIA`) con reintentos exponenciales vía `.with_retry()`. |
| `src/rag.py` | Pipeline RAG: recupera fragmentos relevantes, formatea el contexto, invoca el LLM con un prompt estructurado y devuelve la respuesta con citas. |

### Capa de exposición

| Módulo | Responsabilidad |
|--------|----------------|
| `src/mcp_server.py` | Servidor FastMCP con 4 tools, 2 resources y 2 prompts. Inicialización perezosa del pipeline. |
| `src/cli.py` | CLI con Typer para probar el RAG sin Claude Desktop (`ask`, `search`). |
| `src/ingest.py` | CLI de ingesta: carga documentos, genera chunks, indexa en ChromaDB. |

### Configuración

| Módulo | Responsabilidad |
|--------|----------------|
| `src/config.py` | Lee variables de entorno (`.env`) y las expone como `Settings` (dataclass inmutable). Valida que las claves obligatorias estén presentes. |

## Flujo de datos

### Ingesta (`python -m src.ingest --reset`)

1. `loaders.iter_documents()` recorre `APUNTES_DIR` y produce un `Document` por archivo.
2. `chunking.split_documents()` divide cada documento en chunks de ~800 caracteres con 120 de overlap.
3. `LLMClient.embeddings` genera vectores de cada chunk vía NVIDIA NIM.
4. `VectorStore.add_documents()` persiste chunks + embeddings en ChromaDB con IDs deterministas (`source_path#chunk_index`).

### Consulta (MCP tool `responder_pregunta`)

1. Claude Desktop envía la pregunta al servidor MCP vía stdio.
2. `RagPipeline.retrieve()` busca los top-k fragmentos más similares en ChromaDB (opcionalmente filtrados por unidad).
3. `RagPipeline.answer()` formatea los fragmentos como contexto y los envía al LLM con un system prompt que define el rol de asistente de estudio.
4. El LLM genera una respuesta sintetizada con citas a las fuentes.
5. El servidor devuelve la respuesta + lista de fuentes a Claude Desktop.

## Decisiones de diseño

- **ChromaDB local** en lugar de Pinecone/Weaviate: no requiere cuenta cloud, simplifica el setup para un proyecto académico.
- **Modelo chat `nvidia/llama-3.1-nemotron-nano-8b-v1`**: buen equilibrio coste/calidad para síntesis en español. Gratuito en NVIDIA NIM.
- **Modelo embeddings `nvidia/llama-3.2-nv-embedqa-1b-v2`**: optimizado para retrieval Q&A, buen rendimiento en español.
- **Splitter recursivo por encabezados**: los apuntes están bien estructurados en markdown; respetar `##`/`###` como separadores mejora la coherencia semántica de los chunks.
- **Filtro por unidad en metadatos**: permite consultas dirigidas (`unidad=5`) sin necesidad de reentrenar ni crear colecciones separadas.
- **IDs deterministas en ChromaDB**: `source_path#chunk_index` permite re-ingestas idempotentes sin duplicar chunks.
- **Inicialización perezosa del pipeline MCP**: importar el módulo no requiere API key; el pipeline se crea en la primera llamada a una tool.
