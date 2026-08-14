# Changelog

## 0.2.0b0 — public beta

### Added

- Soft execution sandbox (`loomable.sandbox`), `ShellTools`, deep `code_exec`/`shell` wiring
- Bundled `browser` skill (Lightpanda MCP)
- `CodeIndex` + `CodeTools` + bundled `coding` skill + `create_deep_agent(profile="code")` (Alibaba zvec / pluggable store)
- `loomable.retrieval`: chunk strategies, multi-doc ingest, `build_retriever` (vector/lexical/hybrid; Alibaba zvec / Postgres / custom)
- Real **Alibaba zvec** file backend (`loomable[zvec]`), `open_vector_store()`, `InMemoryVectorBackend` for tests; `PgVectorBackend` remains the Postgres option
- **FAISS** vector backend (`loomable[faiss]` / `faiss-gpu`): `FaissVectorBackend` + `open_vector_store(engine="faiss", device="cpu"|"gpu"|"auto")`
- Agent L3 memory **defaults to Alibaba zvec** at `.loomable/memory_zvec` (`LongTermStore()` / `open_vector_store()`); use `engine="memory"`|`faiss`|`postgres` to opt out
- **Agentic retrieval**: `ingest` / `Corpus`, `AgenticRetriever`, `CompositeRetriever` with pluggable rewrite / mode router / rerank / compress / corpus router
- Embedders: **Gemini**, **Azure OpenAI**, **Hugging Face** (local MiniLM / Inference API) + batch `embed_many`; MMR reranker for diversity
- Ship-any-retriever: `Agent(retrievers=[...])` registers each as a `search_*` tool with query/k schema + system-prompt hint; `ensure_search_tool_name`
- Default ingest: popular docs/code/HTML/PDF/DOCX/PPTX/JSON/CSV + `http(s)` URLs; `json`/`csv` chunk strategies; complex multi-format RAG example/tests
- Stability policy ([docs/STABILITY.md](docs/STABILITY.md)), SECURITY.md, beta graduation plan
- `BuiltAgent.cancel()` / `Agent.cancel()` with active `RunContext` tracking
- SSE / stream client disconnect triggers cooperative cancel
- `mount_agent` / `mount_case` optional `api_key=` (Bearer or `X-API-Key`)
- `create_deep_agent(..., discovery_core="research-slim"|"research"|list)`
- Expanded CI: properties, toolkits, non-live integration; `feature/**` branches

### Changed

- Package status: **Beta** (`0.2.0b0`)
- `create_research_agent` emits `DeprecationWarning` (alias retained)

### Limits (documented)

- Local workspace FS only
- Cooperative cancel only
- Shared API-key serve auth (not full RBAC)

## 0.1.0 — alpha

- Agent · Team · Workflow · Case · AG-UI SSE
- Memory.compose, Postgres checkpointer
- Deep agent + progressive discovery (skills / tools / MCP)
