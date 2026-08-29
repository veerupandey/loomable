# Changelog

## Unreleased

### Added

- **Graph engineering** primitives on Workflow / Step (see `examples/patterns/07_graph_engineering.py`):
  - `Step(on_failure=...)` / `Workflow.step(..., on_failure=)` — local failure policies: `raise`, `retry`, `skip`, `fallback`, `stop` (`StepFailed`)
  - `Step(reads=...)` / edge `payload_key` — edge data contracts so a step consumes a named SharedState key
  - `Workflow.verify(body, check=..., max_retries=)` — generate → verify → repair with a hard budget
  - `Step(complexity="low"|"high")` — cost hint for model-tier routing
  - Inspectable `route_decision` / SharedState `_route_decision` from `RouterNode` and `Workflow.branch`
  - Hard-path fixes: parallel `stop` commits successful siblings before escalating; `Workflow.state` preserved after `StepFailed`; `max_retries` honored for all policies; cancel interrupts retries; nested `Parallel_Group` inherits scoped checkpointer; `CancelledError` not swallowed as skip/fallback
- **LangGraph / Agno control-plane parity:**
  - `Workflow.route(chooser, **choices)` — N-way Router (Agno Router / LangGraph multi-edge)
  - `Command(goto=..., update=...)` — combine routing with state patches from steps/choosers
  - `Workflow.get_state()` / `update_state()` / `list_states()` — checkpoint inspection, patch, time-travel list
  - `Workflow(reducers={...})` — expose SharedState reducers for parallel joins
- Nested `Flow` / `Parallel_Group` reuses the parent `SharedState` (parent keys no longer wiped by fan-out)
- Final checkpoints record only nodes that actually ran (unselected `.route` / `.branch` arms are not marked completed)

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
- Vector engines: **zvec**, **FAISS**, **Chroma** (file/HTTP), **Milvus** (Lite `.db` / server), **Postgres/pgvector**; `uri=` shorthand + matrix integration tests
- PDF ingest: page extract + bounded page chunks (no 8k truncation / overlap-shard bug); large-PDF quality matrix across all engines
- Retrieval **metadata**: ingest-time fields on every chunk/hit (author, tags, page, filename, …) + `retrieve(..., filters=)`
- `Agent(knowledge_base=)` / `create_deep_agent(knowledge_base=)` — knowledge base **is** a vector store (optional ingest); `retrievers=` for extra search tools; Team / Case / Workflow / Flow inherit the same object. Search tools stay advertised under discovery.
- SECURITY.md
- `BuiltAgent.cancel()` / `Agent.cancel()` with active `RunContext` tracking
- SSE / stream client disconnect triggers cooperative cancel
- `mount_agent` / `mount_case` optional `api_key=` (Bearer or `X-API-Key`)
- `create_deep_agent(..., discovery_core="research-slim"|"research"|list)`
- Expanded CI: properties, toolkits, non-live integration; `feature/**` branches

### Changed

- Package status: **Beta** (`0.2.0b0`)
- CI installs `.[dev,toolkits]`; README badge pins `main`
- Default zvec L3 store reuses an in-process collection handle (no exclusive-lock deadlock across `Agent()` builds)
- Examples audited for 0.2.0 APIs: unique numbering, live `knowledge_base` / Team inherit demos, retrieval demos use live providers
- Pattern examples teach `Workflow` / `Team` (not `sequential`/`parallel`/`route` helpers)
- Docs Level 4–6 teach `Workflow` parallel/branch/HITL; Flow demoted to escape hatch
- Root README teaches `Memory.compose` + `create_deep_agent(profile=...)`
- Memory compose example uses `Memory.compose` (not flat `session_store=` / `note_store=`)
- Shared provider helper adds `make_embedder()` aligned with chat credentials
- Renamed `memory/03_workflow_shared_memory.py`, `advanced/02_workflow_branch.py`
- Fail loud on silent no-ops: `knowledge=` without `embedder=`, `memory_tool=`/`UserMemory(auto_extract=)` without a note store, `Team(hard=True)` on soft modes
- `mount_case` / case-mode Agent omit NDJSON `/run/stream`; `Agent(mode="case").astream` raises
- `knowledge_base=` on Agent / Team / Case / Workflow / `create_deep_agent`
- `BuiltAgent.astream` falls back to `arun` when tools / complexity router are present (no silent tool skip)
- Reject unknown `Agent(mode=...)`, invalid `dispatch=`, and case-only `checkpointer=`/`max_rounds=` without `mode="case"`
- Remove unused `ConversationMemory.scope`; wire `Agent(description=)` into the system prompt
- Docs honesty: complexity router is opt-in; Team has no `scopes=`
- Cooperative `cancel()` on Workflow / Case / Team / `Agent(mode="case")`; serve disconnect walks those targets
- `strict_require_tools=True` raises `RequireToolsError` (WR-021)
- `Workflow(require_tools=...)` / `.step(..., require_tools=)` inherit onto Agent steps (WR-022)
- Soft `Team(mode="coordinate")` auto-requires `delegate_to_*` and falls back to running skipped members (WR-020)
- Workflow branch join preserves `AgentOutput` (so `result.output.text()` is the branch text)
- `Case.from_agent` copies `require_tools` / `strict_require_tools` / `require_confirmation` / `approver`
- `Agent(approver=)` is applied on `build()`
- `confirm=True` without `checkpointer=` + `session_id=` raises; HITL is rejected inside `.parallel()` / `.branch()` / `.loop()`
- `Memory.to_agent_kwargs()` forwards `UserMemory(auto_extract=True)` as `memory_auto_extract`
- Custom Flow engines with a checkpointer fail loud instead of silently dropping HITL/checkpoint kwargs
- Soft `except: pass` on serve cancel / bind_session and discovery activation → logged
- Docs: NDJSON demoted; Flow Engine Workflow-first; KnowledgeMemory in compose example
- Docs: public surface is README + `docs/API.md`; removed internal planning notes

### Removed (greenfield clean — no compatibility shims)

- `Agent(multimodal=...)` — media is default; use `modalities=` / `text_only=`
- `Memory.compose(short=` / `long=`) — use `conversation=` / `user=`
- `Memory.with_user_id()` — use `with_scopes(user_id=...)`
- `ScopedNoteStore(user_id=)` — require `scope=MemoryScope.of(...)`
- Flat store kwargs overriding `memory=` — conflict raises `AgentConfigError`
- `WorkingMemory` inside `Agent(memory=...)` — raises; use `Workflow(memory=True)`
- `Loop(end_condition=)` — use `verifier=` (Workflow uses `until=`)
- `Workflow.branch(name=)` no-op kwarg
- Char-based offload `threshold=` — use `threshold_tokens=` / `offload_threshold_tokens=`
- Top-level `sequential` / `parallel` / `route` / `coordinate` exports — use `Workflow` / `Team` (helpers remain under `loomable.flow.helpers`)
- `Map` / `Router` aliases — use `MapNode` / `RouterNode`
- `rank_match` discovery helper — use `rank_bm25`
- Passing kernel `MemoryManager` as `Agent(memory=...)` — use `Memory.compose`
- `HITLPause` — never raised; Workflow HITL uses `FlowPaused`
- Internal planning docs (`docs/STABILITY.md`, `docs/BETA_PLAN.md`, `docs/COMPETITIVE.md`)
- `ExtensionRegistry` removed from `loomable.kernel.__all__` (import `loomable.kernel.registry`)

### Limits (documented)

- Local workspace FS only
- Cooperative cancel only
- Shared API-key serve auth (not full RBAC)

## 0.1.0 — alpha

- Agent · Team · Workflow · Case · AG-UI SSE
- Memory.compose, Postgres checkpointer
- Deep agent + progressive discovery (skills / tools / MCP)
