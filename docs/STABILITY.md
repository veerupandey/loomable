# Loomable API stability

Loomable **0.2.0b0** is a **public beta**. The surface below is feature-complete for
the declared scope. Breaking changes are rare and require a deprecation note in
`CHANGELOG.md` before removal.

## Stable (supported in beta)

| Surface | Notes |
|---------|--------|
| `Agent`, `BuiltAgent`, `Team` | Core runnable agents |
| `Workflow`, `Step`, `Loop`, `Condition`, `Parallel_Group` | Durable multi-step |
| `Case`, `Board`, `WorkItem` | Goal + WorkItems board |
| `Memory`, `MemoryScope`, `ConversationMemory`, `UserMemory`, `KnowledgeMemory`, `WorkingMemory`, `open_session_store` | Composable memory |
| `create_deep_agent` | Long-horizon / research harness |
| `tool`, `RunResult`, `ContextPolicy`, `spawn_specialist` | DX helpers |
| Checkpointers (`JsonFile`, `SQLite`, `InMemory`, `Postgres`) | Durability |
| `loomable.serve.mount_agent` / `mount_case` | AG-UI HTTP + SSE (optional `api_key=`) |
| Bundled skills via `resolve_skills` / `list_bundled_skills` | Progressive skills |

## Deprecated but kept

| Symbol | Replacement |
|--------|-------------|
| `create_research_agent` | `create_deep_agent(..., profile="research")` |

## Experimental / kernel-internal

| Surface | Notes |
|---------|--------|
| `loomable.kernel.registry.ExtensionRegistry` | Not wired into Agent discovery; may change |
| Flow optimizer (`loomable.flow.optimizer`) | Advanced; not part of beta DX |
| `discovery_core="research-slim"` | Schema-budget profile; defaults may shift |
| `loomable.sandbox` / `ShellTools` / Docker sandbox | Soft isolation; Docker experimental |
| Bundled `browser` skill | Assumes Lightpanda (or compatible) MCP |
| `loomable.codeindex` / `CodeTools` / `profile="code"` | Deep code; Alibaba zvec file store by default, pluggable `VectorBackend` |
| `loomable.retrieval` (`ingest`, `build_agentic_retriever`, chunk strategies) | Framework RAG; pluggable agentic stages |
| `AgenticRetriever` / `CompositeRetriever` | Rewrite / route / rerank / multi-corpus — all Protocol-pluggable |
| `FaissVectorBackend` / `open_vector_store(engine="faiss")` | Optional FAISS CPU/GPU ANN |
| Agent L3 / `LongTermStore()` | Defaults to Alibaba zvec at `.loomable/memory_zvec` |

## Beta limits

- **Workspace FS** is local (in-memory + optional disk mirror). No remote object-store backend in this cut.
- **Cancel** is cooperative (tool-loop / step boundaries), not hard-kill of in-flight provider HTTP.
- **Serve auth** is a shared API key baseline, not full RBAC/OIDC.
- PyPI publish may lag the git tag; install via git tag is supported for beta.

## Version policy

- Beta versions: `0.2.0bN`
- Breaking changes in beta: documented in CHANGELOG with a migration note
- `__version__` in `loomable.__init__` must match `pyproject.toml`
