# Loomable API stability

Loomable **0.2.0b0** is a **public beta**. The surface below is the supported
API. Prefer the high-level path; treat advanced Flow types as an escape hatch.

## Stable (supported in beta)

| Surface | Notes |
|---------|--------|
| `Agent`, `BuiltAgent`, `Team` | Core runnable agents |
| `Workflow`, `Step`, `Loop`, `Condition`, `Parallel_Group` | Durable multi-step |
| `Verifier`, `VerdictResult`, `FlowPaused` | Loop / HITL helpers |
| `Case`, `Board`, `WorkItem`, `build_case_workflow`, `map_specialists` | Goal + WorkItems board |
| `Memory`, `MemoryScope`, `ConversationMemory`, `UserMemory`, `KnowledgeMemory`, `open_session_store`, `open_vector_store` | Composable Agent memory |
| `WorkingMemory` | Workflow blackboard helper — use `Workflow(memory=True)` / `.store`; **not** for `Agent(memory=...)` |
| `create_deep_agent` | Long-horizon harness (`profile="research"` / `"code"`) |
| `tool`, `RunResult`, `ContextPolicy`, `spawn_specialist` | DX helpers |
| `plan_and_execute` | Used by `Workflow.map`; also importable at top level |
| `JsonFileCheckpointer`, `SQLiteCheckpointer`, `InMemoryCheckpointer`, `PostgresCheckpointer` | Durability |
| `loomable.serve.mount_agent` / `mount_case` | AG-UI HTTP + SSE (optional `api_key=`); NDJSON `/run/stream` on Agent only; disconnect → `cancel()` |
| `Agent.cancel` / `Workflow.cancel` / `Case.cancel` / `Team.cancel` | Cooperative cancel at tool-loop / step boundaries |
| `Agent(knowledge_base=)` / `create_deep_agent(knowledge_base=)` | Vector-DB knowledge base + optional `retrievers=` |
| Bundled skills via `resolve_skills` / `list_bundled_skills` | Progressive skills |

## Advanced escape hatch (not primary DX)

| Surface | Notes |
|---------|--------|
| `Flow`, `Node`, `Edge`, `MapNode`, `RouterNode` | Low-level graph |
| `TieredMemoryStore` | Internal blackboard for `Workflow(memory=True)` |
| `loomable.flow.helpers` (`sequential` / `parallel` / `route` / `coordinate`) | Prefer `Workflow` / `Team` |
| Flow optimizer / custom engines | Power users only |

## Experimental / may change

| Surface | Notes |
|---------|--------|
| `FlowClass` / `start` / `listen` / `router` | Decorator DSL; prefer `Workflow` — no examples yet |
| `loomable.kernel.registry.ExtensionRegistry` | Not in `kernel.__all__`; import from `loomable.kernel.registry` — not wired into Agent discovery |
| `discovery_core="research-slim"` | Schema-budget profile; defaults may shift |
| `loomable.sandbox` / `ShellTools` / Docker sandbox | Soft isolation; Docker experimental |
| Bundled `browser` skill | Assumes Lightpanda (or compatible) MCP |
| `loomable.codeindex` / `CodeTools` / `profile="code"` | Deep code; Alibaba zvec by default |
| `loomable.retrieval` | Framework RAG; pluggable agentic stages |
| `AgenticRetriever` / `CompositeRetriever` | Rewrite / route / rerank / multi-corpus |
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
