# Framework audit — industry-grade pass (Case + Postgres)

Date: 2026-08-13 (second pass). Re-audited after AG-UI/tool/engine fixes; hardened
Case durability + shipped Postgres backends.

## Live / unit gates

| Suite | Result |
|-------|--------|
| Full `tests/unit/` | targeted green (see CI / local gate) |
| Case / SSE / FastAPI / Postgres fake-pool | covered |
| Live Gemini Case (`10_case.py`) | run on this branch |

## Bugs found & fixed (this pass)

| Sev | Issue | Fix |
|-----|-------|-----|
| P0 | `Case.astream_events` resumed with empty board snapshot and wiped SharedState | Hydrate before first STATE_SNAPSHOT; re-hydrate after stream |
| P0 | FastAPI `session_id` did not bind Case Workflow checkpoint thread | `Case.bind_session` + FastAPI `_apply_session` updates `_kwargs` / live Workflow |
| P1 | `Agent(mode="case")` dropped checkpointer | `Agent(checkpointer=...)` → `Case.from_agent` |
| P1 | Board cards marked `done` even on soft ERROR outputs | Mark `blocked` when step text starts with `ERROR` |
| P1 | Board tools not wired into Case workers | Auto-attach `board_tools` in `build_case_workflow` |
| P1 | Docs claimed Postgres backends that did not exist | Implement + document correctly |

## Postgres (new)

| Component | Protocol | Module |
|-----------|----------|--------|
| `PostgresCheckpointer` | `Checkpointer` | `loomable.persist.postgres` |
| `PostgresMemoryBackend` | `MemoryBackend` | `loomable.providers.backends.postgres` |
| `PgVectorBackend` | `VectorBackend` | `loomable.providers.backends.postgres` |

Install: `pip install 'loomable[postgres]'` (asyncpg). Tables auto-create on first use.
`user_id` scopes KV/vector rows. Vectors use `DOUBLE PRECISION[]` + cosine (no pgvector extension required).

## Still watch

1. Accept loop still re-runs synthesizer only (by design today) — full re-plan/re-dispatch is a product choice.
2. `Agent.user_id` is still app-level metadata; multi-tenant isolation for Agent turns uses session_store / Postgres KV scoping explicitly.
3. Case Workflow remains sequential (plan → act → accept).

## Architecture confirmation

- Agent / Team / Workflow / Flow / Case share Runnable + AG-UI SSE.
- Durable resume: JsonFile / SQLite / **Postgres** checkpointers.
- Durable memory: ShortTermStore + LongTermStore plug Postgres backends.
