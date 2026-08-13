# Framework audit

Date: 2026-08-13. Case + AG-UI + Postgres pass.

## Gates

| Gate | Result |
|------|--------|
| `tests/unit/` | 1496+ passed |
| Gemini Case / Case SSE | verified |
| Docker Postgres 16 live E2E | 4/4 passed |

## Fixed

| Sev | Issue |
|-----|-------|
| P0 | Case SSE emptied board on resume → hydrate before snapshot |
| P0 | FastAPI `session_id` missed Case checkpoint thread → `bind_session` |
| P1 | Tool AG-UI ARGS/RESULT, engine `state_updates` / node durations |
| P1 | `require_tools` path substring → exact/suffix |
| P1 | Team SSE, Agent checkpointer → Case, board tools, ERROR→blocked |
| P1 | Board hydrate skipped `complete=True` checkpoints |
| P2 | `pytest.mark.unit` registered |

## Postgres

`pip install 'loomable[postgres]'` · `docker compose up -d`

| API | Protocol |
|-----|----------|
| `PostgresCheckpointer` | `Checkpointer` |
| `PostgresMemoryBackend` | `MemoryBackend` |
| `PgVectorBackend` | `VectorBackend` |

Tables auto-create. `user_id` scopes KV/vector rows.

## Notes

- Accept loop re-runs synthesizer only (by design).
- Case Workflow is sequential (plan → act → accept).
- `Agent.user_id` is metadata; tenant isolation uses Postgres `user_id` on backends.
