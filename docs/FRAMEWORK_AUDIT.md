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

## Postgres + Agent memory

`pip install 'loomable[postgres]'` · `docker compose up -d`

| API | Role |
|-----|------|
| `open_session_store("sqlite"\|"file"\|"postgres"\|"memory")` | L1/L2 via `session_store=` |
| `Agent(..., memory_backend=...)` | L1/L2 via any `MemoryBackend` |
| `PostgresCheckpointer` | Workflow/Case resume |
| `PgVectorBackend` | L3 vectors for `NoteStore` / `LongTermStore` |

Custom backends: implement `MemoryBackend` (`read`/`write`/`delete`/`exists`).

## Notes

- Accept loop re-runs synthesizer only (by design).
- Case Workflow is sequential (plan → act → accept).
- `Agent.user_id` is metadata; tenant isolation uses Postgres `user_id` on backends.
