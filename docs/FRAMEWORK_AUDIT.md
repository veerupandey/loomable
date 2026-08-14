# Framework audit

Date: 2026-08-14. Case + AG-UI + Postgres pass (updated for Memory.compose).

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
| `Memory.compose(conversation=ConversationMemory(store=open_session_store(...)))` | L1/L2 (preferred) |
| `Memory.compose(user=UserMemory(...))` | L3 notes |
| `open_session_store("sqlite"\|"file"\|"postgres"\|"memory")` | Session store factory |
| `PostgresCheckpointer` | Workflow/Case resume |
| `PgVectorBackend` / `LongTermStore()` | L3 vectors |

Flat `session_store=` / `memory_backend=` remain available when not using compose.
Do not pass both `memory=` and flat store kwargs.

## Notes

- Accept loop re-runs synthesizer only (by design).
- `WorkingMemory` is for `Workflow(memory=True)` blackboards — not `Agent(memory=...)`.
