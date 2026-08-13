# Framework audit — main (post Case + AG-UI SSE)

Date: 2026-08-13. Rigorous review after merging Case/SSE onto `main`.

## Test results (this pass)

| Suite | Result |
|-------|--------|
| Case / stream / FastAPI SSE / MCP adapter | **26 passed** |
| Core enterprise + require_tools (earlier) | **82 passed** |
| Live Gemini Agent SSE (`12_agent_agui_sse.py`) | **passed** |
| Live Gemini Case SSE (`11_case_sse.py`) | **passed** (board STATE_DELTA + coerced text input) |

## Bugs found & fixed (this branch)

| Sev | Issue | Fix |
|-----|-------|-----|
| P0 | `Agent(mode="case")` rebuilt Case every call → empty board | Cache `_case` via `_get_case()` |
| P0 | FastAPI/`Case` got `AgentInput` without text coercion | Coerce via `_input_text` / `Case._coerce_task_text` |
| P1 | `session_id` accepted but unused in FastAPI | Apply session to agent/case; pass into `astream_events` |
| P1 | `Workflow.state` empty unless caller passed `RunContext` | Always create/capture context SharedState |
| P1 | `BuiltAgent.astream_events` did not cancel on consumer break | Cancel runner task like Case/Flow |
| P1 | MCP SDK drift: `isError`/`mimeType`/`Server.list_tools` | `is_error` / `mime_type` / `MCPServer.add_tool` |

## Still open (improvement backlog)

| Sev | Issue | Suggested fix |
|-----|-------|---------------|
| P1 | Tool AG-UI: `TOOL_CALL_ARGS` / `TOOL_CALL_RESULT` documented but not emitted | Emit around real tool dispatch with call ids |
| P1 | Case board not rehydrated from checkpoint SharedState | `Board.from_dict` on resume |
| P1 | Flow stream `session_id` labels events only (checkpoints use Flow's id) | Bind thread id for stream runs |
| P2 | Parallel/hierarchical node durations = superstep wall time | Emit start/end inside each worker |
| P2 | Parallel/hierarchical ignore `metadata["state_updates"]` | Share sequential merge logic |
| P2 | `require_tools` path match is substring | Normalize + sandbox equality/suffix |
| P2 | Team has no `astream_events` | Bridge hard-mode member events |
| P2 | Unknown `pytest.mark.unit` warnings | Register mark in `pyproject.toml` |

## Mistakes / risks to watch

1. **Docs ahead of code** — AG-UI ARGS/RESULT and “session routing” were advertised before fully wired (session now partially fixed).
2. **Case via Agent vs bare Case** — must keep cached Case; never `from_agent` per request.
3. **Engine asymmetry** — Case Workflow is sequential today; plan glue breaks if someone forces parallel engine.
4. **Env test deps** — `pytest-httpx`, `beautifulsoup4` needed for full unit green; declare in `[project.optional-dependencies] dev` (already listed — install with `pip install -e ".[dev]"`).

## Architecture confirmation

- Agent / Flow / Case / Workflow share **Runnable** (`arun` → `RunResult`) and AG-UI **SSE** vocabulary.
- **SharedState** is the Workflow/Flow blackboard (plan_steps, map, board dict, node outputs).
- Standalone Agent tool-loops do not create SharedState unless nested in a Flow.
