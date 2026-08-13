# Framework audit — main (post Case + AG-UI SSE)

Date: 2026-08-13 (updated). Rigorous review after merging Case/SSE onto `main`.

## Test results (this pass)

| Suite | Result |
|-------|--------|
| Case / stream / FastAPI SSE / MCP adapter | covered by unit suites |
| Audit bugfix regressions (`test_audit_bugfixes.py` + related) | targeted green gate |
| Live Gemini Agent/Case SSE | previously passed on main |

## Bugs found & fixed

| Sev | Issue | Fix |
|-----|-------|-----|
| P0 | `Agent(mode="case")` rebuilt Case every call → empty board | Cache `_case` via `_get_case()` |
| P0 | FastAPI/`Case` got `AgentInput` without text coercion | Coerce via `_input_text` / `Case._coerce_task_text` |
| P1 | `session_id` accepted but unused in FastAPI | Apply session to agent/case; pass into `astream_events` |
| P1 | `Workflow.state` empty unless caller passed `RunContext` | Always create/capture context SharedState |
| P1 | `BuiltAgent.astream_events` did not cancel on consumer break | Cancel runner task like Case/Flow |
| P1 | MCP SDK drift: `isError`/`mimeType`/`Server.list_tools` | `is_error` / `mime_type` / `MCPServer.add_tool` |
| P1 | Tool AG-UI: `TOOL_CALL_ARGS` / `TOOL_CALL_RESULT` not emitted | Emit START/ARGS before dispatch and RESULT/END after; skip legacy duplicate via `agui_skip` |
| P1 | Case board not rehydrated from checkpoint SharedState | `_hydrate_board_from_checkpoint` + `_hydrate_board_from_state` |
| P1 | Flow stream `session_id` labeled events only | Temporarily bind `Flow._session_id` for stream runs (checkpoints use stream session) |
| P2 | Parallel/hierarchical node durations = superstep wall time | Emit `node_start`/`node_end` inside each worker factory |
| P2 | Parallel/hierarchical ignore `metadata["state_updates"]` | `_apply_state_updates` in barrier / worker commit / manager |
| P2 | `require_tools` path match is substring | `_path_constraint_met` exact or `*/required` suffix |
| P2 | Team has no `astream_events` | Soft modes → Agent SSE; hard modes emit RUN_* + NODE_* per member |
| P2 | Unknown `pytest.mark.unit` warnings | Register `unit` / `integration` markers in `pyproject.toml` |

## Deferred

- Postgres / durable vector memory (explicitly deferred)

## Mistakes / risks to watch

1. **Docs ahead of code** — keep AG-UI ARGS/RESULT and session routing aligned with StreamBridge/BuiltAgent (now wired).
2. **Case via Agent vs bare Case** — must keep cached Case; never `from_agent` per request.
3. **Engine asymmetry** — Case Workflow is sequential today; plan glue breaks if someone forces parallel engine (state_updates merge now exists, but Case planner glue still assumes sequential step order).
4. **Env test deps** — install with `pip install -e ".[dev]"`.

## Architecture confirmation

- Agent / Flow / Case / Workflow / Team share **Runnable** (`arun` → `RunResult`) and AG-UI **SSE** vocabulary.
- **SharedState** is the Workflow/Flow blackboard (plan_steps, map, board dict, node outputs).
- Standalone Agent tool-loops do not create SharedState unless nested in a Flow.
