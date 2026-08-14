# Issue log — Escalation War Room

Historical stress findings. Fixed items stay for audit; open items are next work.

## Open

_None. Stress findings from the war-room exams are closed or promoted into the framework._

## Fixed

| ID | Issue | Fix |
|----|-------|-----|
| WR-010 | `require_tools` name-only | Path constraints `write_file:output/x.md` |
| WR-011 | Empty final after tools | Recover from `write_json` |
| WR-012 | HITL not fluent | `Workflow.step(..., confirm=True)` + `approve()` |
| WR-013 | JsonFile approve race | Timestamp refresh on `put` |
| WR-014 | Plan steps not reaching MapNode | SharedState `state_updates` glue |
| WR-015 | No Case / board / AG-UI SSE | `Case`, `Board`, `astream_events`, `mount_*` |
| WR-020 | Soft Team coordinate LLM-only | Auto `require_tools` on delegates + member fallback |
| WR-021 | No fail-closed require_tools | `strict_require_tools=True` raises `RequireToolsError` |
| WR-022 | Workflow-level require_tools | `Workflow(require_tools=...)` / `.step(..., require_tools=)` |
