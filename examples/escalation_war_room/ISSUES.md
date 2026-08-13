# Issue log — Escalation War Room

Historical stress findings. Fixed items stay for audit; open items are next work.

## Open

| ID | Issue | Severity |
|----|-------|----------|
| WR-020 | Soft Team (`coordinate`) still LLM-dependent for tool counts | P2 |
| WR-021 | Optional `strict_require_tools` fail-closed mode | P2 |
| WR-022 | Workflow-level `require_tools` propagation | P2 |

## Fixed

| ID | Issue | Fix |
|----|-------|-----|
| WR-010 | `require_tools` name-only | Path constraints `write_file:output/x.md` |
| WR-011 | Empty final after tools | Recover from `write_json` |
| WR-012 | HITL not fluent | `Workflow.step(..., confirm=True)` + `approve()` |
| WR-013 | JsonFile approve race | Timestamp refresh on `put` |
| WR-014 | Plan steps not reaching MapNode | SharedState `state_updates` glue |
| WR-015 | No Case / board / AG-UI SSE | `Case`, `Board`, `astream_events`, `mount_*` |
