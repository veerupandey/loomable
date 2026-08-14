# Changelog

## 0.2.0b0 — public beta

### Added

- Stability policy ([docs/STABILITY.md](docs/STABILITY.md)), SECURITY.md, beta graduation plan
- `BuiltAgent.cancel()` / `Agent.cancel()` with active `RunContext` tracking
- SSE / stream client disconnect triggers cooperative cancel
- `mount_agent` / `mount_case` optional `api_key=` (Bearer or `X-API-Key`)
- `create_deep_agent(..., discovery_core="research-slim"|"research"|list)`
- Expanded CI: properties, toolkits, non-live integration; `feature/**` branches

### Changed

- Package status: **Beta** (`0.2.0b0`)
- `create_research_agent` emits `DeprecationWarning` (alias retained)

### Limits (documented)

- Local workspace FS only
- Cooperative cancel only
- Shared API-key serve auth (not full RBAC)

## 0.1.0 — alpha

- Agent · Team · Workflow · Case · AG-UI SSE
- Memory.compose, Postgres checkpointer
- Deep agent + progressive discovery (skills / tools / MCP)
