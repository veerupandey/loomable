"""Competitive plan: beat deepagents / Claude Code / OpenAI Agents / Agno / CrewAI.

Loomable wins by shipping progressive disclosure *and* enterprise primitives
(Agent | Team | Workflow | Case + AG-UI + Memory) on one loomable-only stack.
"""

# Beat industry frameworks — loomable deep agent

## Thesis

Industry leaders converge on **progressive disclosure** (metadata → search →
load/activate → call) plus long-horizon harnesses (plan, workspace, subagents).
Loomable already has the enterprise spine. We beat them by combining:

1. **Cheaper context** than Claude Code / OpenAI tool-search defaults on our
   deep surface (core + search, not 30+ schemas every turn).
2. **Research quality** deepagents can't match without our citation/accept/
   offload/workspace stack.
3. **One product model** (Agent/Team/Workflow/Case + Memory + AG-UI) instead of
   bolting LangGraph middleware onto a thin agent.

## Competitor map

| Competitor | Strength | Loomable counter |
|------------|----------|------------------|
| **deepagents** | Skills progressive disclosure, FS middleware, create_deep_agent DX | Same skill model + citations/verify/accept + Case + no LangGraph lock-in |
| **Claude Code / Agent SDK** | Tool search, MCP defer, skill metadata | Portable across providers; schema budget + MCP defer + skill catalog on any model |
| **OpenAI Agents** | Hosted tool_search / namespaces / defer_loading | Provider-agnostic discovery meta-tools + namespaces + lazy MCP |
| **Agno / CrewAI** | Multi-agent DX, marketing surface | Stronger long-horizon research + Memory.compose + Workflow/Case |

## Implementation waves

### P0 — Context budget parity (this wave)

- [x] Discovery meta-tools + MCP defer + per-turn schema refresh
- [x] **Core-tool allowlist**: advertise meta + core only; defer rest via `activate_tool`
- [x] **Metadata-only skills** under discovery (catalog in prompt; `load_skill` for body)
- [x] **`activate_tool` returns callable schema** in the tool result
- [x] Treat discovery tools as bookkeeping after deliverable-complete

### P1 — Discovery completeness

- [x] **Lazy MCP connect** (catalog from config → connect on activate):
      `Agent(lazy_mcp=...)` (defaults on under `discovery=True`) catalogs
      `ServerStub`s from `mcp_servers=` without opening a transport;
      `activate_mcp_server(server_id)` connects on demand and catalogs the
      server's tools as deferred `ToolStub`s. `search_mcp` surfaces
      unconnected servers with a hint to activate them.
- [x] **Mid-run catalog refresh** (re-scan skills / re-list MCP):
      `DiscoveryRuntime.refresh_capabilities()` (exposed as the
      `refresh_capabilities` tool) re-discovers `skill_roots` and re-lists
      tools from already-connected MCP sessions.
- [x] **Wire `discovery` into `task` / specialists**: `make_task_tools` /
      `spawn_specialist` accept `discovery`, `discovery_core_tools`,
      `defer_local_tools`, `lazy_mcp`, `activation_allowlist/denylist`;
      `create_deep_agent` passes them through by default so specialists
      sharing the parent's large toolkit don't blow their own schema budget.
- [x] **Activation allowlist / denylist** (`prefix*` wildcards supported) +
      pluggable `on_activate_check` HITL-style hook on `DiscoveryRuntime` /
      `Agent`, enforced by `activate_tool` before any MCP connect or local
      registration.
- [x] **`require_tools` aware of deferred tools**: the tool loop calls
      `discovery.ensure_tools_activated(missing_tool_names)` (auto
      `activate_mcp_server` + `activate_tool`) before nudging, so a deferred
      required tool doesn't need a manual `search_tools` round-trip first.

### P2 — Beat on quality & DX

- [x] **Tool namespaces / server groups** (OpenAI-style search surface):
      `CapabilityCatalog.namespaces` (`NamespaceStub`) — one auto `mcp:<id>`
      namespace per server plus caller-declared `Agent(tool_namespaces=[...])`
      groups; `search_namespaces` / `search_tools(namespace=...)` browse them.
- [x] **Better ranking**: `rank_bm25` (BM25-lite over catalog document
      frequency) backs `search_skills` / `search_tools` / `search_mcp` /
      `search_namespaces`; `rank_match` stays exported for back-compat.
- [x] **Skill resources level-3** (`references/`, `assets/`) on demand:
      `list_skill_resources` / `read_skill_resource` (path-traversal safe,
      restricted to `SKILL.md` + `scripts/`/`references/`/`assets/`). Demo
      resource: `loomable/skills/research/references/checklist.md`.
- [x] **Public cookbook**: `examples/deep_agent/02_progressive_discovery.py`
      — scripted `search_skills` → `load_skill` → `search_tools` →
      `activate_tool` walk, no network required.
- [ ] Research live gates vs deepagents baselines (token use, citation
      precision, wall time) — not run in this environment; needs a live
      model (Gemini/OpenAI) budget and a comparable deepagents harness to
      benchmark against. Tracked as the remaining P2 item.

## Success metrics

- **Schema tokens / turn** on `create_deep_agent(profile=research)` ↓ ≥50% vs all-eager
- **Live Gemini research brief**: `stop_reason=final`, accept ok, ≥1 cited source —
  **remaining**: not exercised in this environment (no live model credentials
  configured here); the scripted-provider equivalents in
  `tests/unit/test_deep_competitive.py` and
  `examples/deep_agent/02_progressive_discovery.py` cover the same code paths
  offline. Run `DEEP_AGENT_LIVE=1 GEMINI_API_KEY=... python
  examples/deep_agent/03_live_multimodal_research.py` to validate this metric
  against a live model.
- **Skill start**: research body absent until `load_skill` (unless `eager_skills`)
- **DX**: one API — `create_deep_agent` + skills; no second agent type

## Non-goals (this quarter)

- Hosted provider-native tool_search (OpenAI-only)
- Replacing Memory / Case with LangGraph checkpoints
- Merging `ExtensionRegistry` into DiscoveryRuntime without a design pass
