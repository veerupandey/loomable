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
| **OpenAI Agents** | Hosted tool_search / namespaces / defer_loading | Provider-agnostic discovery meta-tools; namespaces later |
| **Agno / CrewAI** | Multi-agent DX, marketing surface | Stronger long-horizon research + Memory.compose + Workflow/Case |

## Implementation waves

### P0 — Context budget parity (this wave)

- [x] Discovery meta-tools + MCP defer + per-turn schema refresh
- [x] **Core-tool allowlist**: advertise meta + core only; defer rest via `activate_tool`
- [x] **Metadata-only skills** under discovery (catalog in prompt; `load_skill` for body)
- [x] **`activate_tool` returns callable schema** in the tool result
- [x] Treat discovery tools as bookkeeping after deliverable-complete

### P1 — Discovery completeness

- Lazy MCP connect (catalog from config → connect on activate)
- Mid-run catalog refresh (re-scan skills / re-list MCP)
- Wire `discovery` into `task` / specialists
- Activation allowlist + HITL on activate
- `require_tools` aware of deferred tools (auto-activate or nudge activate)

### P2 — Beat on quality & DX

- Tool namespaces / server groups (OpenAI-style search surface)
- Better ranking (BM25 / embeddings optional)
- Research live gates vs deepagents baselines (token use, citation precision, wall time)
- Skill resources level-3 (`references/`, `assets/`) on demand
- Public cookbook: “progressive deep research on loomable only”

## Success metrics

- **Schema tokens / turn** on `create_deep_agent(profile=research)` ↓ ≥50% vs all-eager
- **Live Gemini research brief**: `stop_reason=final`, accept ok, ≥1 cited source
- **Skill start**: research body absent until `load_skill` (unless `eager_skills`)
- **DX**: one API — `create_deep_agent` + skills; no second agent type

## Non-goals (this quarter)

- Hosted provider-native tool_search (OpenAI-only)
- Replacing Memory / Case with LangGraph checkpoints
- Merging `ExtensionRegistry` into DiscoveryRuntime without a design pass
