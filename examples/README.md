# Loomable Examples

## When to use what

| I need… | Use… | Example |
|---------|------|---------|
| One agent | `Agent` | `agents/01_hello_world.py` |
| Tools | `Agent` + `@tool` | `agents/02_with_tools.py` |
| Structured JSON | `response_model` | `agents/03_structured_io.py` |
| Conversation memory | `session_id` (+ optional `Memory.compose`) | `agents/04_with_memory.py`, `memory/01_session_memory.py` |
| User / long-term memory | `Memory.compose` + `UserMemory` | `memory/02_user_memory.py` |
| Shared memory across a Workflow | `Workflow(..., memory=True)` | `memory/03_workflow_shared_memory.py` |
| Compose Postgres L1/L2 + zvec L3 | `Memory.compose` + Postgres / zvec | `memory/04_compose_postgres_zvec.py` |
| Unified Memory.compose | `Memory.compose(conversation=..., user=..., knowledge=...)` | `memory/05_compose_memory.py` |
| Claim / custom scopes | `scopes={"claim_id": "..."}` | `memory/06_claim_scopes.py` |
| Deep agent | `create_deep_agent(profile=...)` | `deep_agent/` |
| Progressive discovery | `search_skills` / `load_skill` / `activate_tool` | `deep_agent/02_progressive_discovery.py` |
| Sandbox + browser skill/MCP | `code_exec` / `shell` / `skills=["browser"]` | `deep_agent/06_sandbox_browser.py` |
| Deep code (repo index + coding) | `profile="code"` / `CodeIndex` | `deep_agent/07_deep_code.py` |
| MCP servers | `Agent(mcp_servers=[...])` | `advanced/01_mcp_servers.py` |
| Conditional branches | `Workflow.branch` | `advanced/02_workflow_branch.py` |
| Checkpoint / resume | `Workflow` + `JsonFileCheckpointer` | `advanced/03_checkpointing.py`, `escalation_war_room/05_checkpoint_resume.py` |
| Multimodal I/O | `Image` / tool media | `advanced/04_multimodal.py` |
| Build retrievers (docs/code) | `build_retriever` | `advanced/05_build_retriever.py` |
| Agentic retriever (pluggable) | `ingest` + `build_agentic_retriever` | `advanced/06_agentic_retriever.py` |
| Ship any Retriever ABC | `Agent(retrievers=[...])` | `advanced/07_ship_any_retriever.py` |
| Complex multi-format RAG | `Agent(knowledge_base={docs, code})` + Workflow | `advanced/08_complex_agentic_rag.py` |
| RAG (searchable vector DB) | `knowledge_base=` on Agent / create_deep_agent | `agents/07_knowledge_base.py` |
| RAG (Team / Workflow inherit KB) | `Team(..., knowledge_base=...)` | `agents/08_team_knowledge_base.py` |
| RAG (passive snippets) | `knowledge` + embedder | `agents/05_with_knowledge.py` |
| Production hardening | resilience + hooks | `agents/06_production.py` |
| Specialists | `Team` / subagents | `subagents/` |
| Quality gate | Agent verifier / `Workflow.loop` | `patterns/01_retry_loop.py` |
| Multi-step process | `Workflow.step` | `patterns/02_pipeline.py` |
| Parallel branches | `Workflow.parallel` | `patterns/03_fan_out.py` |
| Intent router | `Team(mode="route")` | `patterns/04_router.py` |
| Plan → map → synth | `Workflow.map` | `patterns/05_plan_execute.py` |
| Nested composition | nested `Workflow` / parallel | `patterns/06_nested_composition.py` |
| Goal + WorkItems board | `Case` | `escalation_war_room/10_case.py` |
| AG-UI SSE (Agent) | `mount_agent` / `astream_events` | `escalation_war_room/12_agent_agui_sse.py` |
| AG-UI SSE (Case) | `mount_case` | `escalation_war_room/11_case_sse.py` |
| Postgres checkpoints | `PostgresCheckpointer` | `advanced/03_checkpointing.py` (swap JsonFile → Postgres) |
| Full SEV war room | tools → workflow → Case → SSE | `escalation_war_room/` |
| Simple real-world Q&A | toolkits | `simple_use_cases/` |
| Custom skill package | `skills/` layout | `skills/weather-lookup/` (load via `create_deep_agent(..., skills=[...])`) |

## Structure

```
examples/
├── agents/                 # Agent API progression (incl. knowledge_base)
├── deep_agent/             # create_deep_agent + discovery + live Gemini gate
├── subagents/              # Team / delegation
├── patterns/               # Workflow / Team patterns (step, parallel, route, map)
├── memory/                 # session → compose → scopes
├── advanced/               # MCP, Workflow branch, checkpoints, multimodal, retrieval
├── simple_use_cases/       # toolkit-driven Q&A
├── escalation_war_room/    # full SEV demo (Case / SSE / HITL)
└── skills/                 # sample skill packages (weather-lookup)
```

## Running

```bash
# Copy .env.example → .env in the repo root (gitignored; never commit .env)
# GEMINI_API_KEY=...  or OPENAI_API_KEY / Azure vars

python examples/agents/01_hello_world.py
python examples/agents/07_knowledge_base.py        # live LLM + knowledge_base=
python examples/agents/08_team_knowledge_base.py   # live Team KB inherit
python examples/advanced/05_build_retriever.py     # live LLM + retrievers=
python examples/advanced/03_checkpointing.py       # live Workflow + checkpointer
```

## Design principles

1. **One concept per file**
2. **Run it, see one clear output**
3. **Docstring says when to use the pattern**
4. **Progressive** — unique numeric prefixes; `01` is simplest in each folder
5. **Live models** — examples call a real provider via `examples/_provider.py`
   (`GEMINI_API_KEY` / OpenAI / Azure). Copy `.env.example` → `.env`.
6. **Agents understand prior output** — put `Agent`s on `Workflow.step` / `Team`.
   Do **not** add glue functions that parse `AgentOutput` between steps; the
   framework already passes the previous output as the next input.
