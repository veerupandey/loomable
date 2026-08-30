# Loomable Examples

## When to use what

| I need… | Use… | Example |
|---------|------|---------|
| One agent | `Agent` | `agents/01_hello_world.py` |
| Tools | `Agent` + `@tool` | `agents/02_with_tools.py` |
| Structured JSON | `response_model` | `agents/03_structured_io.py` |
| Conversation memory | `session_id` (+ optional `Memory.compose`) | `agents/04_with_memory.py`, `memory/01_session_memory.py` |
| User / long-term memory | `Memory.compose` + `UserMemory` | `memory/02_user_memory.py` |
| Shared Workflow blackboard (callable steps) | `Workflow(..., memory=True)` | `memory/03_workflow_shared_memory.py` |
| Compose Postgres L1/L2 + zvec L3 | `Memory.compose` + Postgres / zvec | `memory/04_compose_postgres_zvec.py` |
| Unified Memory.compose | `Memory.compose(conversation=..., user=..., knowledge=...)` | `memory/05_compose_memory.py` |
| Claim / custom scopes | `scopes={"claim_id": "..."}` | `memory/06_claim_scopes.py` |
| Deep agent | `create_deep_agent(profile=...)` | `deep_agent/` |
| Deep research (live) | `profile="research"` | `deep_agent/01_research.py` |
| Deep code (live) | `profile="code"` / `repo=` | `deep_agent/02_code.py` |
| Sandbox + optional browser MCP | `code_exec` / `shell` | `deep_agent/03_sandbox.py` |
| MCP servers | `Agent(mcp_servers=[...])` | `advanced/01_mcp_servers.py` |
| Binary Workflow fork (`when` / `then` / `else_`) | `Workflow.branch` | `advanced/02_workflow_branch.py` |
| N-way Workflow arms + `Command(goto=…)` | `Workflow.route` | `patterns/08_route_command.py` |
| LLM picks one Team specialist | `Team(mode="route")` | `patterns/04_router.py` |
| Checkpoint persist | `Workflow` + `JsonFileCheckpointer` | `advanced/03_checkpointing.py` |
| Checkpoint kill / resume | `Workflow` + incomplete checkpoint + `resume=True` | `escalation_war_room/05_checkpoint_resume.py` |
| Multimodal I/O | `Image` / tool media | `advanced/04_multimodal.py` |
| RAG (searchable vector DB) | `knowledge_base=` on Agent / create_deep_agent | `agents/07_knowledge_base.py` |
| RAG (Team / Workflow inherit KB) | `Team(..., knowledge_base=...)` | `agents/08_team_knowledge_base.py` |
| RAG (passive snippets) | `knowledge` + embedder | `agents/05_with_knowledge.py` |
| Complex multi-format RAG | `Agent(knowledge_base={docs, code})` + Workflow | `advanced/08_complex_agentic_rag.py` |
| Custom retriever builders (experimental) | `build_retriever` / `Agent(retrievers=)` | `advanced/05_build_retriever.py`, `06_agentic_retriever.py`, `07_ship_any_retriever.py` |
| Production hardening | resilience + hooks | `agents/06_production.py` |
| L1/L2/L3 + PDF + web + subagent fan-out | `build_research_agent()` | `agents/09_research_memory_agent.py` |
| Specialists / Team modes | `Team` / subagents | `subagents/` (`04_team_modes.py`) |
| Agent verifier or open `Workflow.loop` | `verifier=` / `.loop` | `patterns/01_retry_loop.py` |
| Bounded generate→check→repair in a pipeline | `Workflow.verify` | `patterns/07_graph_engineering.py` |
| Multi-step process | `Workflow.step` | `patterns/02_pipeline.py` |
| Parallel Agents (happy path) | `Workflow.parallel` | `patterns/03_fan_out.py` |
| Parallel + `on_failure` / `reads=` / `.verify` | `Step` + graph knobs | `patterns/07_graph_engineering.py` |
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
├── deep_agent/             # live create_deep_agent (research, code, sandbox)
├── subagents/              # Team / delegation
├── patterns/               # Workflow / Team patterns (step, parallel, map, route, verify)
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
python examples/deep_agent/01_research.py           # live research brief
```

## Routing — pick one API

| Need | API | Example |
|------|-----|---------|
| Yes/no fork after a classify step | `Workflow.branch` | `advanced/02_workflow_branch.py` |
| Named arms (quick/full/human) + optional `Command` | `Workflow.route` | `patterns/08_route_command.py` |
| LLM chooses one Team member for the whole query | `Team(mode="route")` | `patterns/04_router.py` |

`Team(mode="route")` is **not** Workflow control flow. `Workflow.branch` is binary;
use `Workflow.route` when you have three or more arms or need `Command(goto=, update=)`.

## Design principles

1. **One concept per file**
2. **Run it, see one clear output**
3. **Docstring says when to use the pattern** (and what *not* to confuse it with)
4. **Progressive** — unique numeric prefixes; `01` is simplest in each folder
5. **Live models** — examples call a real provider via `examples/_provider.py`
   (`GEMINI_API_KEY` / OpenAI / Azure). Copy `.env.example` → `.env`.
   Offline-safe demos: `patterns/07_graph_engineering.py`, `patterns/08_route_command.py`.
6. **Agents understand prior output** — put `Agent`s on `Workflow.step` / `Team`.
   Do **not** add glue functions that parse `AgentOutput` between steps; the
   framework already passes the previous output as the next input.
