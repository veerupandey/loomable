# Loomable Examples

## When to use what

| I need… | Use… | Example |
|---------|------|---------|
| One agent | `Agent` | `agents/01_hello_world.py` |
| Tools | `Agent` + `@tool` | `agents/02_with_tools.py` |
| Structured JSON | `response_model` | `agents/03_structured_io.py` |
| Conversation memory | `session_id` + `session_store` / `memory_backend` | `agents/04_with_memory.py`, `memory/02_user_memory.py` |
| RAG | `knowledge` + embedder | `agents/05_with_knowledge.py` |
| Production hardening | resilience + hooks | `agents/06_production.py` |
| Specialists | `Team` / subagents | `subagents/` |
| Quality gate | `Loop` / verifier | `patterns/01_retry_loop.py` |
| Multi-step process | `Workflow` / `sequential` | `patterns/02_pipeline.py` |
| Parallel branches | `parallel` | `patterns/03_fan_out.py` |
| Plan → map → synth | `plan_and_execute` | `patterns/05_plan_execute.py` |
| Goal + WorkItems board | `Case` | `escalation_war_room/10_case.py` |
| AG-UI SSE (Agent) | `mount_agent` / `astream_events` | `escalation_war_room/12_agent_agui_sse.py` |
| AG-UI SSE (Case) | `mount_case` | `escalation_war_room/11_case_sse.py` |
| Postgres memory / checkpoints | `PostgresCheckpointer` + backends | `memory/02_user_memory.py` |
| Full SEV war room | tools → workflow → Case → SSE | `escalation_war_room/` |
| Simple real-world Q&A | toolkits | `simple_use_cases/` |

## Structure

```
examples/
├── agents/
├── subagents/
├── patterns/
├── memory/
├── advanced/
├── simple_use_cases/
└── escalation_war_room/
```

## Running

```bash
# .env in repo root
GEMINI_API_KEY=...
# or OPENAI_API_KEY / Azure vars

python examples/agents/01_hello_world.py
```

## Design principles

1. **One concept per file**
2. **Run it, see one clear output**
3. **Docstring says when to use the pattern**
4. **Progressive** — `01` is simplest in each folder
