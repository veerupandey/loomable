# Loomable Examples

## When to use what

| I need...                              | Use...                               | Example                       |
|----------------------------------------|--------------------------------------|-------------------------------|
| One agent answering questions          | `Agent`                              | `agents/01_hello_world.py`    |
| One agent with tools                   | `Agent + @tool`                      | `agents/02_with_tools.py`     |
| Structured JSON output                 | `Agent + response_model`             | `agents/03_structured_io.py`  |
| Multi-turn conversation memory         | `Agent + session_id`                 | `agents/04_with_memory.py`    |
| Answers from your documents (RAG)      | `Agent + knowledge + embedder`       | `agents/05_with_knowledge.py` |
| Production hardening                   | `Agent + resilience + hooks`         | `agents/06_production.py`     |
| Multiple specialists on one task       | `Agent(subagents=[...])`             | `subagents/01_simple_delegation.py` |
| Subagents sharing memory               | Shared `session_id`                  | `subagents/02_with_memory_sharing.py` |
| Multi-level delegation                 | Nested `subagents`                   | `subagents/03_nested_delegation.py` |
| Explicit orchestration mode            | `Team(mode="coordinate")`            | `subagents/04_team_modes.py`  |
| Quality-checked output                 | `Agent + verifier`                   | `patterns/01_retry_loop.py`   |
| Steps that feed into each other        | `sequential(a, b, c)`                | `patterns/02_pipeline.py`     |
| Same task, multiple perspectives       | `parallel(a, b, c)`                  | `patterns/03_fan_out.py`      |
| Route to different agents by intent    | `route(chooser, {...})`              | `patterns/04_router.py`       |
| Dynamic task decomposition             | `plan_and_execute(...)`              | `patterns/05_plan_execute.py` |
| Flows inside flows                     | Nested `sequential`/`parallel`       | `patterns/06_nested_composition.py` |
| Memory across conversations            | `Agent + session_id + user_id`       | `memory/02_user_memory.py`    |
| Shared state in a flow                 | `TieredMemoryStore`                  | `memory/03_flow_shared_memory.py` |
| MCP server tools                       | `Agent + mcp_servers`                | `advanced/01_mcp_servers.py`  |
| Custom graph with conditions           | `Flow + Node + Edge`                 | `advanced/02_custom_flow.py`  |
| Durable state / pause-resume           | `CheckpointStore`                    | `advanced/03_checkpointing.py`|
| Multimodal: image input, tool media, feedback | `Agent + multimodal=True + @tool → Image` | `advanced/04_multimodal.py`   |
| Simple real-world Q&A / docs / tools           | `Agent + WebSearch/File/PDF/PPT`         | `simple_use_cases/`           |
| Tough SEV war-room exam (tools→docs→images)    | `Agent + tools + File/PDF/PPT + multimodal` | `escalation_war_room/`     |

## Structure

```
examples/
├── agents/               # Single agent patterns (start here)
├── simple_use_cases/     # News, research, structured I/O, docs, tools
├── escalation_war_room/  # Tough real-world ladder (Phase 1 done)
├── subagents/            # Multi-agent delegation
├── patterns/             # Flow composition patterns
├── memory/               # Memory and persistence
└── advanced/             # MCP, custom flows, multimodal
```

## Running

All examples read credentials from a `.env` file in the project root:

```bash
# .env
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

Run any example:

```bash
python examples/agents/01_hello_world.py
```

## Design principles

1. **One concept per file** — no loops over multiple queries
2. **Run it, see one output** — no `for query in queries:` patterns
3. **Self-explanatory** — docstring says WHEN to use this pattern
4. **Progressive** — within each folder, 01 is simplest

## Display utilities

Most examples use `loomable.display` for pretty output:

```python
from loomable.display import pp, delegation_outputs, step_outputs, show_graph

# Pretty-print any result (auto-detects agent/flow/loop/subagent)
pp(result)

# Access individual subagent outputs by name
outputs = delegation_outputs(result)
print(outputs["researcher"])

# Access individual flow step outputs by node name
steps = step_outputs(result)
print(steps["node_0"])

# Visualize a flow graph (terminal: Mermaid, Jupyter: interactive SVG)
show_graph(my_flow)
```
