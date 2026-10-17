# Learn Loomable by example

Start with one agent, then follow the examples for the features you need. Run the commands from the repository root. Model examples use OpenAI or Gemini and small local fixtures. Examples marked **offline** run without a provider key.

## Setup

```bash
python -m pip install -e '.[examples]'
export LOOMABLE_PROVIDER=openai
# Set OPENAI_API_KEY in your shell or in the ignored repo-root .env file.
python -m examples.agents.01_hello_world
```

OpenAI defaults to `gpt-5.4-mini`. Set `OPENAI_MODEL` to select another model. For Gemini, set `LOOMABLE_PROVIDER=gemini` and `GEMINI_API_KEY`, and optionally set `GEMINI_MODEL`. The Gemini default is `gemini-flash-latest`. Use [`.env.example`](../.env.example) as a template.

Run a module in the tables with `python -m examples.<module>`. For example, `agents.02_with_tools` runs as `python -m examples.agents.02_with_tools`.

## One agent

| Learn | Module | What to look for |
|---|---|---|
| Instructions and `arun` | [`agents.01_hello_world`](agents/01_hello_world.py) | One answer in `result.text`. |
| Typed tools | [`agents.02_with_tools`](agents/02_with_tools.py) | The model calls a read-only inventory function and reports 12 units. |
| Structured output | [`agents.03_structured_io`](agents/03_structured_io.py) | `response_model` validates a billing ticket and exposes it as `result.structured`. |
| Streaming | [`agents.05_streaming`](agents/05_streaming.py) | Text arrives as deltas. The final chunk carries the completed result and usage. |
| Retry and durable jobs | [`agents.06_production`](agents/06_production.py) | A fresh worker retrieves the same completed run through `arun_once`. |
| Image input | [`advanced.04_multimodal`](advanced/04_multimodal.py) | The model describes a sample dashboard. Pass your own image path as an argument. |

## Memory and knowledge

| Learn | Module | What to look for |
|---|---|---|
| Conversation history | [`memory.01_session_memory`](memory/01_session_memory.py) | A rebuilt agent recalls order `QZ-8421` from a saved session. |
| Scoped user notes | [`memory.02_user_memory`](memory/02_user_memory.py) | A different session for the same user recalls a saved preference. Uses real embeddings. |
| Shared workflow memory | [`memory.03_workflow_shared_memory`](memory/03_workflow_shared_memory.py) | Callable steps write and recall a record from the workflow blackboard. **Offline.** |
| Searchable knowledge base | [`agents.07_knowledge_base`](agents/07_knowledge_base.py) | An agent searches a local handbook for the 45-day return policy. Uses real embeddings. |

Conversation history belongs to a session. User notes can span sessions within the same scope. A knowledge base provides searchable application documents. For tenant or claim isolation and persistent backends, see [Memory](../docs/API.md#memory). To share notes with a specialist, give it the same note store and trusted user scope. Pass `knowledge_base=` to a Team to share a searchable collection with its members.

## Workflows and teams

Use `Workflow` when your application defines the steps and edges. Use `Team` to combine specialist agents under a named orchestration mode. Both can appear in the same app.

| Learn | Module |
|---|---|
| Run parallel reviews and chain their outputs into a final step | [`patterns.02_pipeline`](patterns/02_pipeline.py) |
| Choose a yes/no branch | [`advanced.02_workflow_branch`](advanced/02_workflow_branch.py) |
| Route to named branches with `Command` | [`patterns.08_route_command`](patterns/08_route_command.py) · **offline** |
| Read named state, handle step failures, and verify output | [`patterns.07_graph_engineering`](patterns/07_graph_engineering.py) · **offline** |
| Loop until an output passes a check | [`patterns.01_retry_loop`](patterns/01_retry_loop.py) |
| Map a typed plan over independent tasks | [`patterns.05_plan_execute`](patterns/05_plan_execute.py) |
| Pause for approval and reopen a checkpoint | [`advanced.03_checkpointing --approve`](advanced/03_checkpointing.py) |
| Delegate to specialists from an Agent | [`subagents.01_simple_delegation`](subagents/01_simple_delegation.py) |
| Try the team modes | [`subagents.04_team_modes --mode route`](subagents/04_team_modes.py) |
| Plan and execute a Case with an acceptance check | [`advanced.09_case`](advanced/09_case.py) |

In the team demo, `route` picks one specialist, `coordinate` delegates and synthesizes, `broadcast` runs everyone, `sequential` chains their outputs, and `tasks` tracks work in a checklist. A Team can also contain another Team, and a Workflow can contain another Workflow as a step. The checkpoint demo pauses by default. The `--approve` flag permits the final write to a temporary file.

## Extend and serve

| Learn | Example | Extra setup |
|---|---|---|
| Query SQLite with a built-in toolkit | [`advanced.11_sql_toolkit`](advanced/11_sql_toolkit.py) | None |
| Connect tools from a local stdio server | [`advanced.01_mcp_servers`](advanced/01_mcp_servers.py) | Install `.[mcp]` |
| Build or plug in a retriever | [Build from documents](advanced/05_build_retriever.py), [connect a custom source](advanced/07_ship_any_retriever.py) | Experimental retrieval API |
| Load a custom skill with a script tool | [`advanced.12_custom_skill`](advanced/12_custom_skill.py) | Uses a [bundled historical fixture](skills/weather-lookup/) |
| Use a research, coding, or sandbox profile | [Deep Agent examples](deep_agent/README.md) | The research demo uses external web search |
| Serve HTTP and AG-UI events | [`advanced.10_serve`](advanced/10_serve.py) | Install `.[serve]` |

```bash
python -m pip install -e '.[examples,serve]'
python -m uvicorn examples.advanced.10_serve:app --host 127.0.0.1 --port 8000
```

In a second shell, use `/agent/run` for JSON or `/agent/run/events` for AG-UI SSE.

```bash
curl -N http://127.0.0.1:8000/agent/run/events \
  -H 'Content-Type: application/json' \
  -d '{"input_text":"Explain database indexes in one sentence."}'
```

## Complete applications

- [Incident triage](incident_triage/README.md) reads signals and a runbook, validates a structured handoff, and includes a two-worker PostgreSQL staging service.
- [Research and memory agent](agents/09_research_memory_agent.py) combines conversation and user memory, document retrieval, and specialist delegation. Install `.[toolkits]` and pass `--question` or `--pdf` to exercise the research tools. See the [factory](agents/research_memory_agent.py) to adapt it.
- [API reference](../docs/API.md) covers configuration options and extension points beyond these examples.
