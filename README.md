<div align="center" id="top">
  <h1>loomable</h1>
  <p>Enterprise AI agents — Agent · Team · Workflow · Case · AG-UI SSE</p>
  <p>
    <a href="https://github.com/veerupandey/loomable/actions/workflows/ci.yml"><img src="https://github.com/veerupandey/loomable/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI" /></a>
    <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+" />
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT" />
    <img src="https://img.shields.io/badge/status-beta-blue.svg" alt="Status: beta" />
  </p>
</div>

<p align="center">
  <a href="#get-started">Get Started</a> ·
  <a href="#core-primitives">Primitives</a> ·
  <a href="#ag-ui-sse">AG-UI SSE</a> ·
  <a href="#examples">Examples</a> ·
  <a href="docs/API.md">API Reference</a> ·
  <a href="docs/STABILITY.md">Stability</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="SECURITY.md">Security</a> ·
  <a href="docs/BETA_PLAN.md">Beta plan</a>
</p>

## Introduction

**Public beta (`0.2.0b0`)** — durable primitives (Agent · Team · Workflow · Case · AG-UI), expect polish gaps. See [docs/STABILITY.md](docs/STABILITY.md) for the supported surface and beta limits (local workspace FS, cooperative cancel, shared API-key serve auth).

Loomable is a Python framework for production agent systems. One `Runnable` contract (`arun` → `RunResult`), progressive disclosure:

| Primitive | Role |
|-----------|------|
| **Agent** | Model + tools + memory + structured I/O |
| **Team** | Specialists (broadcast / sequential / coordinate) |
| **Workflow** | Durable multi-step process (HITL, checkpoints, SharedState) |
| **Case** | Goal + WorkItems board + plan → dispatch → accept |
| **Deep Agent** | Loomable-only long-horizon research harness (beats LangGraph/Agno/Crew deep stacks) |
| **Flow** | Low-level graph escape hatch |

Everything that runs is a `Runnable`. Agents nest in workflows; cases compile to workflows; workflows stream the same AG-UI events as agents.

```python
from loomable import Agent, Case, Workflow, tool
from loomable.serve import mount_agent

@tool
def search(query: str) -> str:
    """Search the knowledge base."""
    return "Python was created in 1991."

agent = Agent(model=provider, tools=[search], goal="Answer precisely")
result = await agent.arun("When was Python created?")

case = Case(
    model=provider,
    goal="Close INC-88421",
    board=True,
    dispatch="spawn",
    accept=lambda out, ctx: "SEV-" in out.text(),
)
async for event in case.astream_events(email):
    ...  # RUN_* / NODE_* / STATE_* / TEXT_*
```

## Get Started

### Installation

```bash
# Beta tag install (PyPI publish may lag the git tag)
pip install "loomable @ git+https://github.com/veerupandey/loomable.git@v0.2.0b0"
# or track main / editable
pip install "loomable @ git+https://github.com/veerupandey/loomable.git"
uv add "loomable @ git+https://github.com/veerupandey/loomable.git"
git clone https://github.com/veerupandey/loomable.git && cd loomable && pip install -e ".[dev,toolkits]"
```

### Provider credentials

Copy [`.env.example`](.env.example) to `.env` in the repo root (gitignored — never commit it), or export:

```bash
export GEMINI_API_KEY="..."          # Gemini
# or
export OPENAI_API_KEY="..."          # OpenAI
# or Azure: AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_DEPLOYMENT_NAME
```

### Your first agent

```python
import asyncio
from loomable import Agent
from loomable.providers.gemini import GeminiProvider

agent = Agent(model=GeminiProvider(), instructions="Be concise.")
print(asyncio.run(agent.arun("Capital of France?")).output.text())
```

Searchable knowledge base (vector store → `search_*` tools):

```python
agent = Agent(model=GeminiProvider(), knowledge_base=["./handbook.pdf", "./runbooks"])
```

## Core primitives

### Agent

```python
from loomable import Agent, tool

agent = Agent(
    model=provider,
    tools=[multiply],
    knowledge_base=["./handbook.pdf"],  # vector DB → search_knowledge
    response_model=Packet,
)
result = await agent.arun("What is 7 * 8?")
```

### Workflow (+ SharedState)

```python
from loomable import Workflow, JsonFileCheckpointer

cp = JsonFileCheckpointer("./ckpts")  # or PostgresCheckpointer(POSTGRES_URL)
wf = (
    Workflow("sev", session_id="inc-1", checkpointer=cp)
    .step("gather", gatherer)
    .parallel(analyst=analyst, visual=visual)
    .step("scribe", scribe, confirm=True)  # HITL
)
result = await wf.arun(email)
print(wf.state.get("gather"))
```

### Case (goal + board)

```python
from loomable import Case

case = Case(
    model=provider,
    goal="Close INC-88421 with SEV packet",
    board=True,                 # WorkItems: open → in_progress → blocked → done
    dispatch="reuse",           # or "spawn"
    accept=packet_ok,
    max_rounds=3,
)
result = await case.arun(email)
print(result.metadata["board"])
```

### Team

```python
from loomable import Team

team = Team(members=[sre, legal, exec], mode="broadcast", hard=True)
result = await team.arun(brief)
```

## AG-UI SSE

First-class CopilotKit / AG-UI-compatible events — no hard CopilotKit dependency.

```python
from fastapi import FastAPI
from loomable.serve import mount_agent, mount_case

app = FastAPI()
mount_agent(app, agent, prefix="/agent")  # POST /agent/run/events
mount_case(app, case, prefix="/cases")    # POST /cases/run/events

# In-process:
async for ev in agent.astream_events(prompt):
    print(ev.type)  # RUN_STARTED, TEXT_MESSAGE_CONTENT, TOOL_CALL_*, RUN_FINISHED
```

Legacy NDJSON remains at `POST /run/stream`.

## Features

| Feature | What it does |
|---------|--------------|
| **Function tools** | `@tool` — JSON schema from type hints |
| **Tool-use loop** | Automatic tool iteration until final answer |
| **Require tools** | Path-constrained side-effect enforcement |
| **Memory** | Session / user / tiered stores + compaction |
| **Knowledge base** | Vector store → `search_*` tools (`knowledge_base=`); optional passive `knowledge=` |
| **Multimodal I/O** | Image / audio / video in and out |
| **Structured I/O** | Pydantic / dataclass schemas |
| **Verification** | Same verifier protocol on Agent, Loop, Case |
| **HITL** | Fluent `confirm=True` + `approve()` + resume |
| **Checkpoints** | JsonFile / SQLite / in-memory durability |
| **Team modes** | broadcast / sequential / coordinate (+ hard) |
| **Case board** | WorkItems + `STATE_SNAPSHOT` / `STATE_DELTA` |
| **AG-UI SSE** | Lifecycle, text, tools, nodes, state |
| **MCP / Skills** | External tool packages |
| **Observability** | Structured events and run metadata |

## Examples

```
examples/
├── agents/               # Start here (07 = knowledge_base vector DB)
├── deep_agent/           # Loomable-only deep research (beats peer deep agents)
├── subagents/            # Delegation & Team
├── patterns/             # Loop / pipeline / parallel / plan-execute
├── memory/               # Session & shared memory
├── advanced/             # MCP, checkpoints, multimodal
├── simple_use_cases/     # News, research, docs
└── escalation_war_room/  # Full SEV ladder (Case + SSE)
```

```bash
python examples/agents/01_hello_world.py
python examples/agents/07_knowledge_base.py
python examples/agents/08_team_knowledge_base.py
python examples/deep_agent/04_live_multimodal_research.py
python examples/advanced/05_build_retriever.py
python examples/escalation_war_room/10_case.py
python examples/escalation_war_room/12_agent_agui_sse.py
```

See [`examples/README.md`](examples/README.md) for the full map.

## Architecture

```
loomable/
├── agent/       # Agent, Team, tools, memory, events
├── case.py      # Case + Board
├── stream/      # AG-UI StreamEvent + AsyncStreamBus
├── flow/        # Workflow, Flow, Loop, engines, SharedState
├── content/     # Media parts & coercion
├── kernel/      # Core primitives
├── providers/   # Models + backends (Postgres KV/vector)
├── retrieval/   # ingest, KnowledgeBase, agentic retrievers
├── persist/     # JsonFile / SQLite / Postgres checkpointers
├── serve/       # FastAPI + MCP adapters
└── media/       # Image / Audio / Video helpers
```

## Providers

| Provider | Class |
|----------|-------|
| OpenAI | `OpenAIProvider` |
| Azure OpenAI | `AzureOpenAIProvider` |
| Anthropic | `AnthropicProvider` |
| Google Gemini | `GeminiProvider` |
| Groq | `GroqProvider` |
| Ollama | `OllamaProvider` |

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/unit -q

# Postgres live E2E (Agent memory + checkpointers)
pip install -e ".[postgres]"
docker compose up -d
POSTGRES_URL=postgresql://loomable:loomable@127.0.0.1:5432/loomable \
  python -m pytest tests/integration/test_postgres_live.py -q
```

## License

[MIT](LICENSE)

<p align="right"><a href="#top">↑ Back to top</a></p>
