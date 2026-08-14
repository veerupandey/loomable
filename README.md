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
| **Team** | Specialists (broadcast / sequential / coordinate / route) |
| **Workflow** | Durable multi-step process (HITL, checkpoints, SharedState) |
| **Case** | Goal + WorkItems board + plan → dispatch → accept |
| **Memory** | `Memory.compose` — conversation / user / knowledge layers |
| **Deep Agent** | `create_deep_agent(profile="research"\|"code")` long-horizon harness |
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

Composable memory:

```python
from loomable import Agent, ConversationMemory, Memory, UserMemory, open_session_store, open_vector_store
from loomable.agent import NoteStore

# UserMemory(memory_tool=/auto_extract=) needs note_store= or embedder=
from loomable.providers import OpenAIEmbedder  # or GeminiEmbedder / AzureOpenAIEmbedder

embedder = OpenAIEmbedder()
notes = NoteStore(long_term=open_vector_store(engine="memory"), embedder=embedder)
memory = Memory.compose(
    conversation=ConversationMemory(store=open_session_store("sqlite", path="sessions.db")),
    user=UserMemory(note_store=notes, auto_extract=True, memory_tool=True),
)
agent = Agent(model=provider, memory=memory, session_id="chat-1", user_id="alice")
```

Deep research / code harness:

```python
from loomable import create_deep_agent

agent = create_deep_agent(provider, profile="research", workspace="./.deep_workspace")
# agent = create_deep_agent(provider, profile="code", repo="./my-app")
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
from loomable import Workflow, JsonFileCheckpointer, FlowPaused

cp = JsonFileCheckpointer("./ckpts")  # or PostgresCheckpointer(POSTGRES_URL)
wf = (
    Workflow("sev", session_id="inc-1", checkpointer=cp)
    .step("gather", gatherer)
    .parallel(analyst=analyst, visual=visual)
    .step("scribe", scribe, confirm=True)  # HITL
)
try:
    result = await wf.arun(email)
except FlowPaused:
    await wf.approve("scribe")
    result = await wf.arun(resume=True)
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

# hard= applies only to broadcast / sequential (default on for those modes)
team = Team(members=[sre, legal, exec], mode="broadcast", hard=True)
result = await team.arun(brief)
# soft LLM orchestration: Team(..., mode="coordinate") or mode="route"
```

## AG-UI SSE

First-class AG-UI events over `text/event-stream`.

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

Prefer AG-UI SSE (`/run/events`). NDJSON at `POST /run/stream` is **Agent-only**
(requires `astream`); `mount_case` does not register it. Disconnect calls
`cancel()` on Agent, Case, Workflow, and Team.

## Features

| Feature | What it does |
|---------|--------------|
| **Function tools** | `@tool` — JSON schema from type hints |
| **Tool-use loop** | Automatic tool iteration until final answer |
| **Require tools** | Path-constrained side-effects; `strict_require_tools=True` fail-closed; Workflow inherit |
| **Memory** | `Memory.compose` (conversation / user / knowledge) + compaction |
| **Knowledge base** | Vector store → `search_*` tools (`knowledge_base=`); optional passive `knowledge=` + `embedder=` |
| **Retrieval builders** | `loomable.retrieval` helpers are experimental; prefer `knowledge_base=` / `retrievers=` on Agent |
| **Multimodal I/O** | Image / audio / video in and out |
| **Structured I/O** | Pydantic / dataclass schemas |
| **Verification** | Same verifier protocol on Agent, Loop, Case |
| **HITL** | Fluent `confirm=True` + `approve()` + resume |
| **Checkpoints** | JsonFile / SQLite / in-memory durability |
| **Team modes** | `broadcast`/`sequential` (hard by default); `coordinate`/`route` (soft) |
| **Case board** | WorkItems + `STATE_SNAPSHOT` / `STATE_DELTA` |
| **AG-UI SSE** | Lifecycle, text, tools, nodes, state |
| **MCP / Skills** | External tool packages |
| **Observability** | Structured events and run metadata |

## Examples

```
examples/
├── agents/               # Start here (07 = knowledge_base vector DB)
├── deep_agent/           # create_deep_agent(profile=research|code)
├── subagents/            # Delegation & Team
├── patterns/             # Workflow step / parallel / Team route / map
├── memory/               # Memory.compose, Workflow chaining, callable blackboard
├── advanced/             # MCP, Workflow branch, checkpoints, RAG
├── simple_use_cases/     # News, research, docs
└── escalation_war_room/  # Full SEV ladder (Case + SSE)
```

```bash
python examples/agents/01_hello_world.py
python examples/agents/07_knowledge_base.py   # live LLM + knowledge_base=
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
