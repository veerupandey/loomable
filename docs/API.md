# Loomable — High-Level API Reference

A production-grade agent framework. Build agents in a few lines, scale to Teams,
Workflows, and Cases with SharedState, HITL, checkpoints, and AG-UI SSE.

---

## Table of Contents

- [Progressive Disclosure (Levels 0–7)](#progressive-disclosure-levels-07)
- [Quick Start](#quick-start)
- [Agent](#agent)
- [Case](#case)
- [Subagents & Teams](#subagents--teams)
- [Tools](#tools)
- [Model Providers](#model-providers)
- [Memory](#memory)
- [Structured Output](#structured-output)
- [Multimodal I/O](#multimodal-io)
- [Streaming](#streaming)
- [AG-UI SSE](#ag-ui-sse)
- [Flow Engine](#flow-engine)
- [Knowledge / RAG](#knowledge--rag)
- [Production Hardening](#production-hardening)
- [MCP Integration](#mcp-integration)
- [Serving](#serving)
- [Checkpointing](#checkpointing)
- [Display & Visualization](#display--visualization)

---

## Progressive Disclosure (Levels 0–7)

The API is designed so you start simple and add capabilities incrementally — never rewrite into a new DSL. Each level adds configuration to the level below.

### Level 0: Agent in 3 lines

The simplest possible agent. No tools, no loops, no config.

```python
from loomable.agent import Agent

agent = Agent(model="openai:gpt-4o-mini")
result = agent.run("What is the capital of France?")
print(result.output.text())
```

No optimizer, no engine, no checkpointer, no verifier, no deps, no memory. Just works.

### Level 1: Agent + Tools (auto-escalates)

Add tools and the agent automatically uses its tool loop. No strategy selection needed.

```python
from loomable.agent import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

agent = Agent(model="openai:gpt-4o-mini", tools=[search])
result = agent.run("Find the latest AI news")
```

The complexity router auto-escalates: simple inputs get single-shot responses, complex inputs trigger the tool loop or self-plan — all transparent.

### Level 2: Agent + Verifier (output guardrail)

Add a Verifier to gate outputs against a machine-readable success condition.

```python
from loomable.agent import Agent
from loomable.flow import Verifier, VerdictResult

def check_citation(output, context) -> bool:
    """Ensure the output contains a citation."""
    return "[source]" in output.text()

agent = Agent(
    model="openai:gpt-4o-mini",
    verifier=check_citation,
    retry_on_failure=True,
    max_verify_retries=2,
)
result = agent.run("Explain quantum computing with citations")
# On failure, retries with feedback; result.verification shows outcome
```

A plain callable `(output, context) -> bool` works. No import ceremony required for the simple case.

### Level 3: Loop (repeat until verified)

Wrap any Runnable in a Loop for iterative refinement with explicit termination.

```python
from loomable.flow import Loop, Verifier, VerdictResult

def quality_check(output, context):
    if "excellent" in str(output).lower():
        return VerdictResult(ok=True)
    return VerdictResult(ok=False, detail="Needs more polish")

loop = Loop(agent, verifier=quality_check, max_iterations=3)
result = await loop.arun("Write an excellent summary of AI trends")
# Repeats up to 3 times, feeding failure detail forward for self-correction
```

The Loop is itself a Runnable — usable standalone or as a node in a Flow.

### Level 4: Workflow (preferred high-level process API)

Build multi-step processes without Edges, frozensets, or engine enums:

```python
from loomable import Agent, Workflow, Step, JsonFileCheckpointer

researcher = Agent(model="openai:gpt-4o-mini", instructions="Research the topic.")
writer = Agent(model="openai:gpt-4o-mini", instructions="Write a short brief.")
editor = Agent(model="openai:gpt-4o-mini", instructions="Polish the brief.")

wf = (
    Workflow("article", session_id="job-1", checkpointer=JsonFileCheckpointer("./ckpts"))
    .step("research", researcher)
    .step("draft", writer)
    .step("edit", editor)
)
result = await wf.arun("AI agents in 2025")
print(wf.explain())  # inspect graph before/after run
```

Fluent builders for complex cases:

```python
wf = (
    Workflow("sev1", session_id="inc-1", memory=True)
    .step("gather", gatherer)
    .parallel(analyst=analyst, visual=visual)          # concurrent
    .branch(when=needs_human, then=approver, else_=auto)  # conditional
    .loop(polisher, until=quality_ok, max_iterations=3)   # verify/retry
    .step("publish", publisher)
)
```

Declarative style still works: `Workflow("pipe", steps=[Step("a", a), Step("b", b)])`.

Low-level `Flow` / `sequential()` / `Edge` remain available as an advanced escape hatch.

### Level 4b: Flow list shorthand (advanced alias)

Compose multiple agents/functions into a sequential workflow. The simplest Flow — just a list.

```python
from loomable.flow import Flow

def research(input):
    return f"Research findings about: {input}"

def write(input):
    return f"Article based on: {input}"

def edit(input):
    return f"Polished: {input}"

flow = Flow([research, write, edit])
result = await flow.arun("AI agents in 2025")
```

Or use the `sequential()` helper (replaces the old `Pipeline`):

```python
from loomable.flow import sequential

flow = sequential(research, write, edit)
result = await flow.arun("AI agents in 2025")
```

Zero-config: no engine, optimizer, memory, or checkpointer needed.

### Level 5: Flow with parallel engine

Run independent branches concurrently. The engine handles supersteps automatically.

```python
from loomable.flow import Flow, Edge

flow = Flow(
    {"research": researcher, "analyze": analyst, "synthesize": writer},
    edges=[
        Edge(source="research", target="synthesize"),
        Edge(source="analyze", target="synthesize"),
    ],
    engine="auto",  # auto-selects ParallelEngine (research & analyze are independent)
)
result = await flow.arun("Compare AI frameworks")
```

Or use the `parallel()` helper for the fully-concurrent broadcast pattern:

```python
from loomable.flow import parallel

flow = parallel(researcher, analyst, writer)
result = await flow.arun("Analyze the AI market")
```

Engine selection is automatic by default: linear chains get Sequential, independent branches get Parallel, a manager node gets Hierarchical.

### Level 6: Flow with optimizer, memory, checkpointer

Add optimization, shared memory, and durable checkpointing for production workflows.

```python
from loomable.flow import Flow, Optimizer, TieredMemoryStore, MemoryStore
from loomable.persist import JsonFileCheckpointer

flow = Flow(
    {"research": researcher, "draft": writer, "review": reviewer},
    edges=[
        Edge(source="research", target="draft"),
        Edge(source="draft", target="review"),
    ],
    optimizer=True,  # enables parallelization, dead-node elimination, CSE, model-tier rules
    memory=TieredMemoryStore(),
    checkpointer=JsonFileCheckpointer(".checkpoints"),
    session_id="article-v1",
)
result = await flow.arun("Write a technical article")

# Inspect the optimization
plan = flow.explain()
print(plan)  # shows original vs optimized topology + applied rules
```

Everything is opt-in. An unoptimized, memory-free, uncheckpointed flow runs identically to one without these options.

### Level 7: Custom engine, HITL, observability

Full control: custom execution engines, human-in-the-loop gates, and context-snapshot observability.

```python
from loomable.flow import (
    Flow, Node, Edge, FlowPaused,
    ExecutionEngine, ContextSnapshotConfig,
)
from loomable.persist import JsonFileCheckpointer

# Custom engine (satisfies the ExecutionEngine protocol)
class MyStreamingEngine:
    async def run(self, flow, input, state, context):
        # Custom execution logic — dependency-driven, streaming, etc.
        ...

# Human-in-the-loop: mark a node as requiring confirmation
flow = Flow(
    {
        "draft": Node(node_id="draft", runnable=writer),
        "publish": Node(node_id="publish", runnable=publisher, require_confirmation=True),
    },
    edges=[Edge(source="draft", target="publish")],
    engine=MyStreamingEngine(),
    checkpointer=JsonFileCheckpointer(".checkpoints"),
    session_id="pub-flow",
    events=my_event_emitter,  # receives node_start/node_end + context_snapshot events
)

try:
    result = await flow.arun("Publish the quarterly report")
except FlowPaused as paused:
    # Flow paused before 'publish' — checkpoint saved, process can exit
    # Later: resume with approval decision
    ...
```

Context-snapshot observability (opt-in, zero overhead when disabled):

```python
from loomable.flow import ContextSnapshotConfig

# Enable snapshots to see exactly what context each node received
config = ContextSnapshotConfig(enabled=True, metadata_only=False)
# Attach via events emitter — diagnose "green trace but wrong output" failures
```

---

## Quick Start

```python
from loomable.agent import Agent

agent = Agent(model="openai:gpt-4o-mini")
result = agent.run("What is the capital of France?")
print(result.output.text())  # "The capital of France is Paris."
```

---

## Agent

The `Agent` class is the single entry point. It composes all framework features through keyword arguments — progressive disclosure from simple to advanced.

```python
from loomable.agent import Agent

agent = Agent(
    model="openai:gpt-4o-mini",       # model string or ModelSpec or provider instance
    name="researcher",                  # optional name (used in tracing, orchestration)
    role="Senior Researcher",           # who the agent is (used in system prompt + delegation)
    goal="Find accurate information",   # what it optimizes for
    instructions="Be concise.",         # additional system prompt instructions
    tools=[...],                        # list of @tool-decorated functions
    subagents=[...],                    # list of Agent instances for delegation
    session_id="conv-1",                # enables multi-turn memory
    user_id="alice",                    # scopes UserMemory notes when using Memory.compose
    memory=None,                        # Memory.compose(conversation=..., user=...)
    debug=True,                         # prints trace to stderr
)
```

### Persona: role + goal + instructions

The `role`, `goal`, and `instructions` fields are assembled into a single system prompt:

```python
agent = Agent(
    model="openai:gpt-4o-mini",
    role="Senior Security Reviewer",
    goal="Identify vulnerabilities and suggest fixes",
    instructions="Focus on OWASP Top 10.",
)
# System prompt becomes:
# "You are a Senior Security Reviewer.
#  Your goal: Identify vulnerabilities and suggest fixes
#
#  Focus on OWASP Top 10."
```

The `role` is reused when agents collaborate — a manager synthesizing worker outputs sees "Security Reviewer said X" instead of "agent_0 said X".

### Subagents: Dynamic Delegation

Pass other Agent instances as `subagents` and the parent agent can delegate to them at runtime:

```python
researcher = Agent(model="openai:gpt-4o-mini", role="Researcher", goal="Find facts")
writer = Agent(model="openai:gpt-4o-mini", role="Writer", goal="Write clearly")

lead = Agent(
    model="openai:gpt-4o-mini",
    role="Tech Lead",
    instructions="Delegate research, then writing. Synthesize the results.",
    subagents=[researcher, writer],
)
result = await lead.arun("Write docs for our auth system")
```

Each subagent becomes a tool (`delegate_to_researcher`, `delegate_to_writer`) that the parent's LLM can call. The parent decides at runtime who to call, when, and in what order. Subagent failures are isolated — one failing doesn't crash the parent. Subagents can themselves have subagents (nested delegation).

### Team: Explicit Orchestration Modes

For users who want named orchestration patterns without writing custom instructions:

```python
from loomable.agent import Agent, Team

team = Team(
    members=[researcher, writer, critic],
    model="openai:gpt-4o-mini",
    mode="coordinate",  # coordinate | route | broadcast | sequential
)
result = await team.arun("Review our API design")
```

| Mode | Behavior |
|------|----------|
| `coordinate` | Delegate to ALL members, synthesize results |
| `route` | Pick the single best member for the task |
| `broadcast` | Send same input to all, merge labeled results |
| `sequential` | Chain members in order, each builds on previous |

Under the hood, `Team` creates a parent Agent with auto-generated instructions and `subagents=members`.

### Running

```python
# Async (preferred)
result = await agent.arun("hello")

# Sync wrapper
result = agent.run("hello")

# Streaming (real token-level deltas)
async for chunk in agent.astream("hello"):
    print(chunk.delta.data.decode(), end="")
```

### RunResult

Every run returns a `RunResult`:

```python
result.output.text()       # the agent's text response
result.session_id          # session identifier
result.usage               # {"input_tokens": N, "output_tokens": M}
result.structured          # parsed output when response_model is set
result.tool_activity       # tool outcomes from this run
result.metadata            # {"stop_reason": "final", ...}
result.trace               # list of Event objects (when debug=True)
```

---

## Deep Agent

Loomable-native long-horizon harness. **One API:** `create_deep_agent`.
Specialize with skills — research any topic via the bundled `research` skill:

```python
from loomable import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-4o-mini",
    profile="research",  # = skills=["research"] + report/citation gates
    workspace="./.deep_workspace",
    # discovery_core="research" (default, correctness-first)
    # discovery_core="research-slim"  # smaller schema budget (≥50% target)
)
await agent.arun("Research any topic; write reports/brief.md")
```

`create_research_agent(...)` is a **deprecated** alias for `profile="research"`
(emits `DeprecationWarning`; prefer `create_deep_agent`).

Pillars:

1. **Planning** — `TodoTools`
2. **Workspace FS** — sliced reads, `delete_file`, token-aware offload (**local only** in beta)
3. **Subagents** — `task` / `task_batch` + named `specialists=` (inherit `discovery=True`)
4. **Skills** — `skills=["research"]` or any catalog / skill dir (SkillLoader accepts both)
5. **Discovery** — deep agents enable `discovery=True` with a **schema budget**:
   core tools stay advertised; the rest (images, PDF, code, MCP, …) use
   `search_tools` / `search_mcp` / `activate_tool`. Skills are **metadata-first**
   (`load_skill`). Profiles: `discovery_core="research"` (default) or
   `"research-slim"` (experimental slim allowlist); see `docs/COMPETITIVE.md`
   and `docs/STABILITY.md`.
6. **Gates** — research profile requires `reports/` + `register_source` + accept verifier

### Cancel

```python
built = agent.build()
# During an in-flight arun / astream_events:
built.cancel()   # or agent.cancel() — cooperative at tool-loop boundaries
```

SSE / NDJSON client disconnect on `mount_*` also calls cancel.

See `examples/deep_agent/` and `loomable/skills/research/SKILL.md`.

---

## Case

Long-running goal work with an optional WorkItems board. Compiles to a `Workflow`
(plan → dispatch → synthesize → accept) and shares `SharedState` with Flow engines.

```python
from loomable import Case

case = Case(
    model=provider,
    goal="Close INC-88421 with SEV packet",
    board=True,              # open → in_progress → blocked → done
    dispatch="spawn",        # or "reuse" (one worker mapped over steps)
    accept=lambda out, ctx: "SEV-" in out.text(),
    max_rounds=3,
)
result = await case.arun(email)
print(result.metadata["board"])

# Same pipeline as Agent(mode="case", ...)
agent = Agent(model=provider, mode="case", dispatch="reuse", accept=check)
```

`Case.as_workflow()` returns a nestable `Workflow`. Board mutations stream as
`STATE_SNAPSHOT` / `STATE_DELTA` via `astream_events`.

---

## Subagents & Teams

Multi-agent collaboration where an agent delegates to specialist subagents at runtime.

### When to use what

| I need... | Use... |
|-----------|--------|
| One agent with full flexibility on who to delegate to | `Agent(subagents=[...])` |
| Named orchestration mode without custom instructions | `Team(mode="coordinate")` |
| Explicit step-by-step pipeline | `sequential(a, b, c)` |
| Workers + manager synthesis | `coordinate(workers=[...], manager=mgr)` |

### Subagents (recommended for most cases)

```python
from loomable.agent import Agent

researcher = Agent(model="openai:gpt-4o-mini", role="Researcher", goal="Find facts")
writer = Agent(model="openai:gpt-4o-mini", role="Writer", goal="Write clearly")
critic = Agent(model="openai:gpt-4o-mini", role="Editor", goal="Improve quality")

lead = Agent(
    model="openai:gpt-4o-mini",
    role="Tech Lead",
    instructions="Delegate research, writing, then editing. Synthesize the final output.",
    subagents=[researcher, writer, critic],
)
result = await lead.arun("Write docs for our auth system")
```

The parent's LLM sees each subagent as a callable tool (`delegate_to_researcher`, `delegate_to_writer`, `delegate_to_editor`) and decides dynamically who to call, when, and how many times.

### Team (explicit modes)

```python
from loomable.agent import Agent, Team

team = Team(
    members=[researcher, writer, critic],
    model="openai:gpt-4o-mini",
    mode="coordinate",
    instructions="Focus on security aspects.",  # optional extra instructions
)
result = await team.arun("Review our checkout system")
```

Modes: `coordinate` (all + synthesize), `route` (pick one), `broadcast` (all same input), `sequential` (chain in order).

### Nested Delegation

Subagents can themselves have subagents:

```python
junior = Agent(model="openai:gpt-4o-mini", role="Junior Dev")
senior = Agent(model="openai:gpt-4o-mini", role="Senior Dev", subagents=[junior])
lead = Agent(model="openai:gpt-4o-mini", role="Tech Lead", subagents=[senior])
```

---

## Tools

Decorate any Python function to make it a tool:

```python
from loomable.agent import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

@tool(idempotent=False)
def send_email(to: str, body: str) -> str:
    """Send an email (side-effecting)."""
    return f"Sent to {to}"

agent = Agent(model="openai:gpt-4o-mini", tools=[search, calculator, send_email])
```

- Schema auto-derived from type hints and docstrings
- Sync and async functions supported
- `idempotent=False` prevents automatic re-dispatch

---

## Model Providers

### String Shorthand (recommended)

```python
Agent(model="openai:gpt-4o-mini")
Agent(model="anthropic:claude-sonnet-4-20250514")
Agent(model="groq:llama-3.3-70b-versatile")
Agent(model="ollama:mistral")
Agent(model="gemini:gemini-2.0-flash")
Agent(model="azure:gpt-4.1-mini")
Agent(model="gpt-4o-mini")  # bare name defaults to OpenAI
```

API keys read from environment variables automatically:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`
- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_DEPLOYMENT_NAME`

### OpenAI-Compatible Endpoints

```python
from loomable.providers import OpenAIProvider

# vLLM, Together, LM Studio, or any OpenAI-compatible server
provider = OpenAIProvider(model="llama-3", base_url="http://localhost:8000/v1")
agent = Agent(model=provider)
```

### Anthropic-Compatible Endpoints

```python
from loomable.providers import AnthropicProvider

provider = AnthropicProvider(model="claude-3-haiku", base_url="https://my-gateway.example.com")
agent = Agent(model=provider)
```

### All Providers

| Class | Use Case |
|-------|----------|
| `OpenAIProvider` | OpenAI + any compatible endpoint |
| `AzureOpenAIProvider` | Azure OpenAI deployments |
| `AnthropicProvider` | Anthropic Messages API |
| `GroqProvider` | Groq inference |
| `OllamaProvider` | Local Ollama models |
| `GeminiProvider` | Google Gemini |

All support `complete()` and `stream()`.

---

## Memory

Compose memory layers and pass **one object** to the Agent:

```python
from loomable import Agent, Memory, ConversationMemory, UserMemory, open_session_store
from loomable.agent import NoteStore
from loomable.kernel.long_term import LongTermStore

memory = Memory.compose(
    # short-term / thread (L1+L2)
    conversation=ConversationMemory(
        store=open_session_store("postgres", url=DSN, user_id="alice"),
        window=8,
        compaction_threshold=16,
    ),
    # long-term user facts (L3) — scoped by Agent(user_id=...)
    user=UserMemory(
        note_store=NoteStore(LongTermStore(), embedder),
        memory_tool=True,    # agentic tool
        auto_extract=True,   # Always-mode lite (heuristic facts from user text)
    ),
    # optional RAG
    # knowledge=KnowledgeMemory(documents=[...], embedder=embedder),
)

agent = Agent(
    model="openai:gpt-4o-mini",
    memory=memory,
    session_id="conv-1",
    user_id="alice",
    scopes={"claim_id": "CLM-4421"},  # any extra keys: policy_id, case_id, …
)
```

| Layer | Class | What it stores |
|-------|--------|----------------|
| Conversation | `ConversationMemory` (`short=`) | L1 turns + L2 summaries for `session_id` |
| User | `UserMemory` (`long=`) | Cross-session facts via `NoteStore` (scoped) |
| Knowledge | `KnowledgeMemory` | RAG docs into the prompt |
| Working | `WorkingMemory` | Flow blackboard (`TieredMemoryStore`) — not Agent chat |

### Scopes (user_id, claim_id, …)

Long-term notes are isolated by a :class:`~loomable.memory.MemoryScope` — any
key/value map, not only `user_id`:

```python
from loomable import MemoryScope

# Same shape:
scopes={"user_id": "alice", "claim_id": "CLM-4421", "lob": "auto"}
# or
MemoryScope.of(user_id="alice", claim_id="CLM-4421")
```

- Notes are prefixed `claim_id=CLM-4421|user_id=alice:…` and tagged `scope:key=value`.
- Recall requires **all** scope tags — claim A never sees claim B for the same user.
- For **conversation** isolation per claim, put it in `session_id`  
  (e.g. `session_id=f"claim:{claim_id}"`) and/or Postgres `user_id=` tenant  
  (`scope.tenant_key()`).

Legacy kwargs (`session_store=`, `note_store=`, `memory_backend=`) still work and
**override** the matching compose layer when both are set.

`user_id` + `scopes` are applied automatically when using `Memory.compose` or when
you pass a bare `note_store=` with `user_id`/`scopes`.

### How to think about it (perspectives)

1. **Same chat, same process** — only `session_id` (default in-process SQLite).
2. **Durable chat** — `ConversationMemory(store=open_session_store(...))` + `resume=True` on reload.
3. **User remembers across chats** — `UserMemory` + `user_id` (+ optional `auto_extract`).
4. **Postgres chat + zvec notes** — compose both layers; backends stay independent.
5. **Case / Workflow resume** — `checkpointer` is separate from Agent memory.

`resume=True` means “this session row must already exist.” First turn: omit it. Later Agents: pass `resume=True`.

### Same kwargs on Team / Case / Agent-in-Flow

| Surface | Conversation L1/L2 | Long-term L3 | Notes |
|---------|--------------------|--------------|-------|
| **Agent** | `memory=` compose or `session_store` | `UserMemory` / `note_store` | Canonical API |
| **Team** | Same kwargs → **coordinator** | Same → coordinator | Members keep their own memory |
| **Case** | Same → role-scoped sessions | Shared notes | `from_agent` copies memory |
| **Flow step** | Agent’s own memory | Agent’s own notes | `Flow(memory=True)` is TieredMemory blackboard |
| **mount_agent** | `bind_session` reloads L1/L2 | unchanged | |

### Minimal chat (same process)

```python
agent = Agent(model="openai:gpt-4o-mini", session_id="conv-1")
await agent.arun("My name is Alice")
await agent.arun("What's my name?")  # Alice
```

### Conversation store (L1/L2) — including Postgres

```python
from loomable.memory import open_session_store

store = open_session_store("sqlite", path="sessions.db")
# store = open_session_store("file", path="./.sessions")
# store = open_session_store("postgres", url=DSN, user_id="alice")
# store = open_session_store("memory")

agent = Agent(model=..., session_id="conv-1", session_store=store)
await agent.arun("I prefer dark mode")

agent2 = Agent(model=..., session_id="conv-1", session_store=store, resume=True)
```

Or pass a KV backend (whole session saved under `session:{id}`):

```python
agent = Agent(
    model=...,
    session_id="conv-1",
    memory_backend=PostgresMemoryBackend(DSN, user_id="alice"),
)
```

### Long-term (L3): zvec or Postgres vectors

```python
from loomable.agent import NoteStore
from loomable.kernel.long_term import LongTermStore
from loomable.providers import OpenAIEmbedder

notes = NoteStore(long_term=LongTermStore(), embedder=OpenAIEmbedder())

# Durable L3 (Postgres)
from loomable.providers.backends.postgres import PgVectorBackend
notes = NoteStore(
    long_term=LongTermStore(
        backend=PgVectorBackend(DSN, dimensions=1536, user_id="alice"),
        backend_name="postgres",
    ),
    embedder=OpenAIEmbedder(),
)
```

### Compose matrix

| Want | Conversation | User / L3 |
|------|--------------|-----------|
| Local demo | default / file / memory | zvec `LongTermStore()` |
| Prod chat, ephemeral notes | postgres | zvec |
| Prod chat + durable notes | postgres | `PgVectorBackend` |
| Case/Workflow resume | — | `PostgresCheckpointer` (separate) |

### Knobs

| Param | Effect |
|-------|--------|
| `memory=Memory.compose(...)` | Unified layers |
| `session_id` | Enables L1/L2 persist |
| `user_id` | Scopes `UserMemory` notes |
| `resume=True` | Reload L1/L2 (session must exist) |
| `UserMemory(auto_extract=True)` | Heuristic facts after each turn |
| `UserMemory(memory_tool=True)` | Agentic memory tool |
| `memory_window` / `compaction_threshold` | Replay + L2 spill |

### Postgres install

```bash
pip install 'loomable[postgres]'
docker compose up -d
# POSTGRES_URL=postgresql://loomable:loomable@127.0.0.1:5432/loomable
```

`PostgresCheckpointer` is for Case/Workflow resume — not Agent chat history.  
`memory_tool=True` without a note store is a no-op.  
`knowledge` RAG builds its own zvec store today — separate from `note_store`.

---

## Structured Output

```python
from pydantic import BaseModel

class CityInfo(BaseModel):
    name: str
    population: int
    country: str

# Set once on agent (applies to all runs)
agent = Agent(model="openai:gpt-4o-mini", response_model=CityInfo)
result = agent.run("Info about Tokyo")
city = result.structured  # CityInfo(name="Tokyo", population=13960000, country="Japan")

# Or per-call
result = await agent.arun("Info about Paris", output_schema=CityInfo)
```

Supports Pydantic models, dataclasses, and any callable.

---

## Multimodal I/O

Agents can process images, video, and audio as input, and return media as output (model-dependent). Tools can also return typed media objects that are surfaced on `RunResult` convenience properties.

### Media Classes

The `loomable.media` module provides high-level media classes that unify URL, file path, raw bytes, and base64 sources behind a single interface:

```python
from loomable.media import Image, Audio, Video, File
```

These classes are also re-exported from `loomable.agent` for convenience:

```python
from loomable.agent import Image, Audio, Video, File
```

#### Constructors

Each class accepts exactly **one** source parameter (`url`, `filepath`, or `content`):

```python
# From a URL (no fetch until content is accessed)
img = Image(url="https://example.com/photo.png")

# From a file path (lazy — file is not read until needed)
img = Image(filepath="./chart.png")

# From raw bytes
img = Image(content=raw_bytes)

# From a base64-encoded string (auto-decoded to bytes)
img = Image(content="iVBORw0KGgo...")

# Audio and Video work the same way
clip = Audio(url="https://example.com/speech.wav")
vid = Video(filepath="./demo.mp4")
doc = File(filepath="./report.pdf", filename="Q4 Report.pdf")
```

Providing zero or more than one source raises `ValueError`.

**Optional parameters:**

| Parameter | Classes | Description |
|-----------|---------|-------------|
| `format` | All | File format (e.g. `"png"`, `"wav"`). Auto-inferred from extension if omitted. |
| `mime_type` | All | MIME type (e.g. `"image/png"`). Auto-inferred from format/extension if omitted. |
| `detail` | `Image` only | Detail level for OpenAI models: `"high"`, `"low"`, or `"auto"`. |
| `duration` | `Audio`, `Video` | Duration in seconds (informational). |
| `filename` | `File` only | Original filename hint. |

#### Convenience Methods

Every media instance provides these methods:

```python
img = Image(filepath="./chart.png")

# Save resolved content to disk (fetches URL or reads file if needed)
img.save("./output/chart_copy.png")

# Get base64-encoded string
b64 = img.to_base64()

# Get a complete data URI (e.g. "data:image/png;base64,...")
uri = img.to_data_uri()

# Convert to low-level MediaPart for kernel interop
part = img.to_media_part()
```

If content cannot be resolved (file not found, URL unreachable), a `MediaResolveError` is raised:

```python
from loomable.media import MediaResolveError

try:
    img = Image(filepath="./missing.png")
    img.save("./copy.png")  # raises MediaResolveError
except MediaResolveError as e:
    print(f"Cannot resolve media: {e}")
```

### Enabling multimodal

Agents accept image and video input by default. Pass media on `arun` — no flag needed:

```python
from loomable import Agent

agent = Agent(model="openai:gpt-4o-mini")
result = await agent.arun("Describe this chart", images=["./chart.png"])
```

Lock to text-only (no frozensets required):

```python
agent = Agent(model="openai:gpt-4o-mini", modalities="text")
# or
agent = Agent(model="openai:gpt-4o-mini", text_only=True)
```

Other examples: `modalities="text+image"`, `modalities=["text", "audio"]`,
`capabilities="text+audio"`. Audio remains opt-in.

`multimodal=True` is a deprecated no-op kept for back-compat.

### Input: passing images

```python
# File path (simplest)
result = await agent.arun("Describe this chart", images=["./chart.png"])

# URL (auto-detected from http/https prefix)
result = await agent.arun("What's in this?", images=["https://example.com/photo.jpg"])

# Multiple images
result = await agent.arun("Compare these", images=["before.png", "after.png"])

# Raw bytes
with open("photo.jpg", "rb") as f:
    result = await agent.arun("Analyze", images=[f.read()])

# Media class instances
from loomable.media import Image
result = await agent.arun("Analyze", images=[
    Image(filepath="./photo.jpg"),
    Image(url="https://example.com/img.png"),
    Image(content=raw_bytes),
])

# Explicit control via low-level helper (still works)
from loomable.agent import image
result = await agent.arun("Analyze", images=[
    image(path="./photo.jpg"),
    image(uri="https://example.com/img.png"),
    image(data=raw_bytes, media_type="image/webp"),
])
```

### Input: passing video

```python
result = await agent.arun("Summarize this clip", videos=["./demo.mp4"])

# Or via helper
from loomable.agent import video
result = await agent.arun("Describe", videos=[video(path="./clip.mp4")])
```

### Input: passing audio

For models that support audio input (e.g. Gemini, GPT-4o audio):

```python
# File path
result = await agent.arun("Transcribe this", audio=["./recording.wav"])

# URL
result = await agent.arun("Summarize", audio=["https://example.com/podcast.mp3"])

# Media class instance
from loomable.media import Audio
result = await agent.arun("Analyze tone", audio=[Audio(filepath="./call.wav")])

# Raw bytes
result = await agent.arun("Transcribe", audio=[audio_bytes])
```

The same auto-coercion rules apply: URL strings, file path strings, raw bytes, `MediaPart`, and `Audio` instances are all accepted. If the model does not support audio input, an `UnsupportedModalityError` is raised before the model call.

### Output: RunResult convenience properties

`RunResult` provides unified access to all media from a run — combining model-generated and tool-generated media:

```python
result = await agent.arun("Generate a chart and describe it")

# Text output (equivalent to result.output.text())
result.text

# Images from model output + tool results (model-first ordering)
result.images        # list[Image] — never None, empty list if none

# Audio from model output + tool results
result.audio         # list[Audio]

# Video from model output + tool results
result.videos        # list[Video]

# Files from tool results only (models don't generate arbitrary files)
result.files         # list[File]

# Save the first generated image
if result.images:
    result.images[0].save("./output.png")
    print(result.images[0].to_base64())
```

Ordering: model-generated media appears first, followed by tool-generated media in tool invocation order.

### Output: low-level access (still works)

```python
# Image output (when model generates images)
for img in result.output.images():
    img.data          # raw bytes (PNG, JPEG, etc.)
    img.uri           # or URL if returned as reference
    img.media_type    # "image/png", "image/jpeg", etc.

# Video output
for vid in result.output.videos():
    vid.data          # raw bytes
    vid.media_type    # "video/mp4", etc.

# All parts (mixed modalities)
result.output.parts        # list[MediaPart]
result.output.modalities() # set of Modality values present
```

### Tools returning media

`@tool` functions can return `Image`, `Audio`, or `Video` instances directly. Detection is automatic — no extra decorator parameters needed:

```python
from loomable.agent import Agent, tool
from loomable.media import Image

@tool
def generate_chart(data: str) -> Image:
    """Generate a chart from the provided data."""
    chart_bytes = my_chart_library.render(data)
    return Image(content=chart_bytes, format="png")

@tool
def capture_screenshot(url: str) -> Image:
    """Take a screenshot of a webpage."""
    return Image(url=f"https://screenshot-api.example.com/{url}")

agent = Agent(model="openai:gpt-4o-mini", tools=[generate_chart, capture_screenshot])
result = await agent.arun("Create a bar chart of Q4 sales")

# Tool-generated media is surfaced on RunResult properties
result.images[0].save("./chart.png")
print(result.text)  # model's text response referencing the chart
```

Tools can also return a list of media items:

```python
@tool
def generate_slides(topic: str) -> list:
    """Generate slide images for a presentation."""
    return [Image(content=slide, format="png") for slide in render_slides(topic)]
```

When a tool returns media, the framework:
1. Stores the media objects in `ToolResult.metadata["media"]`
2. Produces a text summary for the tool result content (e.g. `"[Image: png, filepath=chart.png]"`)
3. Surfaces media on `result.images` / `result.audio` / `result.videos`

### Feedback injection (multi-step reasoning)

By default, tool-generated media is fed back into the conversation so the model can "see" it in subsequent reasoning turns:

```python
agent = Agent(
    model="openai:gpt-4o-mini",
    tools=[generate_chart],
    feedback_media=True,  # default — model sees tool-generated media
)
result = await agent.arun("Generate a chart, then describe what you see in it")
# The model generates the chart via tool, sees the image, then describes it
```

To disable feedback injection (media still appears on `result.images` etc., but the model won't see it):

```python
agent = Agent(
    model="openai:gpt-4o-mini",
    tools=[generate_chart],
    feedback_media=False,  # media on RunResult but not injected into conversation
)
```

Feedback injection only occurs when the model's capabilities include the relevant modality as input. If a tool produces audio but the model doesn't support audio input, the media is captured on `result.audio` but not injected.

### Full capabilities (advanced)

Defaults already include text + image + video input and text output. Use
`ModelCapabilities` for audio input or image output, or to lock down modalities:

```python
from loomable.content import ModelCapabilities, Modality

agent = Agent(
    model="openai:gpt-4o-mini",
    capabilities=ModelCapabilities(
        input=frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO}),
        output=frozenset({Modality.TEXT, Modality.IMAGE}),
    ),
)
```

---

## Streaming

### Token chunks (NDJSON-friendly)

```python
agent = Agent(model="openai:gpt-4o-mini")

async for chunk in agent.astream("Tell me about AI"):
    if chunk.delta.data:
        print(chunk.delta.data.decode(), end="", flush=True)
    if chunk.done:
        print()  # final chunk
```

- Real token-level deltas when the provider supports `stream()`
- Automatic fallback to chunked output for non-streaming providers
- Same context assembly, memory, and capability gating as `arun()`

### AG-UI events (in-process)

```python
async for ev in agent.astream_events(prompt):
    print(ev.type, ev.data)
# RUN_STARTED → TEXT_* / TOOL_* → RUN_FINISHED

async for ev in case.astream_events(prompt):
    ...  # also NODE_* and STATE_*

async for ev in workflow.astream_events(prompt):
    ...  # NODE_STARTED / NODE_FINISHED
```

---

## AG-UI SSE

CopilotKit-compatible event types over `text/event-stream`. No hard CopilotKit dependency.

| Type family | Events |
|-------------|--------|
| Lifecycle | `RUN_STARTED`, `RUN_FINISHED`, `RUN_ERROR` |
| Text | `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END` |
| Tools | `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`, `TOOL_CALL_RESULT` |
| Graph | `NODE_STARTED`, `NODE_FINISHED` |
| State | `STATE_SNAPSHOT`, `STATE_DELTA` |

```python
from fastapi import FastAPI
from loomable.serve import mount_agent, mount_case

app = FastAPI()
# Optional api_key=: require Authorization: Bearer … or X-API-Key (401 if missing)
mount_agent(app, agent, prefix="/agent", api_key="secret")
mount_case(app, case, prefix="/cases", api_key="secret")
# POST /agent/run/events  → text/event-stream (disconnect → cancel)
# POST /cases/run/events  → text/event-stream
# POST /agent/run/stream  → application/x-ndjson (legacy)
```

See [SECURITY.md](../SECURITY.md) for trust boundaries.

---

## Flow Engine

The unified composition model. One primitive (`Runnable`), one composition path (`Flow` / `Workflow`).

### Core Concepts

| Concept | What |
|---------|------|
| `Runnable` | The protocol everything implements: `arun(input, *, context) -> RunResult` |
| `Loop` | Repeat a Runnable until a Verifier passes or a cap is hit |
| `Workflow` | Fluent multi-step process (compiles to Flow) |
| `Flow` | Directed graph of Runnables with **SharedState** and pluggable engines |
| `SharedState` | Key/value blackboard for the run (node outputs, `plan_steps`, board, …) |
| `Node` | A vertex in a Flow wrapping one Runnable |
| `Edge` | A directed connection between nodes (optionally gated by a condition) |
| `Map` | Fan-out one Runnable over a runtime list |
| `Router` | Select which downstream node(s) run next |

### Convenience Constructors

```python
from loomable.flow import sequential, parallel, route, coordinate, plan_and_execute

# Sequential chain (replaces Pipeline)
flow = sequential(step_a, step_b, step_c)

# Concurrent broadcast (replaces Orchestrator PARALLEL)
flow = parallel(researcher, analyst, writer)

# Predicate routing (replaces Orchestrator ROUTE)
flow = route(chooser_fn, {"research": researcher, "write": writer})

# Hierarchical delegation (replaces Orchestrator COORDINATE)
flow = coordinate(workers=[researcher, analyst], manager=synthesizer)

# Plan → Map → Synthesize (replaces AutoPlan)
flow = plan_and_execute(planner, worker, synthesizer)
```

### Engines

| Engine | When |
|--------|------|
| `SequentialEngine` | Linear chain — one node at a time |
| `ParallelEngine` | Independent branches — BSP supersteps |
| `HierarchicalEngine` | Manager delegates to workers |
| `engine="auto"` | Auto-selected from topology |

### SharedState + Reducers

```python
from loomable.flow import SharedState, overwrite, append, merge

# Default: overwrite (last-write-wins)
# append: accumulate into a list
# merge: shallow dict merge
# Custom: any (existing, incoming) -> merged function
```

### Full Package Exports

```python
from loomable.flow import (
    # Core
    Runnable, FunctionRunnable,
    # Tier 2
    Loop, Verifier, VerdictResult, AlwaysOkVerifier, CallableVerifier,
    # Tier 3
    Flow, FlowPlan, Node, Edge, Map, Router,
    # State
    SharedState, Reducer, overwrite, append, merge,
    # Engines
    ExecutionEngine, SequentialEngine, ParallelEngine, HierarchicalEngine,
    # Optimizer
    Optimizer, OptimizationRule,
    # Memory
    MemoryStore, Tier, TieredMemoryStore,
    # HITL
    FlowPaused,
    # Observability
    ContextSnapshotConfig, MessageDisposition, MessageSnapshot,
    # Helpers
    sequential, parallel, route, coordinate, plan_and_execute,
)
```

---

## Knowledge / RAG

```python
from loomable.providers import AzureOpenAIEmbedder

agent = Agent(
    model="openai:gpt-4o-mini",
    knowledge=["Your product docs...", "FAQ content...", "API reference..."],
    embedder=AzureOpenAIEmbedder(),
    knowledge_top_k=5,
)
# Each run: embed query → recall top-k chunks → inject as context
```

---

## Production Hardening

All opt-in. Zero overhead when unconfigured.

```python
from loomable.providers import RetryPolicy

agent = Agent(
    model="openai:gpt-4o-mini",
    tools=[search, calculator],
    
    # Transport resilience
    resilience=RetryPolicy(max_attempts=3, base_delay=0.5),
    
    # Tool execution bounds
    tool_timeout=5.0,          # seconds per tool call
    tool_concurrency=3,        # max parallel tools
    
    # Context bounds
    token_budget=8192,         # evict low-priority messages to fit
    
    # Loop safety
    loop_repeat_threshold=3,   # stop if same tool called 3x
    
    # Reasoning aids
    think_tool=True,           # scratchpad
    plan_tool=True,            # runtime escalation
    
    # Observability
    debug=True,                # JSON trace to stderr
    
    # Lifecycle hooks
    on_tool_call=lambda name, args: print(f"→ {name}"),
    on_complete=lambda r: log(r),
    
    # Human-in-the-loop
    require_confirmation=["send_email", "deploy"],
)
```

### Features

| Feature | What it does |
|---------|-------------|
| Resilience | Exponential backoff + jitter for 429/5xx/timeouts; fail-fast on 4xx |
| Tool timeout | Kill slow tools, feed error to model to replan |
| Concurrency cap | Limit parallel tool calls (protect downstream) |
| Token budget | Evict-then-admit with pinned system/schema messages |
| Loop detection | Stop no-progress loops with explicit reason |
| Idempotency | Non-idempotent tools never re-dispatched |
| HITL | Require approval for dangerous tools |
| Think tool | Zero-side-effect scratchpad |
| Plan tool | Model can self-escalate to fan-out |

---

## MCP Integration

Connect to Model Context Protocol servers — stdio (local) or SSE/HTTP (remote).

```python
agent = Agent(
    model="openai:gpt-4o-mini",
    mcp_servers=[
        # Stdio (local subprocess)
        {"command": "uvx", "args": ["some-mcp-server"], "env": {"KEY": "val"}},
        
        # SSE/HTTP (remote)
        {"url": "http://localhost:8080/sse", "headers": {"Authorization": "Bearer ..."}},
    ],
)
# Tools from MCP servers are auto-discovered and available to the model
```

- Lazy import: `mcp` SDK not loaded until an MCP server is configured
- Per-server fault isolation: one failing server doesn't break others
- Session lifecycle: transports cleaned up on agent close

---

## Serving

### HTTP (FastAPI) — Agent

```python
from fastapi import FastAPI
from loomable import Agent
from loomable.serve import mount_agent, FastAPIAdapter

agent = Agent(model=provider, tools=[search])

app = FastAPI()
mount_agent(app, agent, prefix="/agent", api_key="optional-shared-secret")
# GET  /agent/health
# POST /agent/run
# POST /agent/run/stream   (NDJSON; disconnect → BuiltAgent.cancel)
# POST /agent/run/events   (AG-UI SSE; disconnect → cancel)

# Or dual-mount at / and /agent:
app = FastAPIAdapter(agent, api_key="optional-shared-secret").app()
```

Auth (when `api_key=` is set): `Authorization: Bearer <key>` or `X-API-Key: <key>`.
Anonymous requests receive `401`. This is a shared-key edge baseline, not full RBAC.

### HTTP (FastAPI) — Case

```python
from loomable import Case
from loomable.serve import mount_case

case = Case(model=provider, goal="...", board=True, accept=ok)
mount_case(app, case, prefix="/cases", api_key="optional-shared-secret")
# POST /cases/run/events → text/event-stream
```

### MCP Server (expose agent as a tool)

```python
from loomable.serve import MCPServerAdapter

agent = Agent(model=provider, tools=[search])
server = MCPServerAdapter(agent.build()).server()
# Other MCP clients can discover and call this agent as a tool
```

---

## Checkpointing

Persist run state for resume, time-travel, and durable HITL.

### File-Based (default, human-readable)

```python
from loomable.persist import JsonFileCheckpointer, CheckpointConfig

checkpointer = JsonFileCheckpointer(
    location=".checkpoints",
    max_checkpoints=20,  # auto-prune oldest
)
```

### SQLite (high-frequency)

```python
from loomable.persist import SQLiteCheckpointer

checkpointer = SQLiteCheckpointer("agent.db", max_checkpoints=100)
```

### PostgreSQL (production)

```python
# pip install 'loomable[postgres]'
from loomable import PostgresCheckpointer

checkpointer = PostgresCheckpointer("postgresql://loomable:loomable@127.0.0.1:5432/loomable")
```

### Event-Driven Triggers

```python
from loomable.persist import CheckpointConfig

config = CheckpointConfig(
    on_events=["run_end", "tool_call"],  # checkpoint on these events
    max_checkpoints=10,
)
# Use with CheckpointListener wired to agent events
```

### Forking (explore alternatives)

```python
forked = await checkpointer.fork("original-thread", "what-if-branch")
# Both timelines continue independently
```

### Durable HITL

```python
from loomable.persist import PendingAction, Checkpoint

# Agent pauses with pending action → checkpointed
# Process can die here safely
# On restart: load checkpoint, see pending action, approve, resume
```

---

## Display & Visualization

Pretty-print results and visualize flow graphs. Works in both terminal and Jupyter.

### Pretty-print any result

```python
from loomable.display import pp

result = await agent.arun("Hello")
pp(result)
# ═══ Agent Result ═══
# Output:
#   Hello! How can I help?
# Tokens: 12 in / 8 out
```

`pp()` auto-detects the result type (agent, flow, loop, subagent) and formats accordingly. In Jupyter, it renders styled HTML.

### Access individual outputs

```python
from loomable.display import delegation_outputs, step_outputs

# Subagent delegation results by name:
result = await lead.arun("...")
outputs = delegation_outputs(result)
print(outputs["researcher"])    # researcher's output text
print(outputs["writer"])        # writer's output text

# Flow step results by node name:
result = await pipeline.arun("...")
steps = step_outputs(result)
print(steps["node_0"])          # first step output
print(steps["node_1"])          # second step output
```

### Visualize flow graphs

```python
from loomable.display import show_graph, mermaid_graph

# Terminal: prints Mermaid syntax (paste into mermaid.live)
# Jupyter: renders interactive SVG via Mermaid.js CDN
show_graph(my_flow, title="Research Pipeline")

# Get the raw Mermaid definition:
print(mermaid_graph(my_flow))
# graph TD
#   researcher[researcher<br/><small>Researcher</small>]
#   writer[writer<br/><small>Writer</small>]
#   researcher --> writer
```

### Output access patterns by primitive

| Primitive | Final output | Individual steps |
|-----------|-------------|-----------------|
| Agent | `result.output.text()` | `result.tool_activity` (tools called) |
| Agent + subagents | `result.output.text()` | `delegation_outputs(result)` → dict by name |
| Flow (sequential/parallel) | `result.output.text()` | `result.sub_results` dict or `step_outputs(result)` |
| Loop | `result.output.text()` | `result.metadata["loop_iterations"]`, `result.metadata["loop_verified"]` |
| Team | `result.output.text()` | Same as subagents — `delegation_outputs(result)` |

---

## Architecture Principles

- **Lean**: no mandatory deps beyond stdlib + httpx; flow-engine adds zero new mandatory dependencies
- **Decoupled**: every feature is a Protocol with a zero-dep default
- **Plug-and-play**: swap backends (vector DB, checkpointer, channels) without code changes
- **Kernel independence**: `loomable.kernel` imports nothing from edge layers; the flow-engine does not modify kernel
- **Opt-in everything**: unconfigured features have zero overhead
- **Fault isolation**: one tool/subagent/server failure never cascades
- **Progressive disclosure**: start with 3 lines, scale to multi-agent DAGs without rewriting
