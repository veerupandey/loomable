# Loomable — High-Level API Reference

A lightweight, production-grade agent framework. Build agents in 3 lines, scale to multi-agent workflows with memory, streaming, and tool orchestration.

---

## Table of Contents

- [Progressive Disclosure (Levels 0–7)](#progressive-disclosure-levels-07)
- [Quick Start](#quick-start)
- [Agent](#agent)
- [Tools](#tools)
- [Model Providers](#model-providers)
- [Memory](#memory)
- [Structured Output](#structured-output)
- [Streaming](#streaming)
- [Flow Engine](#flow-engine)
- [Knowledge / RAG](#knowledge--rag)
- [Production Hardening](#production-hardening)
- [MCP Integration](#mcp-integration)
- [Serving](#serving)
- [Checkpointing](#checkpointing)

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

### Level 4: Flow with sequential list shorthand

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
    description="Finds papers",         # optional description
    instructions="Be concise.",         # system prompt
    tools=[...],                        # list of @tool-decorated functions
    session_id="conv-1",                # enables multi-turn memory
    debug=True,                         # prints trace to stderr
)
```

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

Memory is automatic when you set `session_id`. No configuration needed for the common case.

### Conversation Memory (short-term)

```python
agent = Agent(model="openai:gpt-4o-mini", session_id="conv-1")
agent.run("My name is Alice")
agent.run("What's my name?")  # → "Alice"
```

### Memory Configuration

```python
agent = Agent(
    model="openai:gpt-4o-mini",
    session_id="conv-1",
    use_memory=True,               # default True when session_id is set
    memory_window=8,               # last N turns replayed verbatim
    compaction_threshold=16,       # summarize when turns exceed this
    use_llm_summarizer=True,       # model-based summarization
)
```

### Memory Tiers

| Tier | What | Scope | Trigger |
|------|------|-------|---------|
| L1 (turns) | Raw conversation messages | Per session | Automatic |
| L2 (summaries) | Compressed older history | Per session | Auto-compaction at threshold |
| L3 (episodic) | Vector-indexed long-term facts | Cross-session | NoteStore / memory_tool |

### Pinned Facts

```python
built = agent.build()
built.pin_fact("API key: sk-abc123")  # never summarized away
```

### Cross-Session Notes (long-term memory)

```python
from loomable.agent import NoteStore
from loomable.kernel import LongTermStore
from loomable.providers import AzureOpenAIEmbedder

store = NoteStore(
    long_term=LongTermStore(),
    embedder=AzureOpenAIEmbedder(),
)

agent = Agent(
    model="openai:gpt-4o-mini",
    memory_tool=True,
    note_store=store,
)
# The model can now write/read/recall durable notes across sessions
```

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

## Streaming

```python
agent = Agent(model="openai:gpt-4o-mini")

async for chunk in agent.astream("Tell me about AI"):
    if chunk.delta.data:
        print(chunk.delta.data.decode(), end="", flush=True)
    if chunk.done:
        print()  # final chunk
```

- Real token-level deltas when the provider supports `stream()` (OpenAI, Azure, Anthropic, Groq, Ollama, Gemini)
- Automatic fallback to chunked output for non-streaming providers
- Same context assembly, memory, and capability gating as `arun()`

---

## Flow Engine

The unified composition model replacing the previous `Pipeline`, `Orchestrator`, and `AutoPlan` classes. One primitive (`Runnable`), one composition path (`Flow`).

### Core Concepts

| Concept | What |
|---------|------|
| `Runnable` | The protocol everything implements: `arun(input, *, context) -> RunResult` |
| `Loop` | Repeat a Runnable until a Verifier passes or a cap is hit |
| `Flow` | A directed graph of Runnables with shared state and pluggable engines |
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

### HTTP (FastAPI)

```python
from loomable.serve import FastAPIAdapter

agent = Agent(model="openai:gpt-4o-mini", tools=[search])
app = FastAPIAdapter(agent.build()).app()

# Run with: uvicorn app:app
# Routes: GET /health, POST /run, POST /run/stream
```

### MCP Server (expose agent as a tool)

```python
from loomable.serve import MCPServerAdapter

agent = Agent(model="openai:gpt-4o-mini", tools=[search])
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

## Architecture Principles

- **Lean**: no mandatory deps beyond stdlib + httpx; flow-engine adds zero new mandatory dependencies
- **Decoupled**: every feature is a Protocol with a zero-dep default
- **Plug-and-play**: swap backends (vector DB, checkpointer, channels) without code changes
- **Kernel independence**: `loomable.kernel` imports nothing from edge layers; the flow-engine does not modify kernel
- **Opt-in everything**: unconfigured features have zero overhead
- **Fault isolation**: one tool/subagent/server failure never cascades
- **Progressive disclosure**: start with 3 lines, scale to multi-agent DAGs without rewriting
