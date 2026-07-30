# Loomable — High-Level API Reference

A lightweight, production-grade agent framework. Build agents in 3 lines, scale to multi-agent pipelines with memory, streaming, and tool orchestration.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Agent](#agent)
- [Tools](#tools)
- [Model Providers](#model-providers)
- [Memory](#memory)
- [Structured Output](#structured-output)
- [Streaming](#streaming)
- [Multi-Agent Orchestration](#multi-agent-orchestration)
- [Pipeline](#pipeline)
- [Channels](#channels)
- [Knowledge / RAG](#knowledge--rag)
- [Production Hardening](#production-hardening)
- [MCP Integration](#mcp-integration)
- [Serving](#serving)
- [Checkpointing](#checkpointing)

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

### Universal Memory (works for agents, pipelines, sub-agents)

```python
# Same pattern everywhere — session_id is the key
pipeline = Pipeline(steps=[agent1, agent2], session_id="conv-1")
pipeline.run("Write about AI")
pipeline.run("Make it shorter")  # → remembers

coordinator = Agent(model="...", sub_agents=[a, b], mode="coordinate", session_id="s1")
coordinator.run("Research")
coordinator.run("Follow up")  # → sub-agents see history
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

## Multi-Agent Orchestration

### Modes

```python
from loomable.agent import Agent

researcher = Agent(model="openai:gpt-4o-mini", name="researcher", tools=[search])
writer = Agent(model="anthropic:claude-sonnet-4-20250514", name="writer")
analyst = Agent(model="groq:llama-3.3-70b-versatile", name="analyst")
```

**PARALLEL** — broadcast to all, run concurrently, aggregate:
```python
team = Agent(model="openai:gpt-4o-mini", sub_agents=[researcher, writer, analyst], mode="parallel")
result = await team.arun("Analyze the AI market")
```

**ROUTE** — select one sub-agent, run only that one:
```python
router = Agent(model="openai:gpt-4o-mini", sub_agents=[researcher, writer], mode="route")
```

**COORDINATE** — run all in parallel, then leader synthesizes:
```python
lead = Agent(model="openai:gpt-4o-mini", sub_agents=[researcher, writer], mode="coordinate")
```

**PLAN** — autonomous decomposition → parallel subagents → synthesis:
```python
planner = Agent(model="openai:gpt-4o-mini", mode="plan", max_plan_steps=5, tools=[search])
result = await planner.arun("Compare 3 frameworks step by step and write a recommendation")
```

### Complexity Router (auto-escalation)

```python
from loomable.agent import ComplexityRouter

agent = Agent(
    model="openai:gpt-4o-mini",
    tools=[search],
    complexity_router=ComplexityRouter(),
)
# Simple input → single-shot
# Medium input → tool loop (ReAct)
# Complex input → auto-plan with parallel subagents
```

### Think Tool (scratchpad reasoning)

```python
agent = Agent(model="openai:gpt-4o-mini", tools=[search], think_tool=True)
# Model gets a zero-side-effect scratchpad for reasoning before acting
```

### Plan Tool (runtime escalation)

```python
agent = Agent(model="openai:gpt-4o-mini", tools=[search], plan_tool=True)
# Model can choose to decompose a hard task mid-run
```

---

## Pipeline

Sequential multi-agent execution with optional iterative refinement and memory.

### Basic Pipeline

```python
from loomable.agent import Pipeline

pipeline = Pipeline(steps=[researcher, writer])
result = await pipeline.run("Write about AI agents")
```

### With Iterative Refinement

```python
from loomable.agent import Pipeline, InMemoryChannel

feedback = InMemoryChannel(name="feedback")
pipeline = Pipeline(
    steps=[researcher, writer, critic],
    feedback_channel=feedback,
    max_iterations=3,
    stop_condition=lambda text: "APPROVED" in text,
)
result = await pipeline.run("Write a polished article")
# critic loops back until it says APPROVED (or max_iterations hit)
```

### With Memory (multi-turn follow-up)

```python
pipeline = Pipeline(steps=[researcher, writer], session_id="article-project")
await pipeline.run("Write about frameworks")
await pipeline.run("Add a comparison table")  # remembers the article
await pipeline.run("Now shorten the intro")   # still remembers
```

---

## Channels

Decoupled message-passing between agents. Protocol-based — swap in Redis/Kafka later.

```python
from loomable.agent import InMemoryChannel, ChannelMessage

channel = InMemoryChannel(name="research-to-writer")

# Agent A writes
await channel.send(ChannelMessage(sender="researcher", content="findings..."))

# Agent B reads
msg = await channel.receive(timeout=5.0)
print(msg.content)  # "findings..."

# Inspect history
history = await channel.peek()
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

- **Lean**: no mandatory deps beyond stdlib + httpx
- **Decoupled**: every feature is a Protocol with a zero-dep default
- **Plug-and-play**: swap backends (vector DB, checkpointer, channels) without code changes
- **Kernel independence**: `loomable.kernel` imports nothing from edge layers
- **Opt-in everything**: unconfigured features have zero overhead
- **Fault isolation**: one tool/subagent/server failure never cascades
