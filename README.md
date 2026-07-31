<div align="center" id="top">
  <h1>loomable</h1>
  <p>Build agents that think, use tools, and compose into flows.</p>
</div>

<p align="center">
  <a href="#get-started">Get Started</a> ·
  <a href="#features">Features</a> ·
  <a href="#examples">Examples</a> ·
  <a href="docs/API.md">API Reference</a>
</p>

## Introduction

Loomable is a lightweight Python framework for building AI agents. Three tiers, one interface:

- **Agent** — single model call with tools, memory, and knowledge
- **Loop** — retry with verification until correct
- **Flow** — directed graph of agents, functions, and loops

Everything is a `Runnable`. Agents compose into loops, loops compose into flows, flows compose into flows.

```python
from loomable.agent import Agent, tool
from loomable.providers.openai import AzureOpenAIProvider

@tool
def search(query: str) -> str:
    """Search the knowledge base."""
    return "Result: Python was created in 1991."

agent = Agent(
    model=AzureOpenAIProvider(),
    instructions="You are a research assistant.",
    tools=[search],
)
result = await agent.arun("When was Python created?")
print(result.output.text())
```

## Get Started

### Installation

```bash
# From GitHub (recommended until published to PyPI)
pip install "loomable @ git+https://github.com/veerupandey/loomable.git"

# Or with uv
uv add "loomable @ git+https://github.com/veerupandey/loomable.git"

# Or clone and install locally
git clone https://github.com/veerupandey/loomable.git
cd loomable
pip install -e .
```

### Set up your provider

```bash
# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-4o-mini"

# Or OpenAI
export OPENAI_API_KEY="sk-..."
```

### Your first agent

```python
import asyncio
from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

agent = Agent(
    model=AzureOpenAIProvider(),
    instructions="You are a helpful assistant.",
)

result = asyncio.run(agent.arun("What is the capital of France?"))
print(result.output.text())
# => The capital of France is Paris.
```

### Add tools

```python
from loomable.agent import Agent, tool
from loomable.providers.openai import AzureOpenAIProvider

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

agent = Agent(
    model=AzureOpenAIProvider(),
    tools=[multiply],
)
result = await agent.arun("What is 7 * 8?")
```

The agent enters a tool-use loop automatically: call tools, feed results back, repeat until done.

### Add a loop

```python
from loomable.agent import Agent
from loomable.flow import Loop

agent = Agent(model=provider, instructions="Write a haiku (3 lines).")

loop = Loop(
    body=agent,
    verifier=lambda output, ctx: len(output.text().strip().split("\n")) == 3,
    max_iterations=3,
)
result = await loop.arun("Write a haiku about code.")
```

### Compose a flow

```python
from loomable.flow import sequential, parallel, coordinate, Flow, Edge

# --- Helpers (most common) ---

# Sequential: research → draft → edit
pipeline = sequential(research_fn, draft_fn, edit_fn)

# Parallel: run 3 reviewers concurrently
reviews = parallel(security_fn, performance_fn, ux_fn)

# Coordinate: workers in parallel, then a manager synthesizes
team = coordinate(
    workers=[security_fn, performance_fn, ux_fn],
    manager=synthesize_fn,
)
result = await team.arun("Review this pull request")

# --- Low-level Flow (custom graph topology) ---

flow = Flow(
    nodes={"research": research_fn, "draft": draft_fn, "review": review_fn},
    edges=[
        Edge(source="research", target="draft"),
        Edge(source="research", target="review"),  # fan-out
    ],
    engine="parallel",
)
result = await flow.arun("Build a feature spec")
```

## Features

| Feature | What it does |
|---------|-------------|
| **Function tools** | `@tool` decorator — auto JSON schema from type hints |
| **Tool-use loop** | Model calls tools, results feed back, until done |
| **Think & Plan** | Built-in reasoning scratchpad and dynamic task decomposition |
| **Memory** | Conversational history with automatic compaction |
| **Knowledge (RAG)** | Embed docs at build time, recall into context at run time |
| **Tiered routing** | Primary/fallback model tiers with automatic failover |
| **Structured I/O** | Pydantic/dataclass input validation and output parsing |
| **Verification** | Verifier protocol — same interface for Agent, Loop, and Flow |
| **Skills** | Load tool packages from directories |
| **MCP** | Connect to Model Context Protocol servers |
| **HITL** | Pre/post hooks, tool approval gates, pause & resume |
| **Parallel flows** | Concurrent branches with result collection |
| **Routing** | Dynamic branch selection based on input |
| **Coordination** | Workers + manager synthesis pattern |
| **Observability** | Structured events, traces, and run metadata |

## Examples

Run any example:

```bash
python examples/01_simple_agent.py
```

### Agents

| File | What it shows |
|------|---------------|
| `01_simple_agent.py` | Minimal agent — 3 lines |
| `02_agent_think_plan.py` | Think scratchpad + plan escalation |
| `03_agent_with_tools.py` | `@tool` decorator, automatic tool loop |
| `04_agent_structured_io.py` | Input validation + structured output |
| `05_agent_subagents.py` | Plan-based subagent delegation |

### Loops

| File | What it shows |
|------|---------------|
| `06_simple_loop.py` | Retry until verifier passes |
| `07_loop_with_tools.py` | Tool-using agent inside a loop |
| `08_loop_subagent_delegation.py` | Research + fact-check loop |

### Flows

| File | What it shows |
|------|---------------|
| `09_sequential_flow.py` | Research → Draft → Edit pipeline |
| `10_parallel_flow.py` | Concurrent branches |
| `11_route_flow.py` | Dynamic routing by intent |
| `12_coordinate_flow.py` | Workers + manager synthesis |
| `13_plan_and_execute_flow.py` | Plan → Map → Synthesize |
| `14_complex_flow_with_loops.py` | Loop nodes inside a flow |
| `15_nested_flow_subagents.py` | Multi-level flow composition |

### Memory & Knowledge

| File | What it shows |
|------|---------------|
| `16_agent_memory.py` | Multi-turn memory + compaction |
| `17_flow_memory.py` | TieredMemoryStore across nodes |
| `18_knowledge_rag.py` | Embedder + document recall |

### MCP, Skills & Advanced

| File | What it shows |
|------|---------------|
| `19_mcp_agent.py` | MCP server tools on an agent |
| `20_mcp_in_flow.py` | MCP tools inside a flow |
| `21_skills_agent.py` | Skill directory loading |
| `22_tiered_routing.py` | Model fallback tiers |
| `23_hitl_approval.py` | Tool approval + safety hooks |
| `24_full_production_agent.py` | All features combined |

## Architecture

```
loomable/
├── agent/       # Tier 1: Agent builder, tools, memory, reasoning
├── flow/        # Tier 2 & 3: Loop, Flow, engines, helpers
├── content/     # Input/output coercion and media types
├── kernel/      # Core primitives (never modified by extensions)
├── providers/   # Model providers: OpenAI, Azure, Anthropic, Gemini, Ollama
├── persist/     # Checkpointing and resume
└── serve/       # FastAPI + MCP adapters
```

The kernel is import-independent — it never imports from agent, content, serve, or providers.

## Providers

| Provider | Class |
|----------|-------|
| OpenAI | `OpenAIProvider` |
| Azure OpenAI | `AzureOpenAIProvider` |
| Anthropic | `AnthropicProvider` |
| Google Gemini | `GeminiProvider` |
| Groq | `GroqProvider` |
| Ollama | `OllamaProvider` |

All providers read credentials from environment variables by default.

## Contributing

PRs welcome. Run tests with:

```bash
python -m pytest tests/
```

## License

MIT

<p align="right"><a href="#top">↑ Back to top</a></p>
