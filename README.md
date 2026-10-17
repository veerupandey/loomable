<div align="center">

# Loomable

**A lightweight Python framework for agents, specialist teams, and workflows.**

[![CI](https://github.com/veerupandey/loomable/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/veerupandey/loomable/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-beta-8250df)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-2da44e)](LICENSE)

[Quickstart](#quickstart) · [Building blocks](#building-blocks) · [Examples](examples/README.md) · [API reference](docs/API.md) · [Changelog](CHANGELOG.md)

</div>

---

Start with a model and a Python function. As your application grows, add session memory, retrieval, specialist agents, or checkpointed workflows — without changing frameworks.

The base install depends only on **HTTPX** and **Pydantic**. HTTP serving, MCP, and PostgreSQL are optional extras.

## Why Loomable

- **Progressive disclosure.** Begin with `Agent`. Move to `Team`, `Workflow`, or `Case` only when the problem actually needs them.
- **Composable by design.** An agent or a team can run as a workflow step. Memory, retrieval, and checkpointing attach independently.
- **Durable when it matters.** Checkpointed workflows, resumable sessions, and explicit approval gates for tools that change state.
- **Small surface area.** Two runtime dependencies, typed throughout, with a single import path: `from loomable import ...`.

## Quickstart

### 1. Install

Requires **Python 3.11 or newer**.

```bash
git clone https://github.com/veerupandey/loomable.git
cd loomable
python -m pip install -e '.[examples]'
```

### 2. Configure a provider

```bash
export LOOMABLE_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

For Gemini, use `LOOMABLE_PROVIDER=gemini` and `GEMINI_API_KEY`. Copy [`.env.example`](.env.example) to `.env` (gitignored) to keep keys out of your shell history.

| Provider | Environment | Default model |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-5.4-mini` |
| Gemini | `GEMINI_API_KEY` | `gemini-flash-latest` |
| Azure OpenAI | `AZURE_OPENAI_*` | deployment name |
| Amazon Bedrock | AWS credential chain | `amazon.nova-lite-v1:0` |

### 3. Run an example

```bash
python -m examples.agents.01_hello_world
python -m examples.agents.02_with_tools
```

The second command asks the model to look up stock in a local inventory fixture.

## Your first agent

```python
from loomable import Agent, tool

inventory = {"WIDGET": 12, "SPROCKET": 0}


@tool
def stock_count(sku: str) -> int:
    """Return the number of units in stock for this SKU."""
    return inventory[sku.upper()]


agent = Agent(
    model="openai:gpt-5.4-mini",
    tools=[stock_count],
    require_tools=["stock_count"],
    strict_require_tools=True,
)

print(agent.run("How many WIDGET units are in stock?").text)
```

Use `await agent.arun(...)` in async applications. To switch providers, pass `model="gemini:gemini-flash-latest"` or a configured provider instance.

## Building blocks

| Building block | Use it when |
|---|---|
| [`Agent`](docs/API.md#agent) | A model needs tools, structured output, or conversation history. |
| [`Team`](docs/API.md#team) | Specialists need to review in parallel, work in sequence, or delegate based on the request. |
| [`Workflow`](docs/API.md#workflow) | Your application defines the steps, branches, state, and approval gates. |
| [`Case`](docs/API.md#case) | Work starts from a goal and needs a plan, a work-item board, and an acceptance check. |
| [Deep agent](docs/API.md#deep-agent) | A research or coding task needs planning, a workspace, and task delegation. |

These building blocks compose. An agent or team can run as a workflow step. [Memory](docs/API.md#memory), [document retrieval](docs/API.md#knowledge-and-retrieval), and [checkpointing](docs/API.md#checkpointing) can be added independently.

### Composing a workflow

```python
from loomable import Agent, Workflow

performance = Agent(model, instructions="Identify one performance risk.")
security = Agent(model, instructions="Identify one security risk.")
writer = Agent(model, instructions="Combine the reviews into a short recommendation.")

workflow = (
    Workflow("API review")
    .parallel(performance=performance, security=security)
    .step("recommend", writer)
)

result = await workflow.arun("We plan to add an endpoint that exports all orders as CSV.")
print(result.text)
```

## Learn by doing

| I want to build… | Start here |
|---|---|
| An agent that returns validated JSON | [Classify a support ticket](examples/agents/03_structured_io.py) |
| A conversation that survives rebuilding the agent | [Resume a saved conversation](examples/memory/01_session_memory.py) |
| A workflow that gathers evidence and writes a recommendation | [Run parallel reviews](examples/patterns/02_pipeline.py) |
| A workflow with an approval gate | [Pause, approve, and resume](examples/advanced/03_checkpointing.py) |
| An HTTP service with AG-UI events | [Serve an agent locally](examples/README.md#extend-and-serve) |
| A complete application | [Read-only incident triage](examples/incident_triage/README.md) |

The [examples guide](examples/README.md) lists the commands, prerequisites, and expected outputs. The [API reference](docs/API.md) covers configuration and extension points.

## Optional extras

Install only what you deploy.

```bash
python -m pip install -e '.[serve]'      # FastAPI + uvicorn, AG-UI SSE
python -m pip install -e '.[mcp]'        # Model Context Protocol servers
python -m pip install -e '.[postgres]'   # Postgres memory and checkpoints
python -m pip install -e '.[retrieval]'  # Vector stores and embeddings
```

| Extra | Adds |
|---|---|
| `serve` | FastAPI and uvicorn for AG-UI event streaming |
| `mcp` | MCP client and server transports |
| `postgres` | `asyncpg` for Postgres memory and checkpointing |
| `bedrock` | `boto3` for Amazon Bedrock |
| `retrieval` | zvec, FAISS, Chroma, Milvus, and sentence-transformers |
| `toolkits` | PDF, PPTX, URL, and web-search tools |
| `dev` | pytest, hypothesis, and the full test toolchain |

## Development

```bash
python -m pip install -e '.[dev,toolkits]'
python -m pytest tests/unit -q
```

## Project status

Loomable is in **beta**. APIs may change between minor versions; see the [changelog](CHANGELOG.md) for what moved.

Read the [security policy](SECURITY.md) before exposing an agent over HTTP — it covers serving, sandbox, and deployment boundaries.

Released under the [MIT License](LICENSE).
