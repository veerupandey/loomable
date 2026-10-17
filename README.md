<div align="center">
  <h1>Loomable</h1>
  <p>A lightweight Python framework for agents, specialist teams, and workflows.</p>
  <p>
    <a href="https://github.com/veerupandey/loomable/actions/workflows/ci.yml"><img src="https://github.com/veerupandey/loomable/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB" alt="Python 3.11 or newer">
    <img src="https://img.shields.io/badge/status-beta-8250df" alt="Beta">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2da44e" alt="MIT license"></a>
  </p>
  <p>
    <a href="#quickstart">Quickstart</a> ·
    <a href="examples/README.md">Examples</a> ·
    <a href="docs/API.md">API reference</a> ·
    <a href="CHANGELOG.md">Changelog</a>
  </p>
</div>

Start with a model and a Python function. As your application grows, add session memory, retrieval, specialist agents, or checkpointed workflows. The base install depends on HTTPX and Pydantic. HTTP serving, MCP, and PostgreSQL are optional extras.

## Quickstart

Install from the repository with Python 3.11 or newer.

```bash
git clone https://github.com/veerupandey/loomable.git
cd loomable
python -m pip install -e '.[examples]'
```

Set `LOOMABLE_PROVIDER=openai` and `OPENAI_API_KEY` in your shell or the repository's ignored `.env` file. To use Gemini, set `LOOMABLE_PROVIDER=gemini` and `GEMINI_API_KEY`. Use [`.env.example`](.env.example) as a template.

```bash
python -m examples.agents.01_hello_world
python -m examples.agents.02_with_tools
```

The second command asks the model to look up stock in a local inventory fixture. Here is the same pattern in a standalone script. Set the provider key in your shell before running it.

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

## Choose a building block

| Building block | Use it when |
|---|---|
| [Agent](docs/API.md#agent) | A model needs tools, structured output, or conversation history. |
| [Team](docs/API.md#team) | Specialists need to review in parallel, work in sequence, or delegate based on the request. |
| [Workflow](docs/API.md#workflow) | Your application defines the steps, branches, state, and approval gates. |
| [Case](docs/API.md#case) | Work starts from a goal and needs a plan, a work-item board, and an acceptance check. |
| [Deep agent](docs/API.md#deep-agent) | A research or coding task needs planning, a workspace, and task delegation. |

These building blocks compose. An agent or team can run as a workflow step. [Memory](docs/API.md#memory), [document retrieval](docs/API.md#knowledge-and-retrieval), and [checkpointing](docs/API.md#checkpointing) can be added independently.

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

## Development

```bash
python -m pip install -e '.[dev,toolkits]'
python -m pytest tests/unit -q
```

Loomable is in beta. See the [security policy](SECURITY.md) for serving, sandbox, and deployment boundaries. Released under the [MIT License](LICENSE).
