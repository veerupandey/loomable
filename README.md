<div align="center" id="top">
  <h1>loomable</h1>
  <p>Agent · Team · Workflow · Case · AG-UI SSE</p>
  <p>
    <a href="https://github.com/veerupandey/loomable/actions/workflows/ci.yml"><img src="https://github.com/veerupandey/loomable/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI" /></a>
    <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+" />
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT" />
    <img src="https://img.shields.io/badge/status-beta-blue.svg" alt="Status: beta" />
  </p>
</div>

<p align="center">
  <a href="#get-started">Get Started</a> ·
  <a href="#api">API</a> ·
  <a href="#examples">Examples</a> ·
  <a href="docs/API.md">API Reference</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="SECURITY.md">Security</a>
</p>

**Public beta (`0.2.0b0`)** — Agent, Team, Workflow, Case, Memory, AG-UI. Polish gaps expected. Limits: local workspace FS, cooperative cancel, shared API-key serve auth. Full surface: [docs/API.md](docs/API.md).

```python
from loomable import Agent, Team, Workflow, Case, tool

@tool
def search(query: str) -> str:
    """Search the knowledge base."""
    return "Python was created in 1991."

agent = Agent(model=provider, tools=[search], knowledge_base=["./handbook.pdf"])
print((await agent.arun("When was Python created?")).output.text())
```

## Get Started

```bash
pip install "loomable @ git+https://github.com/veerupandey/loomable.git@v0.2.0b0"
# or
pip install -e ".[dev,toolkits]"
```

Copy [`.env.example`](.env.example) to `.env` (gitignored) or export `GEMINI_API_KEY` / `OPENAI_API_KEY` / Azure vars.

```python
import asyncio
from loomable import Agent
from loomable.providers.gemini import GeminiProvider

agent = Agent(model=GeminiProvider(), instructions="Be concise.")
print(asyncio.run(agent.arun("Capital of France?")).output.text())
```

## API

| Primitive | Role |
|-----------|------|
| **Agent** | Model + tools + memory + `knowledge_base=` |
| **Team** | `broadcast` / `sequential` (hard) · `coordinate` / `route` (soft) |
| **Workflow** | `.step` / `.parallel` / `.branch` / `.loop` + HITL + checkpoints |
| **Case** | Goal + WorkItems board + plan → dispatch → accept |
| **Memory** | `Memory.compose(conversation=..., user=...)` |
| **Deep agent** | `create_deep_agent(profile="research"\|"code")` |

```python
from loomable import Workflow, JsonFileCheckpointer, FlowPaused, Team, Case, create_deep_agent

wf = (
    Workflow("sev", session_id="inc-1", checkpointer=JsonFileCheckpointer("./ckpts"))
    .step("gather", gatherer)
    .parallel(analyst=analyst, visual=visual)
    .step("scribe", scribe, confirm=True)
)
try:
    result = await wf.arun(email)
except FlowPaused:
    await wf.approve("scribe")
    result = await wf.arun(resume=True)

team = Team(members=[sre, legal], mode="broadcast", hard=True)
case = Case(model=provider, goal="Close INC-88421", board=True, dispatch="spawn", accept=ok)
agent = create_deep_agent(provider, profile="research", workspace="./.deep_workspace")
```

```python
from fastapi import FastAPI
from loomable.serve import mount_agent, mount_case

app = FastAPI()
mount_agent(app, agent, prefix="/agent", api_key="secret")  # POST /agent/run/events
mount_case(app, case, prefix="/cases")                      # POST /cases/run/events
```

NDJSON `/run/stream` is Agent-only. Disconnect on `mount_agent` / `mount_case` calls `cancel()`.

Providers: `OpenAIProvider`, `AzureOpenAIProvider`, `AnthropicProvider`, `BedrockProvider`, `GeminiProvider`, `GroqProvider`, `OllamaProvider`.

```python
from loomable import Agent
from loomable.providers import BedrockProvider  # pip install "loomable[bedrock]"

# Amazon Bedrock (Converse API) — Claude, Nova, Llama, Mistral, ... via one surface.
# Auth uses the standard AWS chain (env / shared profile / SSO).
agent = Agent(model=BedrockProvider("amazon.nova-lite-v1:0", region_name="us-east-1"))
```

## Examples

```bash
python examples/agents/01_hello_world.py
python examples/agents/07_knowledge_base.py
python examples/escalation_war_room/10_case.py
```

See [`examples/README.md`](examples/README.md).

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/unit -q
```

## License

[MIT](LICENSE)
