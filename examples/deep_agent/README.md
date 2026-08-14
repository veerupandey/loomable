# Deep Agent (loomable)

LangGraph-style **deep agent harness** on loomable — built to **match and beat**
[langchain-ai/deepagents](https://github.com/langchain-ai/deepagents), and to
outpace Agno / CrewAI on **research defaults** (search → fetch → cite → vision →
verified deliverable) without locking you into LangGraph or a mega-toolkit catalog.

| Pillar | Loomable |
|--------|----------|
| Planning | `TodoTools` (`write_todos` / `read_todos` / `update_todo`) |
| Filesystem | `WorkspaceTools` (virtual FS + disk mirror + offload) |
| Subagents | `task` / `task_batch` + named `specialists=` + `delegate_to_*` |
| Context | think/plan, `Memory.compose`, LLM summarizer, **tool offload** (not truncate) |
| Research | Search + URL + Image + Citation (`verify_source` / `register_claim`) |
| Hard tasks | research accept gate, optional `mode="case"`, AG-UI / Team / Workflow |

## Why loomable wins on deep research

| Capability | deepagents | Agno | CrewAI | loomable |
|------------|------------|------|--------|----------|
| Search + full-page fetch | BYO | Toolkits / paid | BYO | **Bundled by default** |
| Large tool results | Offload to FS | Session/memory | Weak | Offload to **shared workspace** (`.offload/`) |
| Subagents | Named + async preview | Strong Team | Strong Crew | **`task` + `task_batch` + named specialists + shared FS** |
| Citations / claim basis | None | Integrations | Weak | **`register_source` / `verify_source` / `register_claim`** |
| Multimodal research | Text-first | Via integrations | Weak | **`fetch_image` + `analyze_image`** (SSRF-safe) |
| Deliverable gate | Soft | Soft | Manager review | **`write_file:reports/` + accept verifier** |
| Parallel fan-out | Multi-task | broadcast/tasks | async kickoff | **`tool_concurrency=4` + `task_batch`** |
| Code exec | Sandbox execute | Workspace shell | Limited | Opt-in `code_exec=True` (HITL) |
| Enterprise spine | LangGraph-only | AgentOS | Crews/Flows | **Agent / Team / Case / Workflow + AG-UI** |
| Stack lock-in | LangChain + LangGraph | Agno stack | Crew stack | **One framework** |

## Quick start

```python
from pathlib import Path
from loomable import (
    create_research_agent,
    Memory,
    ConversationMemory,
    UserMemory,
    open_session_store,
)
from loomable.agent.deep import SpecialistSpec

agent = create_research_agent(
    model="gemini:gemini-flash-latest",
    workspace="./.deep_workspace",
    memory=Memory.compose(
        conversation=ConversationMemory(store=open_session_store("file")),
        user=UserMemory(auto_extract=True),
    ),
    specialists={
        "web-researcher": SpecialistSpec(
            name="web-researcher",
            description="Finds and fetches primary sources",
        ),
    },
)
await agent.arun("Research X; write reports/x.md with citations")
```

Or the lower-level factory:

```python
from loomable import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-4o-mini",
    workspace="./.deep_workspace",
    code_exec=True,  # optional analysis sandbox
)
```

## Examples

| File | What it shows |
|------|----------------|
| `01_research_brief.py` | Scripted (CI) or live deep research loop |
| `02_case_deep_hard.py` | Deep tools + Case accept gate |
| `03_live_multimodal_research.py` | **Full research win**: search/fetch/cite/image/memory |
| `skills/research/SKILL.md` | Progressive skill loaded by the live demo |

```bash
python examples/deep_agent/01_research_brief.py
python examples/deep_agent/03_live_multimodal_research.py

DEEP_AGENT_LIVE=1 GEMINI_API_KEY=... \
  python examples/deep_agent/03_live_multimodal_research.py
```

## Framework notes

- Deep defaults: `token_budget=128000` (context), `max_run_tokens=0` (unbounded spend),
  `max_tool_iterations=40`, `tool_concurrency=4`, `tool_timeout=60`, model `RetryPolicy`.
- Truncating large tools loses evidence → deep post-hook **offloads** to `.offload/`.
  Specialists inherit the same offload hooks + skills + budgets.
- Uncapped fetch can blow context → deep caps URL extract at **8000** chars with markers;
  private hosts blocked on URL **and** image fetch (hop-by-hop redirect checks).
- `read_file` supports `offset`/`limit` so offload dumps stay sliced.
- `create_research_agent` requires `write_file` under `reports/` **and** `register_source`,
  then runs a research accept verifier (retry once on failure).
- `task_batch` fans out up to 8 specialists in parallel; `subagent_type` selects named
  specialists from `specialists=`.
- Live Gemini: use `GEMINI_MODEL=gemini-flash-latest` (older `gemini-2.0-flash` IDs may 404).
