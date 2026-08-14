# Deep Agent (loomable)

**A loomable-only deep agent that beats other deep-agent stacks** — no LangGraph,
no deepagents dependency, no Agno/Crew lock-in.

Built to outperform [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents),
Agno teams, and CrewAI crews on **research + long-horizon delivery**: shared
workspace offload, verified citations, parallel specialists, and an accept gate
that refuses to finish without a real report.

| Pillar | Loomable |
|--------|----------|
| Planning | `TodoTools` (`write_todos` / `read_todos` / `update_todo`) |
| Filesystem | `WorkspaceTools` + **token-aware** offload (not truncate) |
| Subagents | `task` / `task_batch` + named `specialists=` + Case spawn |
| Context | `compact_conversation`, think/plan, `Memory.compose`, summarizer |
| Research | Search + URL + Image + Citation (`verify_source` / `register_claim`) |
| Hard tasks | research accept gate, optional `mode="case"`, AG-UI / Team / Workflow |

## Why loomable deep agents win

| Capability | deepagents | Agno | CrewAI | **loomable** |
|------------|------------|------|--------|--------------|
| Pure stack | LangChain+LangGraph | Agno | Crew | **loomable only** |
| Search + full-page fetch | BYO | Toolkits / paid | BYO | **Bundled** |
| Large tool results | Offload | Session/memory | Weak | **Token-aware shared `.offload/`** |
| Subagents | Named + async preview | Strong Team | Strong Crew | **`task` + `task_batch` + shared FS** |
| Citations / claims | None | Integrations | Weak | **`register` / `verify` / `register_claim`** |
| Multimodal research | Text-first | Integrations | Weak | **Vision tools (SSRF-safe)** |
| Deliverable gate | Soft | Soft | Manager | **`reports/` + accept verifier** |
| Parallel fan-out | Multi-task | broadcast | async | **`tool_concurrency` + `task_batch`** |
| Context compact | Strong story | Good | Good | **`compact_conversation` + L2** |
| Code exec | Sandbox | Shell+HITL | Limited | Opt-in `code_exec` + HITL |
| Enterprise spine | LangGraph | AgentOS | Flows | **Agent / Team / Case / Workflow + AG-UI** |

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
    memory_files=[Path("AGENTS.md")],  # optional always-on project memory
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

- Deep defaults: `token_budget=128000`, `max_run_tokens=0`, `max_tool_iterations=40`,
  `tool_concurrency=4`, `tool_timeout=60`, model `RetryPolicy`,
  **token-aware offload** (`offload_threshold_tokens=3000`).
- Specialists inherit offload hooks + skills + budgets from the deep factory.
- Case mode inherits the same runtime knobs (iterations, hooks, skills, budgets).
- URL + image fetch: private hosts blocked with hop-by-hop redirect checks.
- `create_research_agent` requires `write_file` under `reports/` **and**
  `register_source`, then runs a research accept verifier.
- `task_batch` fans out up to 8 specialists; `subagent_type` selects named specialists.
- Live Gemini: use `GEMINI_MODEL=gemini-flash-latest`.
