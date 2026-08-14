# Deep Agent (loomable)

LangGraph-style **deep agent harness** on loomable — built to **match and beat**
[langchain-ai/deepagents](https://github.com/langchain-ai/deepagents) for research.

| Pillar | Loomable |
|--------|----------|
| Planning | `TodoTools` (`write_todos` / `read_todos` / `update_todo`) |
| Filesystem | `WorkspaceTools` (virtual FS + disk mirror) |
| Subagents | `task` → `spawn_specialist` (**shared workspace**) |
| Context | think/plan, `Memory.compose`, LLM summarizer, **tool offload** (not truncate) |
| Research | `WebSearchTools` + `URLTools` + `ImageTools` + `CitationTools` |
| Hard tasks | optional `mode="case"` accept gate + AG-UI / Team / Workflow |

## Why loomable beats deepagents (research)

| Capability | deepagents | loomable |
|------------|------------|----------|
| Search + full-page fetch | BYO tools | Bundled by default |
| Large tool results | Offload to FS | Offload to **shared workspace** (`.offload/`) |
| Subagents | Isolated context | Isolated LLM context **+ shared todos/files** |
| Citations | None | `register_source` / `format_bibliography` |
| Multimodal research | Text-first harness | `fetch_image` + `analyze_image` (vision) |
| Memory | AGENTS.md / store | `Memory.compose` (conversation + user + knowledge) |
| Enterprise spine | LangGraph-only | Agent / Team / **Case** / Workflow + AG-UI |
| Stack lock-in | LangChain + LangGraph | **One framework** |

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

agent = create_research_agent(
    model="gemini:gemini-2.0-flash",
    workspace="./.deep_workspace",
    memory=Memory.compose(
        conversation=ConversationMemory(store=open_session_store("file")),
        user=UserMemory(auto_extract=True),
    ),
)
await agent.arun("Research X; write reports/x.md with citations")
```

Or the lower-level factory:

```python
from loomable import create_deep_agent

agent = create_deep_agent(model="openai:gpt-4o-mini", workspace="./.deep_workspace")
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

- Default Agent `max_tool_iterations=12` is too low → deep uses **40**.
- Truncating large tools loses evidence → deep post-hook **offloads** to `.offload/`.
- `modalities` default for deep is **`text+image`** (override with `modalities="text"`).
- `create_research_agent` is an opinionated alias over `create_deep_agent`.
