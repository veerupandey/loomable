# Deep Agent (loomable)

LangGraph-style **deep agent harness** built on loomable primitives:

| Pillar | Loomable |
|--------|----------|
| Planning | `TodoTools` (`write_todos` / `read_todos` / `update_todo`) |
| Filesystem | `WorkspaceTools` (virtual FS + disk mirror) |
| Subagents | `task` tool → `spawn_specialist` (+ optional `subagents=` / Case) |
| Context | `think_tool`, memory stores, higher `max_tool_iterations` |

## Quick start

```python
from loomable import create_deep_agent

agent = create_deep_agent(model="openai:gpt-4o-mini", workspace="./.deep_workspace")
await agent.arun("Research X and write a brief to reports/x.md")
```

## Examples

| File | What it shows |
|------|----------------|
| `01_research_brief.py` | Scripted (CI) or live deep research loop |
| `02_case_deep_hard.py` | Deep tools + Case accept gate for hard tasks |

```bash
python examples/deep_agent/01_research_brief.py
DEEP_AGENT_LIVE=1 GEMINI_API_KEY=... python examples/deep_agent/01_research_brief.py
```

## Framework notes (found while building)

- Default Agent `max_tool_iterations=12` is too low for deep work → `create_deep_agent` uses **40**.
- Real `FileTools` lacked `edit_file` / search → added `edit_file`, `glob_files`, `grep_files`.
- Need a VFS for context offload without escaping sandboxes → `WorkspaceTools`.
- Need first-class todos outside Case boards → `TodoTools`.
- Need a general `task` tool (not only named `delegate_to_*`) → `make_task_tool`.
