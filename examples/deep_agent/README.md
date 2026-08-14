# Deep Agent (loomable)

**One deep agent. Skills specialize it.** Research any topic with the bundled
``research`` skill — you do **not** need a separate research agent type.

```python
from loomable import create_deep_agent

# Research anything (science, policy, product, …)
agent = create_deep_agent(model, profile="research", workspace="./.deep_workspace")
# equivalent: create_deep_agent(model, skills=["research"], ...)
```

`create_research_agent` remains a thin alias for `profile="research"`.

| Pillar | Loomable |
|--------|----------|
| Planning | `TodoTools` |
| Filesystem | `WorkspaceTools` + token-aware offload + `delete_file` |
| Subagents | `task` / `task_batch` + named specialists |
| Context | `compact_conversation`, Memory, summarizer |
| Skills | Bundled `research` (any topic) via `loomable.skills` |
| Discovery | `search_skills` / `load_skill`, `search_tools`, `search_mcp` / `activate_tool` (on by default) |
| Hard tasks | accept gates, Case, AG-UI / Team / Workflow |

## Why this beats peer deep agents

| Capability | deepagents | Agno | CrewAI | **loomable** |
|------------|------------|------|--------|--------------|
| Pure stack | LangChain+LangGraph | Agno | Crew | **loomable only** |
| Research specialization | Separate harness ideas | Toolkits | Crew roles | **Skill + profile** |
| Shared FS offload | Yes | Session | Weak | **Token-aware `.offload/`** |
| Citations / claims | None | Integrations | Weak | **verify + claim tools** |
| Parallel fan-out | Multi-task | broadcast | async | **`task_batch`** |
| Enterprise spine | LangGraph | AgentOS | Flows | **Case / Team / Workflow / AG-UI** |

## Quick start

```python
from loomable import create_deep_agent, Memory, ConversationMemory, UserMemory

agent = create_deep_agent(
    model="gemini:gemini-flash-latest",
    profile="research",          # loads skills=["research"] + report gates
    workspace="./.deep_workspace",
    memory=Memory.compose(
        conversation=ConversationMemory(),
        user=UserMemory(auto_extract=True),
    ),
)
await agent.arun("Research X; write reports/x.md with citations")
```

General long-horizon (no research skill):

```python
agent = create_deep_agent(model, workspace="./.deep_workspace", profile="general")
```

## Examples

| File | What it shows |
|------|----------------|
| `01_research_brief.py` | Scripted / live `create_deep_agent` |
| `02_case_deep_hard.py` | Deep + Case accept |
| `03_live_multimodal_research.py` | `profile="research"` live loop |

```bash
python examples/deep_agent/01_research_brief.py
DEEP_AGENT_LIVE=1 GEMINI_API_KEY=... \
  python examples/deep_agent/03_live_multimodal_research.py
```
