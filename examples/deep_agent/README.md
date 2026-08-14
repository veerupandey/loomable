# Deep Agent (loomable)

**One deep agent. Skills specialize it.** Research any topic with the bundled
``research`` skill — you do **not** need a separate research agent type.

```python
from loomable import create_deep_agent

# Research anything (science, policy, product, …)
agent = create_deep_agent(model, profile="research", workspace="./.deep_workspace")
# equivalent: create_deep_agent(model, skills=["research"], ...)
```

`create_deep_agent` is Agent, so the same RAG kwargs work:

```python
create_deep_agent(model, knowledge_base=store_or_sources, retrievers=[custom])
```

See ``examples/agents/07_knowledge_base.py`` for a live ``knowledge_base=`` demo.

| Pillar | Loomable |
|--------|----------|
| Planning | `TodoTools` |
| Filesystem | `WorkspaceTools` + token-aware offload + `delete_file` |
| Subagents | `task` / `task_batch` + named specialists |
| Context | `compact_conversation`, Memory, summarizer |
| Skills | Bundled `research` (any topic) via `loomable.skills` |
| Discovery | Schema budget: core tools + `search_*` / `load_skill` / `activate_tool` |
| Hard tasks | accept gates, Case, AG-UI / Team / Workflow |

## Quick start

```python
from loomable import create_deep_agent, Memory, ConversationMemory

agent = create_deep_agent(
    model="gemini:gemini-flash-latest",
    profile="research",          # loads skills=["research"] + report gates
    workspace="./.deep_workspace",
    memory=Memory.compose(
        conversation=ConversationMemory(),
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
| `02_progressive_discovery.py` | `search_skills` / `load_skill` / `search_tools` / `activate_tool` |
| `03_case_deep_hard.py` | Deep + Case accept |
| `04_live_multimodal_research.py` | `profile="research"` live loop |
| `05_live_gemini_gate.py` | Live Gemini correctness + schema-budget gate |
| `06_sandbox_browser.py` | Soft sandbox + bundled `browser` skill |
| `07_deep_code.py` | `profile="code"` + `CodeIndex` |

```bash
python examples/deep_agent/01_research_brief.py
DEEP_AGENT_LIVE=1 GEMINI_API_KEY=... \
  python examples/deep_agent/05_live_gemini_gate.py
```
