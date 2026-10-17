# Deep Agent

One factory: `create_deep_agent`. Skills specialize it. Same `arun()` as `Agent`.

```python
from loomable import create_deep_agent

agent = create_deep_agent(
    model,                          # Gemini or OpenAI — see examples/_provider.py
    profile="research",             # or "code" / "general"
    workspace="./.deep_workspace",
)
result = await agent.arun("Research X; write reports/x.md with citations")
print(result.output.text())
```

`arun()` builds the agent. `agent.build()` is only if you need the `BuiltAgent`
(inspect tools, attach a checkpoint listener). You do not call it to run.

| Profile | What you get |
|---------|----------------|
| `research` | Bundled research skill, web search, URL fetch, citations, `reports/` + `register_source` gate |
| `code` | Bundled coding skill, `CodeIndex` on `repo=`, `run_python` / `run_shell` |
| `general` | Todos + workspace FS + `task` specialists. Add kits yourself (`code_exec=`, `web_search=`, …) |

Planning (`TodoTools`), local workspace files, `task` / `task_batch`, skills
(`load_skill`), discovery (`search_tools` / `activate_tool`) are on by default.
`knowledge_base=` / `retrievers=` are the same kwargs as `Agent`.

## Examples

These call a **real model**. Copy `.env.example` → `.env` (`GEMINI_API_KEY` or
`OPENAI_API_KEY`).

| File | What it does |
|------|----------------|
| `01_research.py` | Web research → cited `reports/brief.md` |
| `02_code.py` | Index a sample shop and draft a pricing fix and regression test |
| `03_sandbox.py` | Approve a known local script with `PythonTools` |
| `live_gate.py` | CI correctness gate (not a tutorial) |

```bash
python -m examples.deep_agent.01_research
python -m examples.deep_agent.02_code
python -m examples.deep_agent.03_sandbox
```

Install `.[toolkits]` for the web research example. The code demo writes to an ignored sample workspace. The subprocess sandbox example only approves its bundled script.

Override the research topic with `DEEP_RESEARCH_TOPIC='...'`.

Custom skill packages live next to the agent (`skills/research/` is a readable
copy of the bundled research skill). Load extras with `skills=["./examples/skills/weather-lookup"]`.
