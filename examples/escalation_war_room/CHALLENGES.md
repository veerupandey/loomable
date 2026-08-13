# Challenges & design locks

Findings from stress (`08`) and HITL (`09`) exams, plus Case / AG-UI SSE.

## What works under pressure

| Capability | Result |
|------------|--------|
| Fluent `Workflow.step` + kill/resume | Resume skips completed nodes |
| Hard `Team(mode="broadcast", hard=True)` | Deterministic parallel specialists |
| `spawn_specialist` | Ephemeral workers OK |
| Multimodal vision | Dashboard glance OK |
| `require_tools` with path constraints | Forces side-effect writes |
| Fluent HITL `confirm=True` + `approve()` | Pause without low-level `Node` APIs |
| `Case` + WorkItems board | Plan → dispatch → accept with `STATE_*` |
| Agent / Case AG-UI SSE | CopilotKit-compatible event types |

## Easy API contract

```python
from loomable import Agent, Team, Workflow, Case, spawn_specialist, tool
from loomable.serve import mount_agent, mount_case

agent = Agent(model=..., tools=[...], require_tools=["write_json:output/packet.json"])
team = Team(members=[...], mode="broadcast", hard=True)
wf = (
    Workflow("sev", session_id=..., checkpointer=cp)
    .step("gather", gatherer)
    .step("scribe", scribe, confirm=True)
)

case = Case(
    model=...,
    goal="Close INC-88421 with SEV packet",
    board=True,
    dispatch="spawn",   # or "reuse"
    accept=my_check,
    max_rounds=3,
)

app = FastAPI()
mount_agent(app, agent, prefix="/agent")   # POST /agent/run/events
mount_case(app, case, prefix="/cases")     # POST /cases/run/events
```

Everything that runs is a `Runnable` (`arun` → `RunResult`). Workflow/Flow/Case share
`SharedState` for plan steps, maps, and board persistence. Agent / Case / Workflow
share the same AG-UI event vocabulary over SSE.

## Fixed gaps (do not re-open)

- Structured output skipping side-effect tools → `require_tools` + path constraints
- Empty final after `write_json` → recover structured from last write
- HITL only on low-level `Node` → fluent `confirm=True`
- JsonFileCheckpointer approve race → refresh timestamp on `put`
- Soft Team flakiness → prefer `hard=True` + broadcast/sequential for SEV rooms
