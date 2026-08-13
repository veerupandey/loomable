# Challenges — making Loomable best + easy

Findings from stress exam (`08`) and challenge (`09`) after Phases A–D.
Goal: enterprise spine that stays one-line-easy on the happy path.

## What worked under pressure

| Capability | Exam | Result |
|------------|------|--------|
| Fluent `Workflow.step` + kill/resume | 08 | Resume skipped gather; `resumed` / `skipped_nodes` correct |
| Hard `Team(mode="broadcast", hard=True)` | 08 | Deterministic parallel specialists |
| `spawn_specialist` | 08 | Ephemeral cert auditor OK |
| Multimodal vision (`modalities="text+image"`) | 08 | Dashboard glance OK |
| `response_model` + empty-final recovery | 08/09 | Structured finals recover after tool-only turns |
| Soft `Team(mode="coordinate")` | 09 | Works but tool-count varies (LLM must delegate) |
| `Workflow.step(..., confirm=True)` + `approve()` | 09 | HITL pause without dropping to `Node` / `Flow` |
| `require_tools=["write_file","write_json"]` | 08/09 | Fixes scribe skipping side-effect writes |

## Behaviors observed (optimize these)

1. **Structured output vs side-effect tools (P0 — fixed)**  
   Models often return `FinalPacket` JSON and skip `write_file` / `write_json`.  
   Instructions alone are not enough.  
   **Fix:** `Agent(require_tools=[...])` — re-nudge with tools still enabled until
   all required tools are satisfied; metadata `require_tools_nudged` /
   `require_tools_nudges` / `required_tools_missing`.

1b. **Empty final after successful write_json (P0 — fixed)**  
   After tools (especially under `require_tools` nudge), Gemini sometimes returns
   empty content; schema validation then hard-fails even though `write_json` already
   wrote a valid packet.  
   **Fix:** Recover `structured` (and fill output text) from the last successful
   `write_json` payload; metadata `structured_from_write_json=True`.

1c. **Wrong write paths still "satisfy" name-only require_tools (P0 — fixed)**  
   Model called `write_file`/`write_json` but wrote `final_packet.txt` at repo root
   instead of `output/stress_brief.md`.  
   **Fix:** Path constraints — `require_tools=["write_file:output/stress_brief.md",
   "write_json:output/final_packet.json"]` match the tool `path` argument.

2. **Soft Team flakiness (P1)**  
   `coordinate` / `route` depend on the LLM calling `delegate_to_*`. Sometimes zero
   tools. Prefer `hard=True` + `broadcast`/`sequential` for SEV war rooms; keep soft
   modes for exploratory UX.

3. **HITL was advanced-only (P1 — fixed)**  
   Pause/resume existed on `Node(require_confirmation=True)` but not on fluent Workflow.  
   **Fix:** `Step(..., confirm=True)` / `.step("scribe", agent, confirm=True)` +
   `wf.approve("scribe")` then `arun(resume=True)`.

3b. **JsonFileCheckpointer dropped approve() (P0 — fixed)**  
   `approve()` mutated an older checkpoint and re-`put` it with the same timestamp;
   `get()` (latest-by-filename) often returned the pre-approve pending file → resume
   re-paused.  
   **Fix:** `JsonFileCheckpointer.put` always refreshes `timestamp` so approvals win.

4. **Empty finals after tools (already fixed)**  
   `require_final_text` + schema re-prompt still essential on Gemini.

5. **Gather tool sprawl**  
   Doc+ops gather routinely needs 8–12 tools; default `max_tool_iterations=12` is OK
   but gather steps should set 16–18 explicitly.

## Still missing / next challenges

| Gap | Why it matters | Suggested fix |
|-----|----------------|---------------|
| `FlowPaused` not top-level export | Users import `loomable.flow.hitl` | Re-export from `loomable` |
| No `strict_require_tools` fail | Nudge can still be ignored | Optional raise / `RunResult.ok=False` |
| Soft Team observability | Hard to see which member ran | Label `sub_results` by role always |
| Compaction under live Gemini tool dumps | Phase C is scripted | Long-run war-room with huge PDF dumps |
| `write_pptx` / PDF write toolkit | Status decks are manual | First-class write tools |
| FileTools sandbox defaults | Easy to point at source tree | Safer example defaults + deny globs |
| Workflow-level `require_tools` | Only Agent-level today | Propagate from Step metadata |

## Easy API contract (lock this)

```python
from loomable import Agent, Team, Workflow, spawn_specialist, tool

agent = Agent(model=..., tools=[...], response_model=Packet, require_tools=["write_json"])
team = Team(members=[...], mode="broadcast", hard=True)
wf = (
    Workflow("sev", session_id=..., checkpointer=cp)
    .step("gather", gatherer)
    .step("scribe", scribe, confirm=True)
)
# crash-safe: await wf.arun(..., resume=True)
# HITL: except FlowPaused → await wf.approve("scribe") → resume
```

No frozensets, modality enums, or engine types on the happy path.
