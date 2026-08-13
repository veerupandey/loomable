# Framework issues — Escalation War Room Phase 1

Found while stressing a real AcmePay SEV-1 war-room agent on Gemini
(`gemini-flash-latest`). Ordered by severity for a lightweight framework.

## Fixed in this branch

### ISSUE-WR-004 — Tool media feedback broke Gemini (P0)
- **Symptom:** After a tool returned `Image`, next model call failed with
  `Invalid content part type: image_url`.
- **Cause:** `feedback_media` appended `image_url` parts onto `role=tool`
  messages. Gemini’s OpenAI-compat API only allows text in tool messages.
- **Fix:** Inject tool media as a follow-up `user` message instead.

### ISSUE-WR-005 — Empty final text + `response_model` hard-fails (P0)
- **Symptom:** After successful tool writes, `output.text()` was `""` and
  `response_model` raised `StructuredOutputError`.
- **Cause:** Some providers finish a tool-only turn with empty content.
- **Fix:** When `output_schema` is set and final text is empty, re-prompt
  once without tools using the schema instruction.

### ISSUE-WR-006 — `max_tool_iterations` not on `Agent(...)` (P1)
- **Symptom:** Doc+write jobs need >6 tool steps; default is 6 and only
  mutable on `BuiltAgent` after `build()`.
- **Fix:** Added `Agent(max_tool_iterations=...)`.

### ISSUE-WR-007 — Gemini tool-loop thought signatures (P0, earlier)
- Already fixed: preserve `extra_content` / `thought_signature` across turns.

### ISSUE-WR-001 — Empty final assistant text after tool writes (P1)
- **Symptom:** Unstructured runs often ended with `""` after side-effect tools.
- **Fix:** `require_final_text=True` (default) re-prompts once without tools
  for a short confirmation; sets `metadata["final_text_reprompted"]=True`.

### ISSUE-WR-002 — Default `max_tool_iterations=6` too low (P1)
- **Fix:** Default raised to **12**. Override still via
  `Agent(max_tool_iterations=...)`.

### ISSUE-WR-003 — `write_file` JSON not schema-checked (P1)
- **Fix:** `FileTools.write_json` with optional `json_schema=` (Pydantic).
  Validation errors return as tool result strings (no crash). War-room scribe
  uses `FileTools(..., json_schema=EscalationPacket)`.

## Still open (to fix next)

### ISSUE-WR-012 — Scribe skips write tools when `response_model` set (P0 — fixed)
- **Symptom:** Model returns FinalPacket JSON and never calls `write_file`/`write_json`.
- **Fix:** `Agent(require_tools=["write_file", "write_json"])` nudges once with tools
  enabled; surfaces `require_tools_nudged` / `required_tools_missing` in metadata.
- **Gates:** `08_stress_exam.py`, `09_challenge_hitl.py`, `02_documents.py`.

### ISSUE-WR-013 — HITL only on low-level `Node` (P1 — fixed)
- **Symptom:** Fluent Workflow could not pause for approval without building `Node`s.
- **Fix:** `Step(..., confirm=True)` / `Workflow.step(..., confirm=True)` + `approve()`.

### ISSUE-WR-010 — Workflow/Flow/Graph API sprawl (P1 → largely fixed)
- Fluent `Workflow.step/parallel/branch/loop/map` is the preferred process API
- `checkpointer`/`session_id`/`resume=` wired; JSON checkpoints round-trip AgentOutput
- Top-level `from loomable import Agent, Team, Workflow, ContextPolicy, spawn_specialist`

### ISSUE-WR-011 — Memory L1/L2/L3 + auto-compaction (fixed)
- `ContextPolicy` unifies compaction + hard spill; Agent uses it on persist/bound
- War-room Phase C gate: `06_memory_compaction.py`

### Durability + Team (Phases B/D)
- Kill/resume gate: `05_checkpoint_resume.py`
- Hard Team modes + `spawn_specialist`: `07_team_spawn.py`

### ISSUE-WR-008 — No first-class PPT/PDF **write** toolkit (P2)
- Can read pptx/pdf; writing requires custom `@tool` + `python-pptx` / PDF
  bytes. Fine for now, but war-room status decks want `write_pptx`.

### ISSUE-WR-009 — FileTools sandbox is easy to misconfigure (P2)
- Pointing `base_dir` at the example package lets the model read `.py`
  sources and burn iterations.
- **Wanted:** Examples/docs should default to a dedicated workspace dir;
  consider `allow_globs=` / deny lists.

### ISSUE-WR-014 — Soft Team coordinate flaky under stress (P2)
- LLM may finish without calling `delegate_to_*`. Prefer hard broadcast/sequential
  for production SEV paths; see `CHALLENGES.md`.

## Phase 1 results (Gemini)

| Step | Status | Notes |
|------|--------|-------|
| 1a tools + unstructured/structured I/O | PASS | 5 tools each; solid SEV packet |
| 1b md/pdf/pptx in + md/json out | PASS | Needed `max_tool_iterations` bump + sandbox |
| 1c image in + tool image out + structured | PASS | After feedback-media + empty-text fixes |

## Next toughness increments

1. **Memory:** multi-turn shift handoff (L1 session, L2 summaries, L3 notes)
2. **Unify workflow API** (collapse workflow/flow/graph)
3. **Attach complex workflow to Agent** with per-step monitoring
4. **Skills + MCP** for external ticketing / Slack bridge
