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

## Still open (to fix next)

### ISSUE-WR-001 — Empty final assistant text after tool writes (P1)
- Even with recovery for structured mode, unstructured runs often end with
  `""` after `write_file` / badge tools.
- **Wanted:** Always request a short confirmation text when the last action
  was a side-effecting tool (or expose `require_final_text=True`).

### ISSUE-WR-002 — Default `max_tool_iterations=6` is too low for real jobs (P1)
- Reading md+pdf+pptx + 2 lookups + 2 writes needs ~8–10 steps.
- **Wanted:** Raise default (e.g. 12–16) or auto-bump when many tools/docs
  are attached; document the knob prominently.

### ISSUE-WR-003 — `write_file` JSON is not validated by `response_model` (P1)
- Agent freely invents JSON shapes on disk (`P1` vs `SEV-1`, nested objects).
- **Wanted:** Optional `typed_write` / schema-checked file tool, or a
  post-write validator hook tied to `response_model`.

### ISSUE-WR-008 — No first-class PPT/PDF **write** toolkit (P2)
- Can read pptx/pdf; writing requires custom `@tool` + `python-pptx` / PDF
  bytes. Fine for now, but war-room status decks want `write_pptx`.

### ISSUE-WR-009 — FileTools sandbox is easy to misconfigure (P2)
- Pointing `base_dir` at the example package lets the model read `.py`
  sources and burn iterations.
- **Wanted:** Examples/docs should default to a dedicated workspace dir;
  consider `allow_globs=` / deny lists.

### ISSUE-WR-010 — Workflow/Flow/Graph API sprawl (P1, planned)
- User goal: unify workflow API into one smart surface, then attach complex
  workflows to Agent with step monitoring. Not started in Phase 1.

### ISSUE-WR-011 — Memory L1/L2/L3 not exercised yet (planned)
- Phase 2 of this exam: shift-handoff memory across war-room turns.

## Phase 1 results (Gemini)

| Step | Status | Notes |
|------|--------|-------|
| 1a tools + unstructured/structured I/O | PASS | 5 tools each; solid SEV packet |
| 1b md/pdf/pptx in + md/json out | PASS | Needed `max_tool_iterations=20` + sandbox |
| 1c image in + tool image out + structured | PASS | After feedback-media + empty-text fixes |

## Next toughness increments

1. **Memory:** multi-turn shift handoff (L1 session, L2 summaries, L3 notes)
2. **Unify workflow API** (collapse workflow/flow/graph)
3. **Attach complex workflow to Agent** with per-step monitoring
4. **Skills + MCP** for external ticketing / Slack bridge
