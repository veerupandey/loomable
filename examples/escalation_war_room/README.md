# Escalation War Room

Real-world SEV exam for Loomable: AcmePay settlement failure with partner bank
**BharatNova** (INC-88421). Build an on-call Escalation Analyst a war-room lead can trust.

## Ladder

| # | Script | Proves |
|---|--------|--------|
| 01 | `01_tools_and_io.py` | Domain tools → brief + structured JSON |
| 02 | `02_documents.py` | PDF / PPT / Markdown I/O |
| 03 | `03_multimodal.py` | Image input + tool image output |
| 04 | `04_workflow.py` | Fluent `Workflow` orchestration |
| 05 | `05_checkpoint_resume.py` | Kill / resume with checkpointer |
| 06 | `06_memory_compaction.py` | Agent L1/L2 `ContextPolicy` compaction |
| 07 | `07_team_spawn.py` | Hard `Team` + `spawn_specialist` |
| 08 | `08_stress_exam.py` | Full stress path |
| 09 | `09_challenge_hitl.py` | Fluent HITL (`confirm` + `approve`) |
| 10 | `10_case.py` | `Case` — plan → dispatch → accept + board |
| 11 | `11_case_sse.py` | Case AG-UI SSE + WorkItems `STATE_*` |
| 12 | `12_agent_agui_sse.py` | Agent FastAPI `text/event-stream` |

Engineering notes: `CHALLENGES.md`

## Setup

```bash
export GEMINI_API_KEY="..."
# optional: export GEMINI_MODEL="gemini-flash-latest"

pip install -e ".[web,pdf,ppt]"
cd examples/escalation_war_room
python build_fixtures.py
python 01_tools_and_io.py
# …
python 12_agent_agui_sse.py
```
