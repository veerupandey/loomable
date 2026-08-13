# Escalation War Room — tough real-world Agent exam

**Scenario:** AcmePay (B2B UPI / settlement SaaS) has a **SEV-1** with partner bank
**BharatNova**. Settlement batches are failing. You are building the on-call
**Escalation Analyst** agent that a human war-room lead will trust.

## Toughness ladder (build in order)

| Phase | What we prove | Script |
|------|----------------|--------|
| **1a** | Agent → domain tools → unstructured brief + structured JSON | `01_tools_and_io.py` |
| **1b** | PDF / PPT / Markdown **input + output** | `02_documents.py` |
| **1c** | Image **input** + tool **image output** (multimodal) | `03_multimodal.py` |
| 2 | Memory L1 / L2 / L3 across shifts *(later)* | — |
| 3 | Unify workflow / flow / graph API + complex orchestration *(later)* | — |
| 4 | Skills + MCP + plan + multi-agent war room *(later)* | — |

## Why this example (not a toy)

- Real stakes: SLA breach, bank partner, money movement
- Mixed evidence: tickets, health checks, contracts, decks, runbooks, dashboards
- Mixed outputs: human brief + machine-readable packet + written artifacts
- Natural growth path into memory (shift handoff) and workflows (escalate → page → notify)

## Setup

```bash
export GEMINI_API_KEY="..."
export GEMINI_MODEL="gemini-flash-latest"   # optional
pip install -e ".[web,pdf,ppt]"

cd examples/escalation_war_room
python build_fixtures.py
python 01_tools_and_io.py
python 02_documents.py
python 03_multimodal.py
# or: python run_phase1.py
```

Issues found while stressing the framework are logged in `ISSUES.md`.
