# Competitive audit — Agent / planner / todos / subagents / Team

Honest comparison of Loomable (0.2β) against **Agno**, **LangGraph**, and **Semantic Kernel / Microsoft Agent Framework (MAF)**.

Legend: **Y** first-class · **P** partial / pattern · **N** absent or deprecated · **W** Loomable wedge

## Feature matrix

| Feature | Agno | LangGraph | SK/MAF | Loomable |
|---------|------|-----------|--------|----------|
| Agent `arun` → typed result | Y | Y | Y | Y |
| Built-in tool / ReAct loop | Y | Y | Y | Y |
| Named Team modes | Y coord/route/broadcast/**tasks** | P (graphs) | Y orchestrations | Y coord/route/broadcast/sequential/**tasks** |
| Nested teams as members | Y | Y subgraphs | Y | Y (`Team` in `members`) |
| Explicit Workflow / graph | Y | Y (richest) | Y MAF | Y step/parallel/branch/loop/route/verify |
| Todo / task-list planning | P Team tasks | P DIY state | Y Magentic/Harness | Y TodoTools + Team tasks + Case board |
| Kernel / opt-in Planner | N | N | N (legacy removed) | Y `planner=` + `planning_model=` + PLAN |
| `plan` tool escalation | P | P | P | Y (`plan_tool=True`, tool-loop workers) |
| Subagent delegation | Y | Y | Y | Y `subagents=` + `task`/`task_batch` |
| HITL tool confirm | Y | Y | Y | Y (`require_confirmation` + approver) |
| Mid-run interrupt + resume | Y continue_run | **Y** interrupt+Command | Y request_info | **P** Workflow confirm; Agent mid-loop interrupt weaker |
| Durable checkpoints | P | **Y** | Y | Y Workflow/Case |
| Time-travel / edit state | P | **Y** | P | P `get_state` / `update_state` / `fork_session` |
| Token + event streaming | Y | Y | Y | Y `astream` / `astream_events` |
| Session + user memory | Y | Y | Y | Y Memory.compose |
| Structured output | Y | Y | Y | Y `response_model` / `output_schema` |
| Deep agent harness | P | P | Y Harness | **W** `create_deep_agent` profiles |
| Goal board / accept rounds | P | P | Y Magentic | **W** `Case` / `mode="case"` |
| Serve / AgentOS surface | **Y** AgentOS | P Platform | Y Foundry | P `mount_agent` / `mount_case` |
| Dynamic Send map-reduce | P | **Y** Send | P | P `.map` / `task_batch` (fixed fan-out) |

## What Loomable already beats or matches

| Area | Verdict |
|------|---------|
| **Agno Team modes** | Matched: coordinate, route, broadcast (+ hard deterministic), sequential, **tasks**, nested teams |
| **Agno agent DX** | Matched: tools, memory compose, structured output, streaming, multimodal |
| **LangGraph control plane** | Strong: Workflow route/Command/reducers/verify/checkpoints/fork; still behind mid-node `interrupt()` and `Send` |
| **SK legacy planners** | Do not chase (removed upstream); Loomable uses FC + todos + PLAN + Case instead |
| **MAF Magentic / Harness** | Loomable Case + deep agent covers the same product need with a different API |

## Remaining gaps (honest)

1. **LangGraph-grade HITL** — mid-tool-loop `interrupt()` + `Command(resume=)` across process restart
2. **LangGraph `Send`** — dynamic N-way fan-out of unknown subtasks (beyond `.map` / `task_batch`)
3. **AgentOS / Foundry serve** — `mount_team` / `mount_workflow`, background resumable runs, auth/schedules
4. **Magentic progress ledger** — optional manager that replans on stall with human plan review (Case is close; not identical)

## Delegation map (avoid confusion)

| Mechanism | When to use |
|-----------|-------------|
| `Agent(subagents=[…])` | Sticky `delegate_to_*` tools on one parent |
| `Team(mode=…)` | Named orchestration modes (incl. nested Team) |
| `create_deep_agent` `task` / `task_batch` | Ephemeral specialists sharing a workspace |
| `Case` / `mode="case"` | Multi-round goal board + accept gates |
| Kernel `SubagentManager` | Flow parallel/map engines (not Agent `subagents=`) |

## Tests

- `tests/unit/test_competitive_agent_audit.py` — tasks mode, nested Team, planner wiring, plan-tool tools, PLAN `sub_results`
- `tests/unit/test_agent_gap_fixes.py` — planner / astream / Team.astream / sandbox
- `tests/unit/test_agent_features_audit.py` — delegation depth, verifier, HITL deny

Do **not** claim “we beat all three at everything.” Claim: **best unified Agent + Team + Workflow + Case + deep harness**, with Agno Team parity, LangGraph-class Workflow knobs, and Case/deep as the Magentic/Harness alternative — while remaining honest about HITL durability and Send.
