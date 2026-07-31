# Loomable Examples

End-to-end examples covering every feature of the loomable framework, organized
by tier and complexity.

## Agents (Tier 1)

| # | File | What it shows |
|---|------|---------------|
| 01 | `01_simple_agent.py` | Minimal 3-line agent — string in, string out |
| 02 | `02_agent_think_plan.py` | Agent with `think` scratchpad and `plan` escalation tools |
| 03 | `03_agent_with_tools.py` | Function tools via `@tool`, automatic tool-use loop |
| 04 | `04_agent_structured_io.py` | Pydantic input_schema + structured output_schema |
| 05 | `05_agent_subagents.py` | Delegating subtasks to child agents via the plan tool |
| 25 | `25_tough_auto_plan_demo.py` | Tough multi-part task: router → PLAN → dynamic worker fan-out |

## Loops (Tier 2)

| # | File | What it shows |
|---|------|---------------|
| 06 | `06_simple_loop.py` | Basic Loop with a verifier — retry until correct |
| 07 | `07_loop_with_tools.py` | Loop where the body agent uses tools each iteration |
| 08 | `08_loop_subagent_delegation.py` | Loop that spawns subagents per iteration |

## Flows (Tier 3)

| # | File | What it shows |
|---|------|---------------|
| 09 | `09_sequential_flow.py` | Sequential pipeline: research → draft → edit |
| 10 | `10_parallel_flow.py` | Parallel branches merging results |
| 11 | `11_route_flow.py` | Router node directing to different handlers |
| 12 | `12_coordinate_flow.py` | Hierarchical: workers + manager synthesis |
| 13 | `13_plan_and_execute_flow.py` | Dynamic plan → map → synthesize |
| 14 | `14_complex_flow_with_loops.py` | Flow nodes that are themselves Loops |
| 15 | `15_nested_flow_subagents.py` | Multi-level flows with agent nodes |

## Memory

| # | File | What it shows |
|---|------|---------------|
| 16 | `16_agent_memory.py` | Conversational memory + compaction across runs |
| 17 | `17_flow_memory.py` | TieredMemoryStore shared across flow nodes |
| 18 | `18_knowledge_rag.py` | Embedder + knowledge docs for RAG recall |

## MCP & Skills

| # | File | What it shows |
|---|------|---------------|
| 19 | `19_mcp_agent.py` | Agent with MCP server tools |
| 20 | `20_mcp_in_flow.py` | MCP tools available inside a flow |
| 21 | `21_skills_agent.py` | Loading skill directories |

## Advanced

| # | File | What it shows |
|---|------|---------------|
| 22 | `22_tiered_routing.py` | Model tier selection + fallback |
| 23 | `23_hitl_approval.py` | Human-in-the-loop pause + resume |
| 24 | `24_full_production_agent.py` | All features combined: tools, memory, routing, verification |

---

## Running

```bash
# Set your provider API key
export OPENAI_API_KEY=sk-...

# Run any example
uv run python examples/01_simple_agent.py
```

All examples use a `FakeProvider` by default so they run without an API key.
To use a real provider, swap `FakeProvider` for `OpenAIProvider` (commented in each file).
