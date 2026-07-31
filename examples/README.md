# Loomable Examples

Organized by complexity — start with agents, graduate to workflows, then advanced patterns.

## Getting Started

```bash
# Set your Azure OpenAI credentials
cp .env.example .env  # Edit with your keys

# Run any example
uv run python examples/01_agents/simple_agent.py
```

All examples use `AzureOpenAIProvider` which reads credentials from `.env`.

## 01_agents/ — Single Agent Patterns

| File | What it shows |
|------|---------------|
| `simple_agent.py` | Minimal 3-line agent — string in, string out |
| `think_and_plan.py` | Think scratchpad + plan tool for reasoning |
| `function_tools.py` | @tool decorator, automatic tool-use loop |
| `structured_io.py` | Pydantic input validation + structured output |
| `subagent_delegation.py` | Plan tool decomposes tasks into parallel subtasks |

## 02_workflows/ — Multi-Step Pipelines (Recommended)

The **recommended** way to build multi-agent systems. Declarative, composable, inspectable.

| File | What it shows |
|------|---------------|
| `sequential_pipeline.py` | Step + Workflow — named steps in a pipeline |
| `parallel_execution.py` | Parallel_Group — concurrent steps with merged results |
| `conditional_branching.py` | Condition — if/else branching based on state |
| `loops_and_iteration.py` | Loop with steps, end_condition, and verifiers |
| `nested_workflows.py` | Workflow inside Workflow — composing pipelines |
| `flowclass_event_driven.py` | @start/@listen/@router — class-based event-driven flows |

## 03_advanced_flows/ — Low-Level Engine Helpers

Direct access to the flow engine primitives. Use these when you need fine-grained control.

| File | What it shows |
|------|---------------|
| `sequential_helper.py` | `sequential()` — raw pipeline helper |
| `parallel_helper.py` | `parallel()` — concurrent execution helper |
| `routing.py` | `route()` — dynamic branching by classifier function |
| `coordinate.py` | `coordinate()` — workers + manager synthesis |
| `plan_and_execute.py` | `plan_and_execute()` — dynamic decomposition |
| `flow_with_loops.py` | Loop nodes inside a sequential flow |
| `nested_flows.py` | Multi-level flow composition |
| `custom_engine_hitl.py` | Tool approval hooks + safety blocking |

## 04_memory/ — Persistence and Recall

| File | What it shows |
|------|---------------|
| `agent_memory.py` | Conversational memory with compaction |
| `flow_memory.py` | TieredMemoryStore shared across flow nodes |
| `knowledge_rag.py` | Embeddings + knowledge docs for RAG recall |

## 05_integrations/ — External Systems

| File | What it shows |
|------|---------------|
| `mcp_agent.py` | MCP server tools in an agent |
| `mcp_in_flow.py` | MCP tools in a flow pipeline |
| `skills_agent.py` | Loading skill directories |
| `multimodal.py` | Text + image + document analysis |
| `parallel_tool_calls.py` | Concurrent tool dispatch |
| `toolkits.py` | Built-in FileTools, SQLTools, PythonTools |

## 06_production/ — Real-World Patterns

| File | What it shows |
|------|---------------|
| `tiered_routing.py` | Multi-model tiers with automatic fallback |
| `full_production_agent.py` | All features combined in one agent |
