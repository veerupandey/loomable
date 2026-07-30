"""05 — Agent with Subagent Delegation (Plan Tool)

The plan tool lets an agent decompose complex tasks dynamically:
1. Planner breaks the task into steps
2. Workers execute each step in parallel
3. Synthesizer combines results

All powered by the same underlying model.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, make_plan_tool, make_think_tool
from loomable.providers.openai import AzureOpenAIProvider

# --- Build agent with plan + think tools ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions=(
        "You are a research assistant. For complex multi-part questions, "
        "use the `plan` tool to decompose and execute the task in parallel. "
        "For simple questions, answer directly."
    ),
)
built = agent.build()

# Add the plan tool (needs reference to built agent)
plan_tool = make_plan_tool(built)
think_tool = make_think_tool()
built.tool_runtime._tools["plan"] = plan_tool
built.tool_runtime._tools["think"] = think_tool

# --- Run a complex query that triggers plan-based delegation ---

print("Running agent with plan tool (subagent delegation)...")
print("Query: Compare Python, Rust, and Go for building web APIs.\n")

result = asyncio.run(built.arun(
    "Compare Python, Rust, and Go for building web APIs. "
    "Cover performance, developer experience, and ecosystem."
))
print("Answer:")
print(result.output.text())
print(f"\nTool calls: {len(result.tool_activity)}")
