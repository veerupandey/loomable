"""02 — Agent with Think and Plan tools

Demonstrates the built-in reasoning tools:
- `think`: a scratchpad that echoes thoughts back into context (no side effects)
- `plan`: escalates to a plan→map→synthesize flow dynamically

The model can use `think` to reason step-by-step before answering.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, make_think_tool, make_plan_tool
from loomable.providers.openai import AzureOpenAIProvider

# --- Build agent with think tool ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions=(
        "You are a math tutor. When solving problems, use the `think` tool to "
        "reason step by step before giving your final answer."
    ),
    tools=[make_think_tool()],
)

result = asyncio.run(agent.arun("If a train travels 120km in 1.5 hours, what is its speed in m/s?"))
print("Answer:", result.output.text())
print(f"\nTool calls made: {len(result.tool_activity)}")
for activity in result.tool_activity:
    print(f"  - think: {activity.result.content[:80]}...")
