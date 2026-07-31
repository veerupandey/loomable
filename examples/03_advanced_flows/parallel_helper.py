"""10 — Parallel Flow: Concurrent Research Branches

Multiple agents run concurrently on the same input. Their results are
collected into sub_results keyed by node name.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import parallel
from loomable.providers.openai import AzureOpenAIProvider

# --- Parallel research agents (each with specialized instructions) ---

provider = AzureOpenAIProvider()

pros_agent = Agent(
    model=provider,
    instructions="List 3 pros/advantages of the topic. Be concise, one sentence each.",
)

cons_agent = Agent(
    model=provider,
    instructions="List 3 cons/disadvantages of the topic. Be concise, one sentence each.",
)

alternatives_agent = Agent(
    model=provider,
    instructions="List 3 alternatives to the topic. Be concise, one sentence each.",
)


# --- Run in parallel ---

flow = parallel(pros_agent, cons_agent, alternatives_agent, session_id="analysis")

result = asyncio.run(flow.arun("Using microservices architecture"))
print("=== Parallel Analysis Results ===\n")
for node_id, sub_result in result.sub_results.items():
    print(f"[{node_id}]:")
    print(sub_result.output.text())
    print()
