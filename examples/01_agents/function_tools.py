"""03 — Agent with Function Tools

Demonstrates the @tool decorator for turning plain functions into agent tools.
The agent automatically enters a tool-use loop: model calls tools, results feed
back, until the model produces a final answer.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool
from loomable.providers.openai import AzureOpenAIProvider


# --- Define tools with the @tool decorator ---


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@tool
def power(base: int, exponent: int) -> int:
    """Raise base to the power of exponent."""
    return base ** exponent


@tool
async def fetch_exchange_rate(currency: str) -> float:
    """Get the exchange rate for a currency to USD (simulated)."""
    rates = {"EUR": 1.08, "GBP": 1.27, "JPY": 0.0067, "CAD": 0.74}
    return rates.get(currency.upper(), 1.0)


# --- Build and run ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions="You are a calculator assistant. Use the provided tools to compute answers. Show your work.",
    tools=[add, multiply, power, fetch_exchange_rate],
)

result = asyncio.run(agent.arun("What is (5 + 3) * 2^4? Show the steps."))
print("Answer:", result.output.text())
print(f"\nTools executed: {len(result.tool_activity)}")
for activity in result.tool_activity:
    print(f"  - {activity.result.content}")
