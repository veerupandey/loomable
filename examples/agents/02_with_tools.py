"""Agent with Tools — Function calling via @tool decorator.

USE WHEN: Your agent needs to interact with external systems
(APIs, databases, file systems) or perform calculations.

The @tool decorator turns plain Python functions into callable tools.
The agent's LLM decides when to call them.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool
from loomable.providers.openai import AzureOpenAIProvider


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed = set("0123456789+-*/.(). ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric expressions allowed"
    return str(eval(expression))  # noqa: S307


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city (demo stub)."""
    return f"Weather in {city}: 22°C, partly cloudy"


provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    role="Personal Assistant",
    goal="Help with calculations and weather lookups",
    instructions="Use your tools to answer questions. Be concise.",
    tools=[calculate, get_weather],
)

result = asyncio.run(agent.arun("What's 42 * 17, and what's the weather in Tokyo?"))

# Pretty-print shows output + tools used
from loomable.display import pp

pp(result)
