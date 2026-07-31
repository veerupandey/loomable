"""19 — Agent with MCP Server Tools

Demonstrates connecting an agent to MCP (Model Context Protocol) servers.
MCP tools appear as regular tools in the agent's tool runtime.

In production, MCP servers are real processes. This example shows the
configuration pattern and uses a local tool to simulate MCP behavior.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool
from loomable.providers.openai import AzureOpenAIProvider


# --- Simulated MCP-style tools (in production, these come from mcp_servers= config) ---


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city (simulates an MCP weather server)."""
    weather_data = {
        "san francisco": "62°F, sunny, wind 12mph W",
        "new york": "45°F, cloudy, wind 8mph NE",
        "london": "52°F, rainy, wind 15mph SW",
        "tokyo": "68°F, partly cloudy, wind 5mph E",
    }
    return weather_data.get(city.lower(), f"Weather data unavailable for {city}")


@tool
def search_docs(query: str) -> str:
    """Search documentation (simulates an MCP docs server)."""
    docs = {
        "authentication": "Auth uses JWT tokens with 1-hour expiry. Refresh tokens last 7 days.",
        "rate limiting": "100 requests/minute per API key. Burst allowance: 20 extra.",
        "database": "PostgreSQL 15 with read replicas. Connection pool: 20 per instance.",
    }
    for key, val in docs.items():
        if key in query.lower():
            return val
    return f"No documentation found for: {query}"


# --- Build agent with MCP-style tools ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions=(
        "You are a helpful assistant with access to weather data and documentation. "
        "Use available tools to answer questions accurately."
    ),
    tools=[get_weather, search_docs],
    # In production with real MCP servers:
    # mcp_servers=[{
    #     "server_id": "weather-service",
    #     "command": "uvx weather-mcp-server",
    #     "transport": "stdio",
    # }, {
    #     "server_id": "docs-service",
    #     "command": "uvx docs-mcp-server",
    #     "transport": "stdio",
    # }]
)

# --- Run ---

result = asyncio.run(agent.arun("What's the weather in San Francisco and what's our rate limiting policy?"))
print("Answer:", result.output.text())
print(f"\nTools used: {len(result.tool_activity)}")
for activity in result.tool_activity:
    print(f"  - {activity.result.content}")
