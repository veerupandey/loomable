"""MCP Server Integration — Tools from external MCP servers.

USE WHEN: You want to connect your agent to tools exposed by
Model Context Protocol (MCP) servers (e.g. filesystem, database).

MCP tools appear as regular tools to the agent — no special handling needed.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

# MCP servers are specified as connection configs.
# Each server's tools are auto-discovered and registered.
#
# Example with a filesystem MCP server:
# agent = Agent(
#     model=provider,
#     role="File Assistant",
#     goal="Help manage files using MCP tools",
#     mcp_servers=[
#         {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
#     ],
# )

# Demo without an actual MCP server:
agent = Agent(
    model=provider,
    role="Assistant",
    goal="Demonstrate MCP integration pattern",
    instructions="You would have MCP tools available in production.",
)

result = asyncio.run(agent.arun("What tools do you have available?"))
print(result.output.text())
print("\nNote: In production, MCP server tools would appear in the tool list.")
