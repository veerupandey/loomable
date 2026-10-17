"""Call a tool exposed by a real local MCP server over stdio."""

import asyncio
import sys

from examples._provider import make_provider
from loomable import Agent


def serve():
    from mcp.server import MCPServer

    server = MCPServer("shipping")

    def shipping_policy() -> str:
        """Read the shipping policy from the example's local handbook."""
        return "Orders placed before 14:00 ship on the same business day."

    server.add_tool(shipping_policy)
    server.run()


async def main():
    async with make_provider() as model:
        agent = Agent(
            model, mcp_servers=[{
                "name": "shipping",
                "command": sys.executable,
                "args": ["-m", "examples.advanced.01_mcp_servers", "--server"],
            }],
            require_tools=["shipping_policy"], strict_require_tools=True,
            instructions="Use the shipping policy tool before answering.",
        )
        print((await agent.arun("Will an order placed at 13:00 ship today?")).text)


if __name__ == "__main__":
    if sys.argv[1:] == ["--server"]:
        serve()
    else:
        asyncio.run(main())
