"""Regression coverage using the installed SDK and a real stdio subprocess."""

import asyncio
import sys

import pytest

from loomable import Agent, tool
from loomable.providers import OpenAIProvider


def test_stdio_discovery_and_invocation_use_independent_lifecycles():
    pytest.importorskip("mcp.server.mcpserver")
    program = """
from mcp import types
from mcp.server import MCPServer

server = MCPServer("roundtrip")

def lookup() -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.ImageContent(type="image", mime_type="image/png", data="iVBORw0KGgo=")],
        structured_content={"marker": "remote"},
    )

server.add_tool(lookup)
server.run()
"""

    @tool
    def lookup() -> str:
        """Return the local marker."""
        return "local"

    agent = Agent(
        OpenAIProvider("gpt-5.4-mini"),
        tools=[lookup],
        mcp_servers=[{
            "name": "roundtrip",
            "command": sys.executable,
            "args": ["-B", "-c", program],
            "connect_timeout": 5,
        }],
    ).build()

    assert not agent.mcp_errors
    assert agent.tool_runtime._tools["lookup"] is lookup
    remote_name, = [name for name in agent.tool_runtime.names if name.startswith("mcp_")]
    remote = agent.tool_runtime._tools[remote_name]

    async def invoke():
        async with asyncio.timeout(10):
            return await remote.invoke({})

    for _ in range(2):
        result = asyncio.run(invoke())
        assert result.content == {"marker": "remote"}
        image, = result.metadata["media"]
        assert image.to_media_part().data == b"\x89PNG\r\n\x1a\n"
