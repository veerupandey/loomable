"""Regression coverage using the installed SDK and a real stdio subprocess."""

import asyncio
import os
import sys

import pytest

from loomable import Agent, tool
from loomable.agent.tools import mcp_run_scope
from loomable.providers import OpenAIProvider


def test_stdio_discovery_and_invocation_use_independent_lifecycles():
    pytest.importorskip("mcp")
    program = """
from mcp import types
try:
    from mcp.server import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

server = MCPServer("roundtrip")
calls = 0

def lookup() -> types.CallToolResult:
    global calls
    calls += 1
    return types.CallToolResult(
        content=[types.ImageContent(type="image", mimeType="image/png", data="iVBORw0KGgo=")],
        structuredContent={"marker": "remote", "calls": calls},
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
            "env": {"PYTHONPATH": os.environ["PYTHONPATH"]} if "PYTHONPATH" in os.environ else None,
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
        assert result.content == {"marker": "remote", "calls": 1}
        image, = result.metadata["media"]
        assert image.to_media_part().data == b"\x89PNG\r\n\x1a\n"


    async def stateful_run():
        async with mcp_run_scope(), asyncio.timeout(10):
            first = await remote.invoke({})
            second = await remote.invoke({})
            return first.content["calls"], second.content["calls"]

    assert asyncio.run(stateful_run()) == (1, 2)
    assert asyncio.run(stateful_run()) == (1, 2)
