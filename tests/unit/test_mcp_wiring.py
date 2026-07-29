"""Unit tests for MCP server wiring in the Agent builder (Task 5.2).

Tests that:
- Agent(mcp_servers=[...]) connects to MCP servers via the kernel MCPClient
- Each server's tools are registered as MCPTool instances in the ToolRuntime
- MCPTool.invoke delegates to MCPClient.call_tool
- A failed connection yields MCPConnectionError for that server while others proceed
- No kernel code is modified (uses kernel MCPClient/MCPConnectionError unchanged)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from loomable.agent import Agent, BuiltAgent, MCPTool
from loomable.kernel.errors import MCPConnectionError
from loomable.kernel.mcp_client import MCPCapabilities, MCPClient, MCPSession
from loomable.kernel.models import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider:
    """Minimal ModelProvider stub."""

    async def complete(self, request):
        from loomable.kernel.models import ModelResponse

        return ModelResponse(content="ok", usage={})


class MockMCPClient(MCPClient):
    """MCPClient with controllable responses per server_id."""

    def __init__(self, server_responses: dict[str, MCPCapabilities | Exception]):
        """
        Args:
            server_responses: mapping from server_id to either MCPCapabilities
                (success) or an Exception (failure).
        """
        self._responses = server_responses
        self._tool_results: dict[str, ToolResult] = {}
        self._sessions: dict[str, MCPSession] = {}

    async def _establish_connection(self, spec: dict[str, Any]) -> MCPCapabilities:
        server_id = spec.get("server_id", spec.get("name", "unknown"))
        response = self._responses.get(server_id)
        if isinstance(response, Exception):
            raise response
        return response or MCPCapabilities()

    async def _invoke_tool(
        self, session: MCPSession, tool_name: str, args: dict[str, Any]
    ) -> ToolResult:
        return self._tool_results.get(
            tool_name, ToolResult(content=f"{tool_name} called with {args}")
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMCPWiringSuccess:
    """Tests for successful MCP server wiring."""

    def test_mcp_tools_registered_in_tool_runtime(self):
        """MCP tools from a connected server appear in the BuiltAgent's ToolRuntime."""
        caps = MCPCapabilities(
            tools=[
                {"name": "search", "description": "Search the web", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
                {"name": "compute", "description": "Run a calculation", "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}},
            ]
        )
        mock_client = MockMCPClient({"my-server": caps})

        with patch("loomable.agent.builder.MCPClient", return_value=mock_client):
            agent = Agent(
                model=FakeProvider(),
                mcp_servers=[{"server_id": "my-server"}],
            )
            built = agent.build()

        # Both tools should be registered
        assert "search" in built.tool_runtime._tools
        assert "compute" in built.tool_runtime._tools

        # They should be MCPTool instances
        assert isinstance(built.tool_runtime._tools["search"], MCPTool)
        assert isinstance(built.tool_runtime._tools["compute"], MCPTool)

    def test_mcp_tool_has_correct_metadata(self):
        """MCPTool carries correct name, description, and parameters."""
        caps = MCPCapabilities(
            tools=[
                {
                    "name": "translate",
                    "description": "Translate text",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}, "lang": {"type": "string"}},
                        "required": ["text", "lang"],
                    },
                }
            ]
        )
        mock_client = MockMCPClient({"lang-server": caps})

        with patch("loomable.agent.builder.MCPClient", return_value=mock_client):
            agent = Agent(
                model=FakeProvider(),
                mcp_servers=[{"server_id": "lang-server"}],
            )
            built = agent.build()

        tool = built.tool_runtime._tools["translate"]
        assert tool.name == "translate"
        assert tool.description == "Translate text"
        assert tool.parameters["required"] == ["text", "lang"]

    def test_mcp_tool_schema_returns_openai_format(self):
        """MCPTool.schema() returns an OpenAI-style function tool schema."""
        caps = MCPCapabilities(
            tools=[{"name": "echo", "description": "Echo back", "parameters": {"type": "object", "properties": {}}}]
        )
        mock_client = MockMCPClient({"echo-server": caps})

        with patch("loomable.agent.builder.MCPClient", return_value=mock_client):
            agent = Agent(
                model=FakeProvider(),
                mcp_servers=[{"server_id": "echo-server"}],
            )
            built = agent.build()

        tool = built.tool_runtime._tools["echo"]
        schema = tool.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert schema["function"]["description"] == "Echo back"

    @pytest.mark.asyncio
    async def test_mcp_tool_invoke_delegates_to_client(self):
        """MCPTool.invoke calls MCPClient.call_tool and returns the result."""
        caps = MCPCapabilities(
            tools=[{"name": "greet", "description": "Greet someone", "parameters": {"type": "object", "properties": {}}}]
        )
        mock_client = MockMCPClient({"greet-server": caps})
        mock_client._tool_results["greet"] = ToolResult(content="Hello, World!")

        with patch("loomable.agent.builder.MCPClient", return_value=mock_client):
            agent = Agent(
                model=FakeProvider(),
                mcp_servers=[{"server_id": "greet-server"}],
            )
            built = agent.build()

        tool = built.tool_runtime._tools["greet"]
        result = await tool.invoke({"name": "World"})
        assert result.content == "Hello, World!"

    def test_multiple_servers_all_tools_registered(self):
        """Tools from multiple MCP servers are all registered."""
        caps_a = MCPCapabilities(tools=[{"name": "tool_a", "description": "A"}])
        caps_b = MCPCapabilities(tools=[{"name": "tool_b", "description": "B"}])
        mock_client = MockMCPClient({"server-a": caps_a, "server-b": caps_b})

        with patch("loomable.agent.builder.MCPClient", return_value=mock_client):
            agent = Agent(
                model=FakeProvider(),
                mcp_servers=[
                    {"server_id": "server-a"},
                    {"server_id": "server-b"},
                ],
            )
            built = agent.build()

        assert "tool_a" in built.tool_runtime._tools
        assert "tool_b" in built.tool_runtime._tools


class TestMCPWiringIsolation:
    """Tests for MCP server connection error isolation (Req 5.3)."""

    def test_failed_server_yields_mcp_connection_error(self):
        """A failed connection is captured in mcp_errors on the BuiltAgent."""
        mock_client = MockMCPClient({"bad-server": RuntimeError("connection refused")})

        with patch("loomable.agent.builder.MCPClient", return_value=mock_client):
            agent = Agent(
                model=FakeProvider(),
                mcp_servers=[{"server_id": "bad-server"}],
            )
            built = agent.build()

        assert len(built.mcp_errors) == 1
        assert built.mcp_errors[0].server_id == "bad-server"

    def test_failed_server_does_not_block_others(self):
        """One failed MCP server does not prevent other servers from connecting."""
        caps = MCPCapabilities(tools=[{"name": "good_tool", "description": "Works"}])
        mock_client = MockMCPClient({
            "bad-server": RuntimeError("down"),
            "good-server": caps,
        })

        with patch("loomable.agent.builder.MCPClient", return_value=mock_client):
            agent = Agent(
                model=FakeProvider(),
                mcp_servers=[
                    {"server_id": "bad-server"},
                    {"server_id": "good-server"},
                ],
            )
            built = agent.build()

        # Good server's tool is registered
        assert "good_tool" in built.tool_runtime._tools
        # Bad server's error is captured
        assert len(built.mcp_errors) == 1
        assert built.mcp_errors[0].server_id == "bad-server"

    def test_no_mcp_servers_yields_empty_errors(self):
        """When no MCP servers are configured, mcp_errors is empty."""
        agent = Agent(model=FakeProvider())
        built = agent.build()
        assert built.mcp_errors == []

    def test_multiple_failures_all_captured(self):
        """Multiple failed servers each get their own MCPConnectionError."""
        mock_client = MockMCPClient({
            "server-1": RuntimeError("timeout"),
            "server-2": ValueError("auth failed"),
        })

        with patch("loomable.agent.builder.MCPClient", return_value=mock_client):
            agent = Agent(
                model=FakeProvider(),
                mcp_servers=[
                    {"server_id": "server-1"},
                    {"server_id": "server-2"},
                ],
            )
            built = agent.build()

        assert len(built.mcp_errors) == 2
        error_ids = {e.server_id for e in built.mcp_errors}
        assert error_ids == {"server-1", "server-2"}


class TestMCPWiringEdgeCases:
    """Edge case tests for MCP wiring."""

    def test_tool_without_name_is_skipped(self):
        """MCP tools missing a 'name' field are skipped."""
        caps = MCPCapabilities(
            tools=[
                {"description": "No name tool"},  # missing name
                {"name": "valid", "description": "Valid tool"},
            ]
        )
        mock_client = MockMCPClient({"server": caps})

        with patch("loomable.agent.builder.MCPClient", return_value=mock_client):
            agent = Agent(
                model=FakeProvider(),
                mcp_servers=[{"server_id": "server"}],
            )
            built = agent.build()

        assert "valid" in built.tool_runtime._tools
        assert len(built.tool_runtime._tools) == 1

    def test_tool_default_parameters(self):
        """MCP tools missing 'parameters' get a default empty object schema."""
        caps = MCPCapabilities(
            tools=[{"name": "simple", "description": "No params"}]
        )
        mock_client = MockMCPClient({"server": caps})

        with patch("loomable.agent.builder.MCPClient", return_value=mock_client):
            agent = Agent(
                model=FakeProvider(),
                mcp_servers=[{"server_id": "server"}],
            )
            built = agent.build()

        tool = built.tool_runtime._tools["simple"]
        assert tool.parameters == {"type": "object", "properties": {}}

    def test_mcp_tools_coexist_with_explicit_tools(self):
        """MCP tools and explicit tools coexist in the same ToolRuntime."""
        from loomable.agent.tools import FunctionTool

        def my_func(x: int) -> int:
            """Double x."""
            return x * 2

        explicit_tool = FunctionTool(my_func, name="double")

        caps = MCPCapabilities(
            tools=[{"name": "mcp_search", "description": "Search"}]
        )
        mock_client = MockMCPClient({"server": caps})

        with patch("loomable.agent.builder.MCPClient", return_value=mock_client):
            agent = Agent(
                model=FakeProvider(),
                tools=[explicit_tool],
                mcp_servers=[{"server_id": "server"}],
            )
            built = agent.build()

        assert "double" in built.tool_runtime._tools
        assert "mcp_search" in built.tool_runtime._tools
