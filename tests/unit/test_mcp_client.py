"""Unit tests for the MCP Client (Task 8.3).

Tests the error handling contract:
- connect() raises MCPConnectionError naming the server on failure
- call_tool() raises MCPToolError naming the tool on failure
- Successful connect enumerates capabilities
- list_capabilities() returns discovered tools/resources
- Failed connections don't affect other sessions
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from loomable.kernel.errors import MCPConnectionError, MCPToolError
from loomable.kernel.mcp_client import MCPCapabilities, MCPClient, MCPSession
from loomable.kernel.models import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockMCPClient(MCPClient):
    """MCPClient with overridable transport methods for testing."""

    def __init__(
        self,
        capabilities: MCPCapabilities | None = None,
        connect_error: Exception | None = None,
        tool_result: ToolResult | None = None,
        tool_error: Exception | None = None,
    ):
        self._capabilities = capabilities or MCPCapabilities()
        self._connect_error = connect_error
        self._tool_result = tool_result or ToolResult(content="ok")
        self._tool_error = tool_error

    async def _establish_connection(self, spec: dict[str, Any]) -> MCPCapabilities:
        if self._connect_error:
            raise self._connect_error
        return self._capabilities

    async def _invoke_tool(
        self, session: MCPSession, tool_name: str, args: dict[str, Any]
    ) -> ToolResult:
        if self._tool_error:
            raise self._tool_error
        return self._tool_result


# ---------------------------------------------------------------------------
# connect() tests
# ---------------------------------------------------------------------------


class TestMCPClientConnect:
    """Tests for MCPClient.connect()."""

    async def test_connect_success_returns_session(self):
        """Successful connect returns an MCPSession with connected=True."""
        caps = MCPCapabilities(
            tools=[{"name": "echo", "description": "echoes input"}],
            resources=[{"name": "data", "uri": "file:///data.json"}],
        )
        client = MockMCPClient(capabilities=caps)
        spec = {"server_id": "test-server", "transport": "stdio"}

        session = await client.connect(spec)

        assert session.server_id == "test-server"
        assert session.connected is True
        assert session.capabilities.tools == caps.tools
        assert session.capabilities.resources == caps.resources

    async def test_connect_failure_raises_mcp_connection_error(self):
        """Failed connect raises MCPConnectionError naming the server."""
        client = MockMCPClient(connect_error=RuntimeError("transport failed"))
        spec = {"server_id": "failing-server", "transport": "stdio"}

        with pytest.raises(MCPConnectionError) as exc_info:
            await client.connect(spec)

        assert exc_info.value.server_id == "failing-server"
        assert "failing-server" in str(exc_info.value)

    async def test_connect_uses_name_fallback_for_server_id(self):
        """When server_id is missing, falls back to 'name' key."""
        client = MockMCPClient(connect_error=RuntimeError("boom"))
        spec = {"name": "my-mcp-server", "transport": "sse"}

        with pytest.raises(MCPConnectionError) as exc_info:
            await client.connect(spec)

        assert exc_info.value.server_id == "my-mcp-server"

    async def test_connect_uses_unknown_when_no_id_keys(self):
        """When neither server_id nor name is present, uses 'unknown'."""
        client = MockMCPClient(connect_error=RuntimeError("boom"))
        spec = {"transport": "stdio"}

        with pytest.raises(MCPConnectionError) as exc_info:
            await client.connect(spec)

        assert exc_info.value.server_id == "unknown"

    async def test_connect_wraps_arbitrary_exception(self):
        """Arbitrary exceptions are wrapped in MCPConnectionError."""
        client = MockMCPClient(connect_error=ValueError("unexpected"))
        spec = {"server_id": "svr-1"}

        with pytest.raises(MCPConnectionError) as exc_info:
            await client.connect(spec)

        assert exc_info.value.server_id == "svr-1"
        assert exc_info.value.__cause__ is not None

    async def test_connect_reraises_mcp_connection_error_directly(self):
        """MCPConnectionError from transport is re-raised without wrapping."""
        original = MCPConnectionError("inner-server")
        client = MockMCPClient(connect_error=original)
        spec = {"server_id": "inner-server"}

        with pytest.raises(MCPConnectionError) as exc_info:
            await client.connect(spec)

        assert exc_info.value is original


# ---------------------------------------------------------------------------
# list_capabilities() tests
# ---------------------------------------------------------------------------


class TestMCPClientListCapabilities:
    """Tests for MCPClient.list_capabilities()."""

    async def test_list_capabilities_returns_session_capabilities(self):
        """list_capabilities returns the capabilities from the session."""
        caps = MCPCapabilities(
            tools=[{"name": "search"}, {"name": "compute"}],
            resources=[{"name": "kb", "uri": "vector://kb"}],
        )
        client = MockMCPClient(capabilities=caps)
        spec = {"server_id": "cap-server"}

        session = await client.connect(spec)
        result = await client.list_capabilities(session)

        assert result is session.capabilities
        assert len(result.tools) == 2
        assert len(result.resources) == 1

    async def test_list_capabilities_empty_when_none_discovered(self):
        """list_capabilities returns empty lists when server has no tools."""
        client = MockMCPClient(capabilities=MCPCapabilities())
        spec = {"server_id": "empty-server"}

        session = await client.connect(spec)
        result = await client.list_capabilities(session)

        assert result.tools == []
        assert result.resources == []


# ---------------------------------------------------------------------------
# call_tool() tests
# ---------------------------------------------------------------------------


class TestMCPClientCallTool:
    """Tests for MCPClient.call_tool()."""

    async def test_call_tool_success_returns_tool_result(self):
        """Successful tool call returns a ToolResult with content."""
        expected = ToolResult(content={"answer": 42}, metadata={"latency_ms": 10})
        client = MockMCPClient(tool_result=expected)
        spec = {"server_id": "tool-server"}
        session = await client.connect(spec)

        result = await client.call_tool(session, "compute", {"x": 7, "y": 6})

        assert result.content == {"answer": 42}
        assert result.metadata == {"latency_ms": 10}
        assert result.error is None

    async def test_call_tool_failure_raises_mcp_tool_error(self):
        """Failed tool call raises MCPToolError naming the tool."""
        client = MockMCPClient(tool_error=RuntimeError("tool crashed"))
        spec = {"server_id": "tool-server"}
        session = await client.connect(spec)

        with pytest.raises(MCPToolError) as exc_info:
            await client.call_tool(session, "broken-tool", {})

        assert exc_info.value.tool_name == "broken-tool"
        assert "broken-tool" in str(exc_info.value)

    async def test_call_tool_wraps_arbitrary_exception(self):
        """Arbitrary exceptions are wrapped in MCPToolError."""
        client = MockMCPClient(tool_error=TypeError("bad args"))
        spec = {"server_id": "tool-server"}
        session = await client.connect(spec)

        with pytest.raises(MCPToolError) as exc_info:
            await client.call_tool(session, "typed-tool", {"a": 1})

        assert exc_info.value.tool_name == "typed-tool"
        assert exc_info.value.__cause__ is not None

    async def test_call_tool_reraises_mcp_tool_error_directly(self):
        """MCPToolError from transport is re-raised without double-wrapping."""
        original = MCPToolError("direct-tool")
        client = MockMCPClient(tool_error=original)
        spec = {"server_id": "tool-server"}
        session = await client.connect(spec)

        with pytest.raises(MCPToolError) as exc_info:
            await client.call_tool(session, "direct-tool", {})

        assert exc_info.value is original


# ---------------------------------------------------------------------------
# Fault isolation tests
# ---------------------------------------------------------------------------


class TestMCPClientFaultIsolation:
    """Tests for fault isolation: failed server doesn't affect others."""

    async def test_failed_connection_does_not_affect_other_servers(self):
        """A failed connect to one server doesn't prevent connecting to another."""
        # First server fails
        failing_client = MockMCPClient(connect_error=RuntimeError("down"))
        with pytest.raises(MCPConnectionError):
            await failing_client.connect({"server_id": "server-a"})

        # Second server succeeds (independent client instance)
        caps = MCPCapabilities(tools=[{"name": "alive"}])
        working_client = MockMCPClient(capabilities=caps)
        session = await working_client.connect({"server_id": "server-b"})

        assert session.connected is True
        assert session.server_id == "server-b"

    async def test_tool_error_does_not_affect_subsequent_calls(self):
        """A tool error on one call doesn't prevent subsequent calls."""

        class StatefulMockClient(MCPClient):
            def __init__(self):
                self.call_count = 0

            async def _establish_connection(self, spec):
                return MCPCapabilities(tools=[{"name": "tool-a"}])

            async def _invoke_tool(self, session, tool_name, args):
                self.call_count += 1
                if self.call_count == 1:
                    raise RuntimeError("first call fails")
                return ToolResult(content="success")

        client = StatefulMockClient()
        session = await client.connect({"server_id": "stateful"})

        # First call fails
        with pytest.raises(MCPToolError):
            await client.call_tool(session, "tool-a", {})

        # Second call succeeds
        result = await client.call_tool(session, "tool-a", {})
        assert result.content == "success"
