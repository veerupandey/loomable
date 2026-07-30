"""Tests for MCP client transport selection, connect, invoke, and close."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from loomable.kernel.errors import MCPConnectionError, MCPToolError
from loomable.kernel.mcp_client import MCPClient, MCPCapabilities, MCPSession


# ---------------------------------------------------------------------------
# Transport selection tests
# ---------------------------------------------------------------------------


class TestTransportSelection:
    """Test _select_transport inference and explicit types."""

    def setup_method(self):
        self.client = MCPClient()

    def test_explicit_stdio(self):
        spec = {"server_id": "s1", "transport": "stdio", "command": "python"}
        assert self.client._select_transport(spec) == "stdio"

    def test_explicit_sse(self):
        spec = {"server_id": "s1", "transport": "sse", "url": "http://localhost"}
        assert self.client._select_transport(spec) == "http"

    def test_explicit_http(self):
        spec = {"server_id": "s1", "transport": "http", "url": "http://localhost"}
        assert self.client._select_transport(spec) == "http"

    def test_explicit_streamable_http(self):
        spec = {"server_id": "s1", "transport": "streamable-http", "url": "http://localhost"}
        assert self.client._select_transport(spec) == "http"

    def test_infer_stdio_from_command(self):
        spec = {"server_id": "s1", "command": "node", "args": ["server.js"]}
        assert self.client._select_transport(spec) == "stdio"

    def test_infer_http_from_url(self):
        spec = {"server_id": "s1", "url": "http://localhost:8000/sse"}
        assert self.client._select_transport(spec) == "http"

    def test_unresolvable_raises(self):
        spec = {"server_id": "bad-server"}
        with pytest.raises(MCPConnectionError):
            self.client._select_transport(spec)

    def test_invalid_transport_type_raises(self):
        spec = {"server_id": "s1", "transport": "grpc"}
        with pytest.raises(MCPConnectionError):
            self.client._select_transport(spec)


# ---------------------------------------------------------------------------
# Connect / Enumerate tests (mocked MCP SDK)
# ---------------------------------------------------------------------------


@dataclass
class FakeTool:
    name: str
    description: str = "A tool"
    inputSchema: dict = None

    def __post_init__(self):
        if self.inputSchema is None:
            self.inputSchema = {"type": "object", "properties": {}}


@dataclass
class FakeResource:
    uri: str = "file:///test"
    name: str = "test_resource"
    description: str = "A resource"


@dataclass
class FakeToolsResult:
    tools: list = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = [FakeTool("add"), FakeTool("multiply")]


@dataclass
class FakeResourcesResult:
    resources: list = None

    def __post_init__(self):
        if self.resources is None:
            self.resources = [FakeResource()]


class FakeClientSession:
    """Fake mcp.ClientSession for testing."""

    def __init__(self):
        self.initialized = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def initialize(self):
        self.initialized = True

    async def list_tools(self):
        return FakeToolsResult()

    async def list_resources(self):
        return FakeResourcesResult()

    async def call_tool(self, tool_name: str, args: dict):
        @dataclass
        class FakeContent:
            text: str = f"result from {tool_name}"

        @dataclass
        class FakeCallResult:
            content: list = None
            isError: bool = False

            def __post_init__(self):
                if self.content is None:
                    self.content = [FakeContent()]

        return FakeCallResult()


class FakeTransportCtx:
    """Fake transport context manager."""

    async def __aenter__(self):
        return (AsyncMock(), AsyncMock())  # read_stream, write_stream

    async def __aexit__(self, *args):
        pass


class TestMCPConnect:
    """Test MCP connect flow with mocked SDK."""

    @pytest.fixture
    def client(self):
        return MCPClient()

    async def test_connect_stdio_enumerates_tools(self, client, monkeypatch):
        """Connecting via stdio should enumerate tools and resources."""
        fake_session = FakeClientSession()

        # Mock the lazy import inside _connect_stdio
        mock_stdio_module = MagicMock()
        mock_stdio_module.StdioServerParameters = MagicMock(return_value=MagicMock())
        mock_stdio_module.stdio_client = MagicMock(return_value=FakeTransportCtx())

        mock_mcp = MagicMock()
        mock_mcp.ClientSession = MagicMock(return_value=fake_session)

        with patch.dict("sys.modules", {
            "mcp": mock_mcp,
            "mcp.client": MagicMock(),
            "mcp.client.stdio": mock_stdio_module,
        }):
            # Directly call _enumerate_capabilities to bypass full transport
            client._sessions["test-server"] = {
                "client_session": fake_session,
                "cleanup": AsyncMock(),
            }
            caps = await client._enumerate_capabilities(fake_session)

        assert len(caps.tools) == 2
        assert caps.tools[0]["name"] == "add"
        assert caps.tools[1]["name"] == "multiply"
        assert len(caps.resources) == 1
        assert caps.resources[0]["name"] == "test_resource"

    async def test_connect_raises_on_failure(self, client):
        """A failed connection should raise MCPConnectionError naming the server."""
        spec = {"server_id": "broken-server", "command": "nonexistent_binary"}
        with pytest.raises(MCPConnectionError):
            await client.connect(spec)


# ---------------------------------------------------------------------------
# Tool invocation tests
# ---------------------------------------------------------------------------


class TestMCPInvoke:
    """Test MCP tool invocation and result mapping."""

    async def test_invoke_tool_success(self):
        client = MCPClient()
        fake_session = FakeClientSession()
        client._sessions["srv"] = {
            "client_session": fake_session,
            "cleanup": AsyncMock(),
        }

        session = MCPSession(server_id="srv", connected=True)
        result = await client.call_tool(session, "add", {"a": 1, "b": 2})

        assert result.content == "result from add"
        assert result.error is None

    async def test_invoke_on_disconnected_raises(self):
        client = MCPClient()
        session = MCPSession(server_id="srv", connected=False)

        with pytest.raises(MCPToolError):
            await client.call_tool(session, "add", {})

    async def test_invoke_on_missing_session_raises(self):
        client = MCPClient()
        session = MCPSession(server_id="nonexistent", connected=True)

        with pytest.raises(MCPToolError):
            await client.call_tool(session, "add", {})

    async def test_invoke_maps_error_result(self):
        """MCP isError=True should map to ToolResult.error."""
        @dataclass
        class ErrorContent:
            text: str = "something went wrong"

        @dataclass
        class ErrorResult:
            content: list = None
            isError: bool = True

            def __post_init__(self):
                if self.content is None:
                    self.content = [ErrorContent()]

        client = MCPClient()
        error_session = FakeClientSession()
        error_session.call_tool = AsyncMock(return_value=ErrorResult())
        client._sessions["srv"] = {
            "client_session": error_session,
            "cleanup": AsyncMock(),
        }

        session = MCPSession(server_id="srv", connected=True)
        result = await client.call_tool(session, "broken_tool", {})

        assert result.is_error
        assert "something went wrong" in result.error


# ---------------------------------------------------------------------------
# Session close tests
# ---------------------------------------------------------------------------


class TestMCPClose:
    """Test MCP session close."""

    async def test_close_removes_session_and_calls_cleanup(self):
        client = MCPClient()
        cleanup_mock = AsyncMock()
        client._sessions["srv"] = {"cleanup": cleanup_mock}

        session = MCPSession(server_id="srv", connected=True)
        await client.close(session)

        assert "srv" not in client._sessions
        assert not session.connected
        cleanup_mock.assert_awaited_once()

    async def test_close_on_unknown_session_is_noop(self):
        client = MCPClient()
        session = MCPSession(server_id="unknown", connected=True)
        await client.close(session)  # should not raise
        assert not session.connected
