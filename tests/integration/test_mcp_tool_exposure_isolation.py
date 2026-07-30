# Feature: agent-ergonomics, Property 11

"""Property 11: MCP tools are exposed with isolation.

For any set of MCP servers of which an arbitrary subset fails to connect,
building the agent SHALL expose the tools of every connected server and SHALL
report an MCPConnectionError for each failed server without aborting the others.

**Validates: Requirements 5.1, 5.2, 5.3**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from loomable.agent.builder import Agent
from loomable.kernel.errors import MCPConnectionError
from loomable.kernel.mcp_client import MCPCapabilities, MCPClient, MCPSession
from loomable.kernel.models import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ModelProvider implementation (satisfies the structural protocol)."""

    async def invoke(self, request: Any) -> Any:
        from loomable.kernel.models import MediaPart, ModelResponse

        return ModelResponse(
            parts=[MediaPart(modality_type="text", data=b"ok")],
            usage={"input_tokens": 0, "output_tokens": 0},
        )


def _make_server_spec(server_id: str, tool_names: list[str] | None = None) -> dict[str, Any]:
    """Create a minimal MCP server spec dict for testing."""
    if tool_names is None:
        tool_names = [f"{server_id}_tool"]
    return {
        "server_id": server_id,
        "command": "fake-command",
        "transport": "stdio",
        "_test_tools": tool_names,  # metadata for our mock
    }


def _make_session_with_tools(server_id: str, tool_names: list[str]) -> MCPSession:
    """Create an MCPSession with the given tools enumerated."""
    tools = [
        {
            "name": name,
            "description": f"Tool {name} from server {server_id}",
            "parameters": {"type": "object", "properties": {}},
        }
        for name in tool_names
    ]
    capabilities = MCPCapabilities(tools=tools, resources=[])
    return MCPSession(server_id=server_id, capabilities=capabilities, connected=True)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestMCPToolExposureWithIsolation:
    """Integration tests for Property 11: MCP tools are exposed with isolation."""

    def test_single_server_registers_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A single connected MCP server's tools are registered in the built agent."""
        spec = _make_server_spec("server_a", ["tool_alpha", "tool_beta"])

        async def _mock_connect(self, server_spec):
            server_id = server_spec.get("server_id", "unknown")
            tool_names = server_spec.get("_test_tools", [])
            return _make_session_with_tools(server_id, tool_names)

        monkeypatch.setattr(MCPClient, "connect", _mock_connect)

        built = Agent(model=_FakeProvider(), mcp_servers=[spec]).build()

        assert "tool_alpha" in built.tool_runtime._tools
        assert "tool_beta" in built.tool_runtime._tools
        assert built.mcp_errors == []

    def test_multiple_servers_all_connected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple servers each have their tools registered."""
        specs = [
            _make_server_spec("server_a", ["alpha_tool"]),
            _make_server_spec("server_b", ["beta_tool"]),
            _make_server_spec("server_c", ["gamma_tool"]),
        ]

        async def _mock_connect(self, server_spec):
            server_id = server_spec.get("server_id", "unknown")
            tool_names = server_spec.get("_test_tools", [])
            return _make_session_with_tools(server_id, tool_names)

        monkeypatch.setattr(MCPClient, "connect", _mock_connect)

        built = Agent(model=_FakeProvider(), mcp_servers=specs).build()

        assert "alpha_tool" in built.tool_runtime._tools
        assert "beta_tool" in built.tool_runtime._tools
        assert "gamma_tool" in built.tool_runtime._tools
        assert built.mcp_errors == []

    def test_failed_server_reports_error_without_aborting_others(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed server produces an MCPConnectionError while others connect."""
        specs = [
            _make_server_spec("good_server", ["good_tool"]),
            _make_server_spec("bad_server", ["bad_tool"]),
        ]

        async def _mock_connect(self, server_spec):
            server_id = server_spec.get("server_id", "unknown")
            if server_id == "bad_server":
                raise MCPConnectionError(server_id)
            tool_names = server_spec.get("_test_tools", [])
            return _make_session_with_tools(server_id, tool_names)

        monkeypatch.setattr(MCPClient, "connect", _mock_connect)

        built = Agent(model=_FakeProvider(), mcp_servers=specs).build()

        # Good server's tools are registered
        assert "good_tool" in built.tool_runtime._tools
        # Bad server's tools are NOT registered
        assert "bad_tool" not in built.tool_runtime._tools
        # Error reported for the bad server
        assert len(built.mcp_errors) == 1
        assert built.mcp_errors[0].server_id == "bad_server"

    def test_all_servers_failed_produces_errors_and_no_mcp_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all servers fail, errors are collected and no MCP tools registered."""
        specs = [
            _make_server_spec("broken_a", ["tool_a"]),
            _make_server_spec("broken_b", ["tool_b"]),
        ]

        async def _mock_connect(self, server_spec):
            server_id = server_spec.get("server_id", "unknown")
            raise MCPConnectionError(server_id)

        monkeypatch.setattr(MCPClient, "connect", _mock_connect)

        built = Agent(model=_FakeProvider(), mcp_servers=specs).build()

        assert "tool_a" not in built.tool_runtime._tools
        assert "tool_b" not in built.tool_runtime._tools
        assert len(built.mcp_errors) == 2
        error_ids = {e.server_id for e in built.mcp_errors}
        assert "broken_a" in error_ids
        assert "broken_b" in error_ids

    def test_partial_failure_isolates_broken_and_connects_good(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mix of good and bad servers: good ones connect, bad ones produce errors."""
        specs = [
            _make_server_spec("good_one", ["tool_g1"]),
            _make_server_spec("bad_one", ["tool_b1"]),
            _make_server_spec("good_two", ["tool_g2"]),
        ]

        async def _mock_connect(self, server_spec):
            server_id = server_spec.get("server_id", "unknown")
            if server_id == "bad_one":
                raise MCPConnectionError(server_id)
            tool_names = server_spec.get("_test_tools", [])
            return _make_session_with_tools(server_id, tool_names)

        monkeypatch.setattr(MCPClient, "connect", _mock_connect)

        built = Agent(model=_FakeProvider(), mcp_servers=specs).build()

        # Good servers' tools registered
        assert "tool_g1" in built.tool_runtime._tools
        assert "tool_g2" in built.tool_runtime._tools
        # Bad server's tool not registered
        assert "tool_b1" not in built.tool_runtime._tools
        # Exactly one error for the bad server
        assert len(built.mcp_errors) == 1
        assert built.mcp_errors[0].server_id == "bad_one"
        assert isinstance(built.mcp_errors[0], MCPConnectionError)

    def test_mcp_errors_are_mcp_connection_error_instances(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each reported error is an MCPConnectionError naming the failed server."""
        specs = [_make_server_spec("will_fail", ["some_tool"])]

        async def _mock_connect(self, server_spec):
            server_id = server_spec.get("server_id", "unknown")
            raise MCPConnectionError(server_id)

        monkeypatch.setattr(MCPClient, "connect", _mock_connect)

        built = Agent(model=_FakeProvider(), mcp_servers=specs).build()

        assert len(built.mcp_errors) == 1
        err = built.mcp_errors[0]
        assert isinstance(err, MCPConnectionError)
        assert err.server_id == "will_fail"

    @pytest.mark.asyncio
    async def test_connected_mcp_tools_are_invocable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tools from connected servers are registered as invocable Tool instances."""
        specs = [_make_server_spec("invoke_server", ["invocable_tool"])]

        async def _mock_connect(self, server_spec):
            server_id = server_spec.get("server_id", "unknown")
            tool_names = server_spec.get("_test_tools", [])
            return _make_session_with_tools(server_id, tool_names)

        async def _mock_call_tool(self, session, tool_name, args):
            return ToolResult(content=f"called {tool_name} with {args}")

        monkeypatch.setattr(MCPClient, "connect", _mock_connect)
        monkeypatch.setattr(MCPClient, "call_tool", _mock_call_tool)

        built = Agent(model=_FakeProvider(), mcp_servers=specs).build()

        tool = built.tool_runtime._tools["invocable_tool"]
        assert tool.name == "invocable_tool"
        assert hasattr(tool, "invoke")

        # Invoke the tool and verify it delegates correctly
        result = await tool.invoke({"key": "value"})
        assert "called invocable_tool" in result.content
        assert "{'key': 'value'}" in result.content

    def test_no_mcp_servers_means_no_errors_or_tools(self) -> None:
        """When no MCP servers are configured, no errors or MCP tools are present."""
        built = Agent(model=_FakeProvider()).build()

        assert built.mcp_errors == []
        # No MCP tools registered (only internal tools if any)


# ---------------------------------------------------------------------------
# Property-based test (hypothesis): arbitrary subset failures
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    server_names=st.lists(
        st.from_regex(r"[a-z]{3,8}", fullmatch=True),
        min_size=1,
        max_size=6,
        unique=True,
    ),
    # A boolean mask indicating which servers fail to connect
    failure_mask=st.lists(st.booleans(), min_size=1, max_size=6),
)
def test_property_arbitrary_subset_failures(
    server_names: list[str], failure_mask: list[bool]
) -> None:
    """Property 11: For any set of MCP servers with an arbitrary subset failing,
    the agent exposes tools of connected servers and reports MCPConnectionError
    for each failed server without aborting the others.

    **Validates: Requirements 5.1, 5.2, 5.3**
    """
    # Align failure_mask to server_names length
    mask = failure_mask[: len(server_names)]
    while len(mask) < len(server_names):
        mask.append(False)

    # Create specs: each server exposes one tool named "{server_name}_tool"
    specs = [_make_server_spec(name, [f"{name}_tool"]) for name in server_names]

    # Determine which servers should fail
    failing_names = {name for name, fails in zip(server_names, mask) if fails}

    # Monkeypatch MCPClient.connect
    original_connect = MCPClient.connect

    async def _controlled_connect(self, server_spec):
        server_id = server_spec.get("server_id", "unknown")
        if server_id in failing_names:
            raise MCPConnectionError(server_id)
        tool_names = server_spec.get("_test_tools", [])
        return _make_session_with_tools(server_id, tool_names)

    MCPClient.connect = _controlled_connect
    try:
        built = Agent(model=_FakeProvider(), mcp_servers=specs).build()

        # All non-failing servers should have their tools registered
        for name, fails in zip(server_names, mask):
            tool_name = f"{name}_tool"
            if fails:
                assert tool_name not in built.tool_runtime._tools, (
                    f"Failing server '{name}' should NOT have tool '{tool_name}' registered"
                )
            else:
                assert tool_name in built.tool_runtime._tools, (
                    f"Connected server '{name}' should have tool '{tool_name}' registered"
                )

        # MCPConnectionErrors should be reported for each failing server
        error_ids = {e.server_id for e in built.mcp_errors}
        assert error_ids == failing_names, (
            f"Expected errors for {failing_names}, got {error_ids}"
        )

        # All errors are MCPConnectionError instances
        for err in built.mcp_errors:
            assert isinstance(err, MCPConnectionError)

    finally:
        MCPClient.connect = original_connect
