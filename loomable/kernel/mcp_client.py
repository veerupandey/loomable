"""MCP Client for the loomable agent framework.

Implements the Model Context Protocol at the tools boundary (Req 4).
On connect, enumerates exposed tools and data resources.
Connection failures yield MCPConnectionError naming the server.
Tool invocation errors yield MCPToolError naming the tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loomable.kernel.errors import MCPConnectionError, MCPToolError
from loomable.kernel.models import MCPServerSpec, ToolResult


@dataclass
class MCPCapabilities:
    """Capabilities discovered from an MCP server on connect.

    Attributes:
        tools: List of tool info dicts describing available tools.
        resources: List of resource info dicts describing data resources.
    """

    tools: list[dict[str, Any]] = field(default_factory=list)
    resources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class MCPSession:
    """Represents an active session with an MCP server.

    Attributes:
        server_id: Identifier of the connected MCP server.
        capabilities: Tools and resources discovered during connect.
        connected: Whether the session is currently connected.
    """

    server_id: str
    capabilities: MCPCapabilities = field(default_factory=MCPCapabilities)
    connected: bool = False


class MCPClient:
    """Client for connecting to and invoking tools on MCP servers.

    Implements the core MCP boundary contract:
    - connect(): establishes connection and enumerates capabilities
    - list_capabilities(): returns discovered tools and resources
    - call_tool(): invokes a named tool with arguments

    Error contract:
    - Failed connection raises MCPConnectionError naming the server_id.
    - Tool invocation failure raises MCPToolError naming the tool.
    """

    async def connect(self, spec: MCPServerSpec) -> MCPSession:
        """Connect to an MCP server and enumerate its capabilities.

        Args:
            spec: Server specification dict containing at minimum a 'server_id'
                  key, plus transport/auth configuration.

        Returns:
            An MCPSession with discovered capabilities.

        Raises:
            MCPConnectionError: If the connection to the server fails,
                naming the server_id from the spec.
        """
        server_id = spec.get("server_id", spec.get("name", "unknown"))

        try:
            capabilities = await self._establish_connection(spec)
        except MCPConnectionError:
            raise
        except Exception as exc:
            raise MCPConnectionError(server_id) from exc

        session = MCPSession(
            server_id=server_id,
            capabilities=capabilities,
            connected=True,
        )
        return session

    async def list_capabilities(self, session: MCPSession) -> MCPCapabilities:
        """Return the capabilities discovered during connect.

        Args:
            session: An active MCPSession returned by connect().

        Returns:
            The MCPCapabilities (tools and resources) for this session.
        """
        return session.capabilities

    async def call_tool(
        self, session: MCPSession, tool_name: str, args: dict[str, Any]
    ) -> ToolResult:
        """Invoke a tool on the connected MCP server.

        Args:
            session: An active MCPSession returned by connect().
            tool_name: Name of the tool to invoke.
            args: Arguments to pass to the tool.

        Returns:
            A ToolResult containing the tool's response.

        Raises:
            MCPToolError: If the tool invocation fails, naming the tool.
        """
        try:
            result = await self._invoke_tool(session, tool_name, args)
        except MCPToolError:
            raise
        except Exception as exc:
            raise MCPToolError(tool_name) from exc

        return result

    # ------------------------------------------------------------------
    # Internal methods — these form the integration seam for testing.
    # In production, these would use the actual MCP protocol transport.
    # ------------------------------------------------------------------

    async def _establish_connection(
        self, spec: MCPServerSpec
    ) -> MCPCapabilities:
        """Establish transport connection and enumerate capabilities.

        This is the integration point for the actual MCP protocol.
        Subclass or mock this method for testing.

        Raises:
            Exception: On any transport or protocol failure.
        """
        # Default implementation attempts to use the spec to connect.
        # For real usage, this would open stdio/SSE transport and call
        # initialize + tools/list + resources/list per the MCP protocol.
        raise NotImplementedError(
            "MCPClient._establish_connection must be provided by a "
            "concrete transport implementation or mocked for testing."
        )

    async def _invoke_tool(
        self, session: MCPSession, tool_name: str, args: dict[str, Any]
    ) -> ToolResult:
        """Invoke a tool via the MCP protocol transport.

        This is the integration point for actual tool calls.
        Subclass or mock this method for testing.

        Raises:
            Exception: On any invocation failure.
        """
        raise NotImplementedError(
            "MCPClient._invoke_tool must be provided by a "
            "concrete transport implementation or mocked for testing."
        )
