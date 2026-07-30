"""MCP Client for the loomable agent framework.

Implements the Model Context Protocol at the tools boundary.
On connect, enumerates exposed tools and data resources.
Connection failures yield MCPConnectionError naming the server.
Tool invocation errors yield MCPToolError naming the tool.

Transports (stdio and SSE/HTTP) import the ``mcp`` SDK lazily so callers
who configure no MCP servers never pay for the import.
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
    - close(): terminates the transport for a session

    Error contract:
    - Failed connection raises MCPConnectionError naming the server_id.
    - Tool invocation failure raises MCPToolError naming the tool.
    """

    def __init__(self) -> None:
        # Live transport contexts keyed by server_id for invoke/close.
        self._sessions: dict[str, Any] = {}

    async def connect(self, spec: MCPServerSpec) -> MCPSession:
        """Connect to an MCP server and enumerate its capabilities.

        Args:
            spec: Server specification dict containing at minimum a 'server_id'
                  or 'name' key, plus transport/auth configuration.

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

    async def close(self, session: MCPSession) -> None:
        """Close a session and terminate its underlying transport.

        Stops launched subprocesses and closes sockets. After close, the session
        is no longer connected and tool calls will fail.
        """
        live = self._sessions.pop(session.server_id, None)
        if live is not None:
            cleanup = live.get("cleanup")
            if cleanup is not None:
                try:
                    await cleanup()
                except Exception:
                    pass  # best-effort cleanup
        session.connected = False

    # ------------------------------------------------------------------
    # Transport implementation
    # ------------------------------------------------------------------

    def _select_transport(self, spec: MCPServerSpec) -> str:
        """Determine the transport type from the spec.

        Returns "stdio" or "http". Raises MCPConnectionError if unresolvable.
        """
        server_id = spec.get("server_id", spec.get("name", "unknown"))
        transport = spec.get("transport")
        if transport:
            if transport == "stdio":
                return "stdio"
            if transport in ("sse", "http", "streamable-http"):
                return "http"
            raise MCPConnectionError(server_id)

        # Infer from available fields
        if spec.get("command"):
            return "stdio"
        if spec.get("url"):
            return "http"
        raise MCPConnectionError(server_id)

    async def _establish_connection(
        self, spec: MCPServerSpec
    ) -> MCPCapabilities:
        """Establish transport connection and enumerate capabilities.

        Imports the ``mcp`` SDK lazily at the point of use so callers who
        never configure MCP servers never import it.
        """
        server_id = spec.get("server_id", spec.get("name", "unknown"))
        transport_type = self._select_transport(spec)

        if transport_type == "stdio":
            return await self._connect_stdio(spec, server_id)
        else:
            return await self._connect_http(spec, server_id)

    async def _connect_stdio(
        self, spec: MCPServerSpec, server_id: str
    ) -> MCPCapabilities:
        """Connect via stdio transport (launches subprocess)."""
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError as exc:
            raise MCPConnectionError(server_id) from exc

        command = spec["command"]
        args = spec.get("args", [])
        env = spec.get("env")

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

        try:
            # stdio_client returns an async context manager yielding (read, write) streams
            transport_ctx = stdio_client(server_params)
            streams = await transport_ctx.__aenter__()
            read_stream, write_stream = streams

            # Create and initialize the client session
            session_ctx = ClientSession(read_stream, write_stream)
            client_session = await session_ctx.__aenter__()
            await client_session.initialize()

            # Store for later invoke/close
            self._sessions[server_id] = {
                "client_session": client_session,
                "session_ctx": session_ctx,
                "transport_ctx": transport_ctx,
                "cleanup": self._make_cleanup(session_ctx, transport_ctx),
            }

            # Enumerate tools and resources
            return await self._enumerate_capabilities(client_session)

        except MCPConnectionError:
            raise
        except Exception as exc:
            raise MCPConnectionError(server_id) from exc

    async def _connect_http(
        self, spec: MCPServerSpec, server_id: str
    ) -> MCPCapabilities:
        """Connect via SSE/HTTP transport (remote server)."""
        try:
            from mcp import ClientSession
        except ImportError as exc:
            raise MCPConnectionError(server_id) from exc

        url = spec["url"]
        headers = spec.get("headers", {})

        # Try streamablehttp first, fall back to sse_client
        try:
            from mcp.client.streamable_http import streamablehttp_client

            transport_factory = streamablehttp_client
        except ImportError:
            try:
                from mcp.client.sse import sse_client

                transport_factory = sse_client
            except ImportError as exc:
                raise MCPConnectionError(server_id) from exc

        try:
            transport_ctx = transport_factory(url=url, headers=headers)
            streams = await transport_ctx.__aenter__()
            read_stream, write_stream = streams

            session_ctx = ClientSession(read_stream, write_stream)
            client_session = await session_ctx.__aenter__()
            await client_session.initialize()

            self._sessions[server_id] = {
                "client_session": client_session,
                "session_ctx": session_ctx,
                "transport_ctx": transport_ctx,
                "cleanup": self._make_cleanup(session_ctx, transport_ctx),
            }

            return await self._enumerate_capabilities(client_session)

        except MCPConnectionError:
            raise
        except Exception as exc:
            raise MCPConnectionError(server_id) from exc

    async def _enumerate_capabilities(self, client_session: Any) -> MCPCapabilities:
        """Enumerate tools and resources from the connected session."""
        tools: list[dict[str, Any]] = []
        resources: list[dict[str, Any]] = []

        try:
            tools_result = await client_session.list_tools()
            for tool in tools_result.tools:
                tools.append({
                    "name": tool.name,
                    "description": getattr(tool, "description", "") or "",
                    "parameters": (
                        tool.inputSchema if hasattr(tool, "inputSchema")
                        else getattr(tool, "input_schema", {"type": "object", "properties": {}})
                    ),
                })
        except Exception:
            pass  # tools not available

        try:
            resources_result = await client_session.list_resources()
            for resource in resources_result.resources:
                resources.append({
                    "uri": str(getattr(resource, "uri", "")),
                    "name": getattr(resource, "name", ""),
                    "description": getattr(resource, "description", ""),
                })
        except Exception:
            pass  # resources not available

        return MCPCapabilities(tools=tools, resources=resources)

    async def _invoke_tool(
        self, session: MCPSession, tool_name: str, args: dict[str, Any]
    ) -> ToolResult:
        """Invoke a tool via the MCP protocol transport."""
        if not session.connected:
            raise MCPToolError(tool_name)

        live = self._sessions.get(session.server_id)
        if live is None:
            raise MCPToolError(tool_name)

        client_session = live["client_session"]
        try:
            result = await client_session.call_tool(tool_name, args)
        except Exception as exc:
            raise MCPToolError(tool_name) from exc

        # Map MCP result to kernel ToolResult
        return self._map_call_result(result, tool_name)

    def _map_call_result(self, result: Any, tool_name: str) -> ToolResult:
        """Map an MCP CallToolResult to a kernel ToolResult."""
        # MCP results have .content (list of content blocks) and .isError
        is_error = getattr(result, "isError", False)

        # Extract text content from result
        content_parts: list[str] = []
        for block in getattr(result, "content", []):
            if hasattr(block, "text"):
                content_parts.append(block.text)
            elif hasattr(block, "data"):
                content_parts.append(str(block.data))

        text = "\n".join(content_parts) if content_parts else ""

        if is_error:
            return ToolResult(error=text or f"Tool '{tool_name}' returned an error")
        return ToolResult(content=text)

    @staticmethod
    def _make_cleanup(session_ctx: Any, transport_ctx: Any):
        """Create a cleanup coroutine for closing session and transport."""
        async def _cleanup():
            try:
                await session_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            try:
                await transport_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        return _cleanup
