"""MCP Client for the loomable agent framework.

Implements the Model Context Protocol at the tools boundary.
On connect, enumerates exposed tools and data resources.
Connection failures yield MCPConnectionError naming the server.
Tool invocation errors yield MCPToolError naming the tool.

Transports (stdio and SSE/HTTP) import the ``mcp`` SDK lazily so callers
who configure no MCP servers never pay for the import.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import AsyncExitStack
from inspect import signature
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
    spec: MCPServerSpec | None = field(default=None, repr=False)


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
        if server_id in getattr(self, "_sessions", {}):
            raise MCPConnectionError(server_id)

        try:
            async with asyncio.timeout(spec.get("connect_timeout", 15)):
                capabilities = await self._establish_connection(spec)
        except MCPConnectionError:
            raise
        except Exception as exc:
            raise MCPConnectionError(server_id) from exc

        session = MCPSession(
            server_id=server_id,
            capabilities=capabilities,
            connected=True,
            spec=dict(spec) if server_id in getattr(self, "_sessions", {}) else None,
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

        async with AsyncExitStack() as stack:
            streams = await stack.enter_async_context(stdio_client(server_params))
            return await self._initialize(stack, streams, ClientSession, server_id, spec)

    async def _connect_http(
        self, spec: MCPServerSpec, server_id: str
    ) -> MCPCapabilities:
        """Connect to the selected remote transport."""
        from mcp import ClientSession

        url = spec["url"]
        headers = spec.get("headers") or {}
        async with AsyncExitStack() as stack:
            if spec.get("transport") == "sse":
                from mcp.client.sse import sse_client

                transport = sse_client(url=url, headers=headers)
            else:
                try:
                    from mcp.client.streamable_http import streamable_http_client
                except ImportError:
                    from mcp.client.streamable_http import streamablehttp_client

                    transport = streamablehttp_client(url=url, headers=headers)
                else:
                    from mcp.client.streamable_http import create_mcp_http_client

                    if headers:
                        http = await stack.enter_async_context(create_mcp_http_client(headers=headers))
                        transport = streamable_http_client(url, http_client=http)
                    else:
                        transport = streamable_http_client(url)
            streams = await stack.enter_async_context(transport)
            return await self._initialize(stack, streams, ClientSession, server_id, spec)

    async def _initialize(self, stack, streams, session_factory, server_id, spec):
        client_session = await stack.enter_async_context(session_factory(*streams[:2]))
        initialized = await client_session.initialize()
        capabilities = (MCPCapabilities() if spec.get("_skip_discovery")
                        else await self._enumerate_capabilities(client_session, getattr(initialized, "capabilities", None)))
        self._sessions[server_id] = {
            "client_session": client_session,
            "cleanup": stack.pop_all().aclose,
        }
        return capabilities

    @staticmethod
    async def _list_all(method, field: str) -> list[Any]:
        items: list[Any] = []
        seen: set[str] = set()
        page = await method()
        while True:
            items.extend(getattr(page, field))
            cursor = getattr(page, "next_cursor", None) or getattr(page, "nextCursor", None)
            if not cursor:
                return items
            if cursor in seen:
                raise ValueError(f"MCP server repeated a {field} pagination cursor")
            seen.add(cursor)
            if "params" in signature(method).parameters:
                from mcp.types import PaginatedRequestParams

                page = await method(params=PaginatedRequestParams(cursor=cursor))
            else:
                page = await method(cursor=cursor)

    async def _enumerate_capabilities(self, client_session: Any, advertised=None) -> MCPCapabilities:
        """Enumerate all pages for capabilities the server advertises."""
        tools = []
        if advertised is None or getattr(advertised, "tools", None) is not None:
            for tool in await self._list_all(client_session.list_tools, "tools"):
                tools.append({
                    "name": tool.name,
                    "description": getattr(tool, "description", "") or "",
                    "parameters": getattr(tool, "input_schema", None)
                    or getattr(tool, "inputSchema", None)
                    or {"type": "object", "properties": {}},
                })
        resources = []
        if advertised is None or getattr(advertised, "resources", None) is not None:
            for resource in await self._list_all(client_session.list_resources, "resources"):
                resources.append({
                    "uri": str(getattr(resource, "uri", "")),
                    "name": getattr(resource, "name", ""),
                    "description": getattr(resource, "description", ""),
                })
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
        """Preserve text, structured data and typed artifacts."""
        from loomable.media import Audio, File, Image, Video

        text_parts = []
        media = []
        for block in getattr(result, "content", []):
            kind = getattr(block, "type", None)
            if kind in {"image", "audio"}:
                cls = Image if kind == "image" else Audio
                mime = getattr(block, "mime_type", None) or block.mimeType
                media.append(cls(content=base64.b64decode(block.data, validate=True), mime_type=mime))
            elif kind in {"resource", "resource_link"}:
                resource = block.resource if kind == "resource" else block
                mime = getattr(resource, "mime_type", None) or getattr(resource, "mimeType", None)
                blob = getattr(resource, "blob", None)
                if blob is not None:
                    cls = {"image": Image, "audio": Audio, "video": Video}.get(
                        (mime or "").partition("/")[0], File
                    )
                    media.append(cls(
                        content=base64.b64decode(blob, validate=True),
                        mime_type=mime or "application/octet-stream",
                    ))
                elif getattr(resource, "text", None) is not None:
                    text_parts.append(resource.text)
                else:
                    # Keep references as metadata and text; do not fetch arbitrary URLs.
                    text_parts.append(str(resource.uri))
            elif hasattr(block, "text"):
                text_parts.append(block.text)

        text = "\n".join(text_parts)
        if getattr(result, "is_error", False) or getattr(result, "isError", False):
            return ToolResult(error=text or f"Tool '{tool_name}' returned an error")

        structured = getattr(result, "structured_content", None)
        if structured is None:
            structured = getattr(result, "structuredContent", None)
        metadata = {"media": media} if media else {}
        if structured is not None:
            metadata["structured_content"] = structured
            if text:
                metadata["text"] = text
        links = [str(block.uri) for block in getattr(result, "content", [])
                 if getattr(block, "type", None) == "resource_link"]
        if links:
            metadata["resource_links"] = links
        content = structured if structured is not None else text
        if not text and structured is None and media:
            content = f"[MCP tool returned {len(media)} media artifact(s)]"
        return ToolResult(content=content, metadata=metadata)
