"""loomable.serve.mcp_adapter - MCP server transport for a BuiltAgent.

:class:`MCPServerAdapter` exposes a :class:`~loomable.agent.BuiltAgent` as an MCP
server (agent-as-tool over the Model Context Protocol) without embedding any agent
logic (Req 8.6, 9.3). It is a thin translator that:

- advertises a single tool (default ``run_agent``) whose input accepts an
  :class:`~loomable.content.AgentInput` (at minimum a ``text`` string; optionally a
  structured ``messages`` array with multimodal parts) (Req 8.2),
- runs :meth:`BuiltAgent.arun` when the tool is invoked (Req 8.3),
- maps the produced :class:`~loomable.content.AgentOutput` parts to MCP content
  items — text as text content; image/video as image / embedded-resource content
  (base64 of inline ``data``) or a resource link (for a ``uri``) following MCP
  content conventions (Req 8.4),
- returns an MCP error result identifying the failure on any exception during the
  run rather than letting it propagate through the transport (Req 8.5).

The translation logic is factored into pure helpers/methods
(:meth:`output_to_mcp_content`, :meth:`run_tool`) that do not require a live MCP
client, so the mapping and error handling can be unit tested directly.

This module uses the low-level ``mcp.server.Server`` API, whose tool-call handler
accepts a coroutine returning a :class:`mcp.types.CallToolResult` directly — which
is exactly what :meth:`run_tool` produces.

Depends on ``loomable.agent`` and ``loomable.content`` plus the ``mcp`` library.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

import mcp.types as types

from loomable.content import (
    AgentInput,
    AgentOutput,
    Image,
    MediaPart,
    Message,
    Modality,
    Text,
    Video,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from loomable.agent import BuiltAgent


#: JSON schema advertised for the run tool's input (Req 8.2). At minimum a ``text``
#: string is accepted; a structured ``messages`` array allows multimodal parts.
_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "Plain-text input for the agent (a single user message).",
        },
        "messages": {
            "type": "array",
            "description": "Structured multimodal messages (overrides 'text' when present).",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "default": "user"},
                    "parts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "modality": {
                                    "type": "string",
                                    "enum": ["text", "image", "video"],
                                },
                                "media_type": {"type": "string"},
                                "text": {"type": "string"},
                                "data_base64": {"type": "string"},
                                "uri": {"type": "string"},
                            },
                            "required": ["modality"],
                        },
                    },
                },
                "required": ["parts"],
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Pure translation helpers (no live MCP client / transport required)
# ---------------------------------------------------------------------------


def _part_from_dict(part: dict[str, Any]) -> MediaPart:
    """Build a :class:`MediaPart` from a plain dict from the tool arguments.

    Raises ``ValueError`` (including :class:`~loomable.content.MediaPartError`) on an
    unknown modality or an otherwise invalid part.
    """
    modality_str = part.get("modality")
    try:
        modality = Modality(modality_str)
    except ValueError as exc:  # unknown / missing modality
        raise ValueError(f"Unknown modality '{modality_str}'.") from exc

    data_base64 = part.get("data_base64")
    data = base64.b64decode(data_base64) if data_base64 is not None else None
    uri = part.get("uri")
    media_type = part.get("media_type")

    if modality is Modality.TEXT:
        if part.get("text") is not None:
            return Text(part["text"])
        if data is not None:
            return Text(data.decode("utf-8"))
        if uri is not None:
            return MediaPart(
                modality=Modality.TEXT,
                media_type=media_type or "text/plain",
                uri=uri,
            )
        raise ValueError("Text part must provide 'text', 'data_base64', or 'uri'.")

    if modality is Modality.IMAGE:
        return Image(data=data, uri=uri, media_type=media_type or "image/png")

    # Modality.VIDEO
    return Video(data=data, uri=uri, media_type=media_type or "video/mp4")


def arguments_to_agent_input(arguments: dict[str, Any]) -> AgentInput:
    """Translate MCP tool ``arguments`` into an :class:`AgentInput` (Req 8.2).

    Accepts either a structured ``messages`` array (taking precedence) or a plain
    ``text`` string. Raises ``ValueError`` on an empty or invalid payload; the tool
    handler maps that to an MCP error result.
    """
    messages_arg = arguments.get("messages")
    if messages_arg:
        messages = [
            Message(
                role=message.get("role", "user"),
                parts=[_part_from_dict(part) for part in message.get("parts", [])],
            )
            for message in messages_arg
        ]
        return AgentInput(messages=messages)

    text = arguments.get("text")
    if isinstance(text, str) and text != "":
        return AgentInput.from_text(text)

    raise ValueError("Tool arguments must provide non-empty 'text' or 'messages'.")


def _media_part_to_content(part: MediaPart) -> types.ContentBlock:
    """Map a single :class:`MediaPart` to an MCP content item (Req 8.4).

    - TEXT → :class:`mcp.types.TextContent`.
    - IMAGE with inline ``data`` → :class:`mcp.types.ImageContent` (base64 payload).
    - VIDEO with inline ``data`` → :class:`mcp.types.EmbeddedResource` wrapping a
      blob resource (MCP has no dedicated video content block).
    - IMAGE/VIDEO referenced only by ``uri`` → :class:`mcp.types.ResourceLink`.
    """
    if part.modality is Modality.TEXT:
        text = part.data.decode("utf-8") if part.data is not None else (part.uri or "")
        return types.TextContent(type="text", text=text)

    # Media referenced by URI → a resource link (no inline payload).
    if part.data is None and part.uri is not None:
        return types.ResourceLink(
            type="resource_link",
            name=part.uri,
            uri=part.uri,
            mime_type=part.media_type,
        )

    # Inline bytes → base64.
    encoded = base64.b64encode(part.data).decode("ascii") if part.data is not None else ""

    if part.modality is Modality.IMAGE:
        return types.ImageContent(type="image", data=encoded, mime_type=part.media_type)

    # Modality.VIDEO (and any other inline media): embed as a blob resource.
    return types.EmbeddedResource(
        type="resource",
        resource=types.BlobResourceContents(
            uri=part.uri or f"resource://output.{_ext_for(part.media_type)}",
            mime_type=part.media_type,
            blob=encoded,
        ),
    )


def _ext_for(media_type: str) -> str:
    """Best-effort file extension derived from a media type (for synthetic URIs)."""
    if "/" in media_type:
        return media_type.split("/", 1)[1]
    return "bin"


class MCPServerAdapter:
    """Expose a :class:`BuiltAgent` as an MCP server (Req 8.1).

    Holds no agent logic beyond request/response translation (Req 8.6, 9.3): the
    advertised tool runs :meth:`BuiltAgent.arun` and maps the result to MCP content.
    """

    def __init__(self, agent: "BuiltAgent", tool_name: str = "run_agent") -> None:
        self._agent = agent
        self._tool_name = tool_name

    @property
    def tool_name(self) -> str:
        """The name of the advertised run tool."""
        return self._tool_name

    # ------------------------------------------------------------------
    # Pure translation (testable without a live client)
    # ------------------------------------------------------------------

    def output_to_mcp_content(self, output: AgentOutput) -> list[types.ContentBlock]:
        """Map an :class:`AgentOutput` to a list of MCP content items (Req 8.4)."""
        return [_media_part_to_content(part) for part in output.parts]

    async def run_tool(self, arguments: dict[str, Any]) -> types.CallToolResult:
        """Run the agent for ``arguments`` and return an MCP tool result.

        Builds an :class:`AgentInput` from the arguments, runs
        :meth:`BuiltAgent.arun`, and maps the :class:`AgentOutput` to MCP content
        (Req 8.3, 8.4). On any exception — argument translation or the run itself —
        returns a :class:`mcp.types.CallToolResult` with ``is_error=True`` and a
        message identifying the failure rather than propagating (Req 8.5).
        """
        try:
            agent_input = arguments_to_agent_input(arguments)
            result = await self._agent.arun(agent_input)
            content = self.output_to_mcp_content(result.output)
            return types.CallToolResult(content=content, is_error=False)
        except Exception as exc:  # noqa: BLE001 - translate any failure to an error result
            return types.CallToolResult(
                content=[
                    types.TextContent(
                        type="text",
                        text=f"{type(exc).__name__}: {exc}",
                    )
                ],
                is_error=True,
            )

    # ------------------------------------------------------------------
    # MCP server wiring
    # ------------------------------------------------------------------

    def server(self):
        """Build and return an MCP server exposing the agent as one tool.

        Uses :class:`mcp.server.MCPServer` (current SDK). The server advertises
        exactly one tool (``tool_name``) routed to :meth:`run_tool`.
        """
        from mcp.server import MCPServer

        server = MCPServer("loomable-agent")
        tool_name = self._tool_name
        run_tool = self.run_tool

        async def _agent_tool(messages: list[Any] | None = None) -> str:
            """Run the loomable agent and return its text output."""
            result = await run_tool({"messages": messages or []})
            if getattr(result, "is_error", False):
                parts = getattr(result, "content", None) or []
                texts = [getattr(p, "text", "") for p in parts]
                raise RuntimeError(" ".join(t for t in texts if t) or "agent tool failed")
            texts: list[str] = []
            for block in result.content or []:
                text = getattr(block, "text", None)
                if text:
                    texts.append(text)
            return "\n".join(texts) if texts else ""

        server.add_tool(
            _agent_tool,
            name=tool_name,
            description="Run the agent with an AgentInput and return its output.",
        )
        return server

    async def serve_stdio(self) -> None:  # pragma: no cover - thin transport wrapper
        """Run the MCP server over stdio.

        A thin wrapper around the stdio transport; not exercised by unit tests.
        """
        from mcp.server.stdio import stdio_server

        server = self.server()
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
