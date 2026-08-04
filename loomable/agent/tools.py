"""loomable.agent.tools - Ergonomic tool definition.

The :func:`tool` decorator turns a plain Python function into a kernel
:class:`~loomable.kernel.contracts.Tool`, deriving the tool's ``name`` from the
function name, its ``description`` from the docstring, and a JSON-schema for its
arguments from the signature and type hints. Both sync and async functions are
supported. This removes the need to subclass ``Tool`` by hand.

Example
-------
    from loomable.agent import tool, Agent

    @tool
    def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    agent = Agent(model=..., tools=[add])
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, get_type_hints

from loomable.kernel.contracts import Tool
from loomable.kernel.mcp_client import MCPSession
from loomable.kernel.models import ToolResult
from loomable.media.types import _MediaBase

#: Mapping from Python types to JSON-schema type names.
_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> str:
    """Best-effort JSON-schema type name for a parameter annotation."""
    if annotation is inspect.Parameter.empty:
        return "string"
    origin = getattr(annotation, "__origin__", None)
    if origin in (list, tuple, set):
        return "array"
    if origin is dict:
        return "object"
    return _JSON_TYPES.get(annotation, "string")


def _extract_media(value: Any) -> list[_MediaBase] | None:
    """Extract media instances from a tool return value.

    - If ``value`` is a ``_MediaBase`` instance, returns ``[value]``.
    - If ``value`` is a list, filters for ``_MediaBase`` instances; returns the
      non-empty filtered list or ``None``.
    - Otherwise returns ``None`` (no media detected).
    """
    if isinstance(value, _MediaBase):
        return [value]
    if isinstance(value, list):
        media_items = [item for item in value if isinstance(item, _MediaBase)]
        return media_items if media_items else None
    return None


def _media_summary(items: list[_MediaBase]) -> str:
    """Produce a text summary describing a list of media items.

    Single item: ``"[Image: png, url=https://...]"`` or
    ``"[Image: png, filepath=chart.png]"``

    Multiple items: ``"[2 media items: Image(png), Audio(wav)]"``
    """
    def _describe_one(item: _MediaBase) -> str:
        class_name = type(item).__name__
        fmt = item.format or "unknown"
        if item.url is not None:
            return f"[{class_name}: {fmt}, url={item.url}]"
        elif item.filepath is not None:
            return f"[{class_name}: {fmt}, filepath={item.filepath}]"
        else:
            return f"[{class_name}: {fmt}, content=<bytes>]"

    if len(items) == 1:
        return _describe_one(items[0])

    parts = []
    for item in items:
        class_name = type(item).__name__
        fmt = item.format or "unknown"
        parts.append(f"{class_name}({fmt})")
    return f"[{len(items)} media items: {', '.join(parts)}]"


def _build_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON-schema ``parameters`` object from a function signature."""
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:  # pragma: no cover - unusual annotations
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        properties[name] = {"type": _json_type(annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class FunctionTool(Tool):
    """A :class:`Tool` that wraps a plain Python function (see :func:`tool`)."""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        idempotent: bool = True,
    ) -> None:
        self._func = func
        self.name = name or func.__name__
        self.description = description or (inspect.getdoc(func) or "").strip()
        self.parameters = _build_parameters_schema(func)
        self._is_async = asyncio.iscoroutinefunction(func)
        self.idempotent = idempotent

    def schema(self) -> dict[str, Any]:
        """Return an OpenAI-style function-tool schema for this tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        """Invoke the wrapped function with ``args`` and wrap the return value.

        Sync functions are executed in a worker thread so they never block the event
        loop. Any exception is captured as an error :class:`ToolResult` naming the
        tool so a single tool failure stays isolated.

        If the return value is a :class:`_MediaBase` instance (or a list containing
        them), the media items are stored in ``ToolResult.metadata["media"]`` and a
        text summary is produced for the ``content`` field. Non-media return values
        are wrapped unchanged.
        """
        try:
            if self._is_async:
                result = await self._func(**args)
            else:
                result = await asyncio.to_thread(self._func, **args)
        except Exception as exc:  # noqa: BLE001 - isolate tool failures
            return ToolResult(error=f"Tool '{self.name}' failed: {exc}")

        # Media detection
        media_items = _extract_media(result)
        if media_items is not None:
            return ToolResult(
                content=_media_summary(media_items),
                metadata={"media": media_items},
            )
        return ToolResult(content=result)


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    idempotent: bool = True,
) -> Any:
    """Decorator turning a function into a :class:`Tool` (usable with or without args).

    Usage::

        @tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two numbers.\"\"\"
            return a + b

        @tool(name="lookup", description="Look up a record")
        def fetch(id: str) -> dict: ...

        @tool(idempotent=False)
        def send_email(to: str, body: str) -> str:
            \"\"\"Send an email (side-effecting).\"\"\"
            ...

    The returned object is a :class:`FunctionTool` instance (a ``Tool``) that can be
    passed directly in ``Agent(tools=[...])``.
    """

    def wrap(fn: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(fn, name=name, description=description, idempotent=idempotent)

    if func is not None:
        return wrap(func)
    return wrap


# ---------------------------------------------------------------------------
# MCP Tool wrapper (Req 5.1, 5.2)
# ---------------------------------------------------------------------------


class MCPTool(Tool):
    """A :class:`Tool` backed by a remote MCP server tool.

    Each instance wraps a single tool enumerated from an MCP server. The
    ``invoke`` call delegates to :meth:`MCPClient.call_tool` on the active
    session.  This keeps the kernel MCPClient unchanged (Req 5.4).
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        mcp_client: Any,  # MCPClient instance (typed loosely to avoid circular)
        session: "MCPSession",
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._mcp_client = mcp_client
        self._session = session

    def schema(self) -> dict[str, Any]:
        """Return an OpenAI-style function-tool schema for this MCP tool."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        """Invoke the MCP tool via MCPClient.call_tool."""
        return await self._mcp_client.call_tool(self._session, self.name, args)
