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
import re
from hashlib import sha256
from typing import Any, Callable, get_type_hints

from pydantic import ConfigDict, create_model

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


def _argument_model(func):
    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func, include_extras=True)
    except (NameError, TypeError):
        hints = {}
    fields = {}
    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = Any
        default = ... if param.default is inspect.Parameter.empty else None
        fields[name] = (annotation, default)
    extra = "allow" if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()) else "forbid"
    return create_model(f"{func.__name__}Arguments",
                        __config__=ConfigDict(extra=extra, arbitrary_types_allowed=True), **fields)


def _build_parameters_schema(func: Callable[..., Any], model) -> dict[str, Any]:
    """Derive full schemas with the same types used for runtime validation."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.pop("additionalProperties", None)
    if not schema.get("required"):
        schema.pop("required", None)
    signature = inspect.signature(func)
    for name, prop in schema.get("properties", {}).items():
        prop.pop("title", None)
        prop.pop("default", None)
        if signature.parameters[name].annotation is inspect.Parameter.empty:
            prop.setdefault("type", "string")  # compatibility for unannotated tools
        if prop.get("items") == {}:
            prop.pop("items")
        if prop.get("additionalProperties") is True:
            prop.pop("additionalProperties")
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
        self._arguments = _argument_model(func)
        self.parameters = _build_parameters_schema(func, self._arguments)
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
            parsed = self._arguments.model_validate(args)
            validated = {name: getattr(parsed, name) for name in self._arguments.model_fields if name in args}
            validated.update(parsed.model_extra or {})
            if self._is_async:
                result = await self._func(**validated)
            else:
                result = await asyncio.to_thread(self._func, **validated)
        except Exception as exc:  # noqa: BLE001 - isolate tool failures
            return ToolResult(error=f"Tool '{self.name}' failed: {exc}")

        if isinstance(result, ToolResult):
            return result

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


def mcp_tool_name(server_id: str, remote_name: str, occupied: set[str]) -> str:
    """Keep valid unique names and qualify collisions for provider tool schemas."""
    if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", remote_name) and remote_name not in occupied:
        return remote_name
    prefix = re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp_{server_id}_{remote_name}")[:49]
    digest = sha256(f"{server_id}:{remote_name}".encode()).hexdigest()[:12]
    alias = f"{prefix}_{digest}"
    candidate = alias
    suffix = 1
    while candidate in occupied:
        tail = f"_{suffix}"
        candidate = alias[:64 - len(tail)] + tail
        suffix += 1
    return candidate


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
        remote_name: str | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._mcp_client = mcp_client
        self._session = session
        self._remote_name = remote_name or name
        spec = getattr(session, "spec", None) or {}
        self.idempotent = self._remote_name in spec.get("idempotent_tools", ())

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
        spec = getattr(self._session, "spec", None)
        if spec is None or self._session.connected:
            return await self._mcp_client.call_tool(self._session, self._remote_name, args)

        from loomable.kernel.mcp_client import MCPClient

        client = MCPClient()
        session = await client.connect({**spec, "_skip_discovery": True})
        try:
            return await client.call_tool(session, self._remote_name, args)
        finally:
            await client.close(session)
