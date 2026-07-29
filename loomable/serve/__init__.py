"""loomable.serve - Edge transport adapters for a built agent.

This package holds thin request/response translators that expose a
``BuiltAgent`` over a transport without embedding agent logic:

- ``FastAPIAdapter`` (HTTP / REST)
- ``MCPServerAdapter`` (Model Context Protocol / agent-as-tool)

It depends on ``loomable.agent`` and ``loomable.content``, plus the FastAPI and
MCP libraries.
"""

from .fastapi_adapter import FastAPIAdapter
from .mcp_adapter import MCPServerAdapter

__all__ = [
    "FastAPIAdapter",
    "MCPServerAdapter",
]
