"""loomable.serve - Edge transport adapters for agents and cases.

Thin request/response translators:

- ``FastAPIAdapter`` / ``mount_agent`` / ``mount_case`` (HTTP + AG-UI SSE)
- ``MCPServerAdapter`` (Model Context Protocol / agent-as-tool)
"""

from .fastapi_adapter import (
    FastAPIAdapter,
    mount_agent,
    mount_case,
    mount_team,
    mount_workflow,
)
from .mcp_adapter import MCPServerAdapter

__all__ = [
    "FastAPIAdapter",
    "MCPServerAdapter",
    "mount_agent",
    "mount_case",
    "mount_team",
    "mount_workflow",
]
