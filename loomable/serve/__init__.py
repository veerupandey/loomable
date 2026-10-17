"""loomable.serve - Edge transport adapters for agents and cases.

Thin request/response translators:

- ``FastAPIAdapter`` / ``mount_agent`` / ``mount_case`` (HTTP + AG-UI SSE)
- ``MCPServerAdapter`` (Model Context Protocol / agent-as-tool)"""
from __future__ import annotations

from importlib import import_module

__all__ = ['FastAPIAdapter', 'MCPServerAdapter', 'mount_agent', 'mount_case', 'mount_team', 'mount_workflow']

_EXPORTS = {
    'FastAPIAdapter': ('.fastapi_adapter', 'FastAPIAdapter'),
    'MCPServerAdapter': ('.mcp_adapter', 'MCPServerAdapter'),
    'mount_agent': ('.fastapi_adapter', 'mount_agent'),
    'mount_case': ('.fastapi_adapter', 'mount_case'),
    'mount_team': ('.fastapi_adapter', 'mount_team'),
    'mount_workflow': ('.fastapi_adapter', 'mount_workflow'),
}

def __getattr__(name):
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
