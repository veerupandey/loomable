"""Placeholder tests for the new agent-api package layout.

Verifies that the three new additive packages (`loomable.content`,
`loomable.agent`, `loomable.serve`) exist and import cleanly, and that the
newly added dependencies (`fastapi`, `uvicorn`) and existing MCP support
(`mcp`) are available. Real behavior is covered by later tasks.

Feature: agent-api
"""

import importlib

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_name",
    ["loomable.content", "loomable.agent", "loomable.serve"],
)
def test_new_packages_import(module_name: str) -> None:
    """Each new package imports cleanly (Req 10.1)."""
    module = importlib.import_module(module_name)
    assert module is not None


@pytest.mark.unit
@pytest.mark.parametrize("dep", ["fastapi", "uvicorn", "httpx", "mcp"])
def test_dependencies_available(dep: str) -> None:
    """FastAPI, ASGI server, HTTP client, and MCP support are installed (Req 10.2)."""
    module = importlib.import_module(dep)
    assert module is not None
