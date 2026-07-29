"""Integration test placeholder for the agent-api transports.

Confirms the `loomable.serve` transport package is importable so that later
FastAPI and MCP adapter integration tests have a home. Real transport behavior
is covered by tasks 5-7.

Feature: agent-api
"""

import importlib

import pytest


@pytest.mark.integration
def test_serve_package_import() -> None:
    """The transport adapter package is importable (Req 10.1)."""
    module = importlib.import_module("loomable.serve")
    assert module is not None
