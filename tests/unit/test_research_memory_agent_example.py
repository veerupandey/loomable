"""Smoke tests for examples/agents/research_memory_agent.py factory."""

from __future__ import annotations

import tempfile
from pathlib import Path

from examples.agents.research_memory_agent import build_research_agent
from tests.unit.test_agent_builder import _FakeProvider
from tests.unit.test_notes import FakeEmbedder


def test_build_research_agent_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        agent = build_research_agent(
            _FakeProvider(),
            embedder=FakeEmbedder(),
            workspace=root,
            session_id="unit-smoke",
            user_id="researcher",
        )
        built = agent.build()
        assert built.memory is not None
        assert len(getattr(agent, "_subagents", None) or []) == 3
        runtime_tools = built.tool_runtime._tools if built.tool_runtime else {}
        tool_names = set(runtime_tools.keys()) if isinstance(runtime_tools, dict) else {
            getattr(t, "name", "") for t in runtime_tools
        }
        assert "web_search" in tool_names
        assert "plan" in tool_names
        assert "delegate_to_pdf_analyst" in tool_names
