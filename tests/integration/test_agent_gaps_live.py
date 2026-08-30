"""Live-agent tests for agent gap fixes (skip without API keys).

Run:
  python -m pytest tests/integration/test_agent_gaps_live.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from _provider import has_live_provider, make_provider  # noqa: E402

from loomable import Agent, Team, tool  # noqa: E402
from loomable.agent.deep import create_deep_agent  # noqa: E402
from loomable.agent.routing import ComplexityRouter, RunStrategy  # noqa: E402
from loomable.content import AgentInput  # noqa: E402


pytestmark = pytest.mark.skipif(
    not has_live_provider(),
    reason="needs GEMINI_API_KEY / OPENAI_API_KEY / Azure vars in .env",
)


@pytest.fixture(scope="module")
def provider():
    return make_provider()


class TestLivePlannerAndTools:
    @pytest.mark.asyncio
    async def test_tool_loop_with_live_model(self, provider, tmp_path):
        counter = {"n": 0}

        @tool
        def bump() -> str:
            """Increment and return count."""
            counter["n"] += 1
            return str(counter["n"])

        agent = Agent(
            model=provider,
            tools=[bump],
            instructions="Call bump exactly once, then answer with the number.",
            max_tool_iterations=4,
        )
        result = await agent.arun("Use bump then tell me the count.")
        assert counter["n"] >= 1
        assert result.output.text()

    @pytest.mark.asyncio
    async def test_astream_with_tools_live(self, provider):
        @tool
        def echo_word(w: str) -> str:
            """Echo a word."""
            return w.upper()

        agent = Agent(
            model=provider,
            tools=[echo_word],
            instructions="Call echo_word with 'loom' once, then say done.",
            max_tool_iterations=4,
        )
        chunks = [c async for c in agent.astream("stream test")]
        text = "".join(
            (c.delta.data or b"").decode("utf-8", errors="replace")
            for c in chunks
            if getattr(c.delta, "data", None)
        )
        assert chunks[-1].done is True
        assert text.strip()

    @pytest.mark.asyncio
    async def test_complexity_plan_path_live(self, provider):
        class ForcePlan(ComplexityRouter):
            def classify(self, agent_input: AgentInput, *, has_tools: bool) -> RunStrategy:
                return RunStrategy.PLAN

        agent = Agent(
            model=provider,
            complexity_router=ForcePlan(),
            instructions="Be very brief.",
        )
        result = await agent.arun(
            "Break down how to make tea in 2 steps, then summarize."
        )
        assert result.output.text()


class TestLiveTeamAndDeep:
    @pytest.mark.asyncio
    async def test_team_astream_live(self, provider):
        researcher = Agent(
            model=provider,
            role="Researcher",
            goal="One fact only",
            instructions="Reply in one short sentence.",
        )
        team = Team(members=[researcher], model=provider, mode="route")
        chunks = [c async for c in team.astream("What is 2+2? One sentence.")]
        assert chunks
        assert chunks[-1].done is True

    @pytest.mark.asyncio
    async def test_sandbox_profile_builds(self, provider, tmp_path):
        agent = create_deep_agent(
            model=provider,
            profile="sandbox",
            workspace=str(tmp_path / "ws"),
            web_search=False,
            url_fetch=False,
        )
        built = agent.build()
        names = set(built.tool_runtime._tools.keys())
        assert "run_python" in names or "run_shell" in names

    @pytest.mark.asyncio
    async def test_nested_delegate_budget_live(self, provider):
        leaf = Agent(
            model=provider,
            role="Leaf",
            instructions="Say 'leaf-ok' only.",
        )
        mid = Agent(
            model=provider,
            role="Mid",
            subagents=[leaf],
            instructions="Delegate to Leaf when asked.",
            max_depth=2,
            max_tool_iterations=4,
        )
        parent = Agent(
            model=provider,
            subagents=[mid],
            instructions="Delegate to Mid when asked.",
            max_depth=2,
            max_tool_iterations=4,
        )
        result = await parent.arun("Ask Mid to ask Leaf for leaf-ok.")
        assert result.output.text()
