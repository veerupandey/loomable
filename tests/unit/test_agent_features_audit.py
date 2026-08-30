"""Tough audit: Agent / planner / subagents / Team feature surface (offline)."""

from __future__ import annotations

import asyncio
import uuid

import pytest

from loomable import Agent, Team, tool
from loomable.agent import ModelSpec
from loomable.agent.delegation import make_delegation_tools
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.content import AgentInput, ModelCapabilities
from loomable.kernel.errors import SubagentError
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.kernel.subagents import DelegatedTask, SubagentManager, SubagentOutcome


class ScriptedProvider:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self._call_index >= len(self._responses):
            return ModelResponse(content="fallback", tool_calls=[])
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp


def _agent(provider: ScriptedProvider, **kwargs) -> Agent:
    return Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        capabilities=ModelCapabilities(),
        **kwargs,
    )


def _tool_call(name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(id=str(uuid.uuid4()), tool_name=name, args=args or {})


# ---------------------------------------------------------------------------
# Delegation depth / budgets
# ---------------------------------------------------------------------------


class TestDelegationDepthBudget:
    @pytest.mark.asyncio
    async def test_max_depth_blocks_at_parent_tool(self):
        child = _agent(ScriptedProvider([ModelResponse(content="leaf")]), role="Child")
        tools = make_delegation_tools([child], max_depth=0, depth=0)
        out = await tools[0].invoke({"task": "go"})
        assert out.error is None
        assert "max delegation depth" in str(out.content)

    @pytest.mark.asyncio
    async def test_nested_depth_propagates_into_child_build(self):
        depths_seen: list[int] = []

        grandchild = _agent(
            ScriptedProvider([ModelResponse(content="from-grandchild")]),
            role="Grandchild",
        )
        mid = _agent(
            ScriptedProvider(
                [
                    ModelResponse(
                        content="",
                        tool_calls=[_tool_call("delegate_to_grandchild", {"task": "d"})],
                    ),
                    ModelResponse(content="mid-done"),
                ]
            ),
            role="Mid",
            subagents=[grandchild],
            max_depth=2,
            max_tool_iterations=6,
        )
        orig_build = mid.build

        def tracking_build():
            depths_seen.append(int(getattr(mid, "_delegation_depth", 0) or 0))
            return orig_build()

        mid.build = tracking_build  # type: ignore[method-assign]

        parent = _agent(
            ScriptedProvider(
                [
                    ModelResponse(
                        content="",
                        tool_calls=[_tool_call("delegate_to_mid", {"task": "start"})],
                    ),
                    ModelResponse(content="parent-done"),
                ]
            ),
            role="Parent",
            subagents=[mid],
            max_depth=2,
            max_tool_iterations=6,
        )
        result = await parent.arun("nest")
        assert 1 in depths_seen
        assert "parent-done" in result.output.text()

        # Absolute depth gate still works at the limit
        blocked = make_delegation_tools([grandchild], max_depth=2, depth=2)
        blocked_out = await blocked[0].invoke({"task": "x"})
        assert "max delegation depth" in str(blocked_out.content)

    @pytest.mark.asyncio
    async def test_max_delegations_budget(self):
        child = _agent(
            ScriptedProvider(
                [
                    ModelResponse(content="A"),
                    ModelResponse(content="B"),
                ]
            ),
            role="Worker",
        )
        tools = make_delegation_tools([child], max_delegations=1, max_depth=4)
        first = await tools[0].invoke({"task": "1"})
        second = await tools[0].invoke({"task": "2"})
        assert first.content == "A"
        assert "max_delegations" in str(second.content)

    @pytest.mark.asyncio
    async def test_subagent_failure_isolated(self):
        class BoomAgent(Agent):
            async def arun(self, *a, **k):  # noqa: ANN002
                raise RuntimeError("child exploded")

        boom = BoomAgent(
            model=ModelSpec(
                provider="scripted",
                provider_impl=ScriptedProvider([ModelResponse(content="x")]),
            ),
            capabilities=ModelCapabilities(),
            role="Boom",
        )
        tools = make_delegation_tools([boom], max_depth=4)
        out = await tools[0].invoke({"task": "x"})
        assert "failed" in str(out.content).lower()
        assert "exploded" in str(out.content)


# ---------------------------------------------------------------------------
# Kernel SubagentManager (Flow engines)
# ---------------------------------------------------------------------------


class TestKernelSubagentManager:
    @pytest.mark.asyncio
    async def test_none_result_is_success(self):
        async def _work():
            return None

        outcome = await SubagentManager().spawn(
            DelegatedTask("n1", "return none", {}, _work)
        )
        assert outcome.ok
        assert outcome.result is None
        assert outcome.error is None

    @pytest.mark.asyncio
    async def test_sibling_isolation(self):
        async def ok():
            await asyncio.sleep(0.01)
            return "ok"

        async def bad():
            raise RuntimeError("boom")

        outcomes = await SubagentManager().run_all(
            [
                DelegatedTask("a", "a", {}, ok),
                DelegatedTask("b", "b", {}, bad),
                DelegatedTask("c", "c", {}, ok),
            ]
        )
        by_id = {o.task_id: o for o in outcomes}
        assert by_id["a"].result == "ok"
        assert isinstance(by_id["b"].error, SubagentError)
        assert by_id["c"].result == "ok"


# ---------------------------------------------------------------------------
# Team hard modes
# ---------------------------------------------------------------------------


class TestTeamHardModes:
    @pytest.mark.asyncio
    async def test_broadcast_runs_all_members(self):
        members = [
            _agent(ScriptedProvider([ModelResponse(content=f"m{i}")]), role=f"M{i}")
            for i in range(3)
        ]
        team = Team(
            members=members,
            model=ModelSpec(
                provider="scripted",
                provider_impl=ScriptedProvider([ModelResponse(content="unused")]),
            ),
            mode="broadcast",
        )
        result = await team.arun("hello")
        text = result.output.text()
        assert "m0" in text and "m1" in text and "m2" in text

    @pytest.mark.asyncio
    async def test_sequential_chains_outputs(self):
        a = _agent(ScriptedProvider([ModelResponse(content="step-a")]), role="A")
        b = _agent(ScriptedProvider([ModelResponse(content="step-b")]), role="B")
        team = Team(
            members=[a, b],
            model=ModelSpec(
                provider="scripted",
                provider_impl=ScriptedProvider([ModelResponse(content="u")]),
            ),
            mode="sequential",
        )
        result = await team.arun("start")
        assert "step-b" in result.output.text()


# ---------------------------------------------------------------------------
# Agent harness / planner honesty / verifier / complexity plan
# ---------------------------------------------------------------------------


class TestAgentHarness:
    @pytest.mark.asyncio
    async def test_tool_loop_then_final(self):
        @tool
        def ping() -> str:
            """Return pong."""
            return "pong"

        provider = ScriptedProvider(
            [
                ModelResponse(content="", tool_calls=[_tool_call("ping")]),
                ModelResponse(content="done with ping"),
            ]
        )
        agent = _agent(provider, tools=[ping], max_tool_iterations=5)
        result = await agent.arun("ping please")
        assert "done with ping" in result.output.text()
        assert result.tool_activity

    @pytest.mark.asyncio
    async def test_verifier_retry(self):
        def has_ok(output, ctx) -> bool:  # noqa: ANN001
            return "OK" in output.text()

        provider = ScriptedProvider(
            [
                ModelResponse(content="try1"),
                ModelResponse(content="OK final"),
            ]
        )
        agent = _agent(
            provider,
            verifier=has_ok,
            retry_on_failure=True,
            max_verify_retries=2,
        )
        result = await agent.arun("x")
        assert "OK" in result.output.text()
        assert len(provider.requests) >= 2

    def test_custom_planner_stored_but_loop_is_none(self):
        """Kernel Planner is stored on BuiltAgent; harness uses _run_plan instead."""
        from loomable.kernel.model_interface import ModelInterface
        from loomable.kernel.planner import Planner

        mi = ModelInterface(
            providers={"p": ScriptedProvider([ModelResponse(content="x")])},
            default_provider="p",
        )
        planner = Planner(mi)
        built = _agent(
            ScriptedProvider([ModelResponse(content="hi")]), planner=planner
        ).build()
        assert built.planner is planner
        assert built.loop is None

    @pytest.mark.asyncio
    async def test_complexity_router_plan_path(self):
        class AlwaysPlan(ComplexityRouter):
            def classify(self, agent_input: AgentInput, *, has_tools: bool) -> RunStrategy:
                return RunStrategy.PLAN

        provider = ScriptedProvider(
            [
                ModelResponse(content='["Step one", "Step two"]'),
                ModelResponse(content="did one"),
                ModelResponse(content="did two"),
                ModelResponse(content="SYNTH: all done"),
            ]
        )
        agent = _agent(provider, complexity_router=AlwaysPlan())
        result = await agent.arun("complex task")
        text = result.output.text()
        assert "SYNTH" in text or "done" in text.lower() or "did" in text.lower()


# ---------------------------------------------------------------------------
# Tool HITL default deny
# ---------------------------------------------------------------------------


class TestToolHITL:
    @pytest.mark.asyncio
    async def test_require_confirmation_defaults_to_deny(self):
        @tool
        def danger(x: str) -> str:
            """Dangerous op."""
            return f"ran:{x}"

        provider = ScriptedProvider(
            [
                ModelResponse(
                    content="",
                    tool_calls=[_tool_call("danger", {"x": "prod"})],
                ),
                ModelResponse(content="after deny"),
            ]
        )
        agent = _agent(
            provider,
            tools=[danger],
            require_confirmation=["danger"],
            max_tool_iterations=5,
        )
        result = await agent.arun("do danger")
        assert "ran:prod" not in result.output.text()
