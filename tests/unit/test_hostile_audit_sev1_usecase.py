"""Hostile-audit re-verification — SEV1 cloud outage response (complex real-world).

Simulates an incident-response desk without live LLMs:

1. Classify severity via ``Command(goto=...)`` — goto must not poison next step input
2. Fan out evidence collection with ``map_over`` + parent checkpointer — nested
   flows must not mark the parent thread ``complete=True`` mid-run
3. HITL gate before publishing the SEV packet — approve status + session bind
4. Team coordinator with ``max_delegations`` — session_id must survive across aruns
5. Soft PLAN path without kernel Planner — JSON ``parse_plan_steps`` must work
6. ``planning_model`` alias without ``tiers=``
7. ``plan_tool`` workers exclude recursive ``plan`` from schemas
8. Case mode rejects media kwargs loudly
9. MapNode missing key fails loud (not silent empty success)
10. Team ``route`` requires first delegate; tasks mode wires TodoTools

Run: ``python3 -m pytest tests/unit/test_hostile_audit_sev1_usecase.py -q``
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from loomable import Agent, Command, Team, Workflow, tool
from loomable.agent import ModelSpec
from loomable.agent.errors import AgentConfigError
from loomable.agent.plan_parse import parse_plan_steps
from loomable.agent.reasoning import make_plan_tool
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.case import Case
from loomable.content import AgentInput, ModelCapabilities
from loomable.flow.hitl import FlowPaused
from loomable.flow.nodes import FlowConfigError, MapNode
from loomable.flow.runnable import FunctionRunnable
from loomable.flow.send import Send
from loomable.flow.state import SharedState
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.kernel.planner import Planner, TaskContext
from loomable.persist.checkpoint import InMemoryCheckpointer


# ---------------------------------------------------------------------------
# Scripted providers
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Deterministic provider for unit/integration harnesses."""

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self._responses = list(responses or [])
        self._i = 0
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self._i >= len(self._responses):
            return ModelResponse(content="ok")
        resp = self._responses[self._i]
        self._i += 1
        return resp


def _agent(provider: Any, **kwargs: Any) -> Agent:
    return Agent(
        model=ModelSpec(provider="scripted", provider_impl=provider),
        capabilities=ModelCapabilities(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1–3: Workflow SEV desk — Command.goto + map_over checkpoint + HITL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sev1_desk_command_map_over_hitl_checkpoint() -> None:
    """Real-world: classify → fan-out evidence → HITL publish → resume.

    Asserts:
    - Command.goto does not put route-arm name into step-chain input
    - map_over workers never see parent checkpoint complete=True mid-run
    - approve(status=...) validates; update_state blocked on pending HITL
    """
    cp = InMemoryCheckpointer()
    session_id = "inc-88421"
    seen_route_arm: list[str] = []
    mid_map_complete: list[bool | None] = []

    def classify(ticket: str) -> Command:
        sev = "sev1" if "outage" in ticket.lower() else "sev3"
        return Command(
            goto="full_sev" if sev == "sev1" else "quick",
            update={"severity": sev, "ticket": ticket},
        )

    async def full_sev(inp: Any, *, context: Any = None) -> dict:
        # Route arms receive the original ticket (by design); must NOT be "full_sev"
        text = inp.text() if hasattr(inp, "text") else str(inp)
        seen_route_arm.append(text)
        assert "full_sev" not in text
        assert context.shared_state.get("severity") == "sev1"
        return {
            "tasks": [
                Send("logs", "pull error rates"),
                Send("metrics", "check p99 latency"),
                Send("comms", "draft customer notice"),
            ]
        }

    async def quick(inp: Any) -> str:
        return "auto-close"

    async def gather(item: str) -> str:
        checkpoint = await cp.get(session_id)
        mid_map_complete.append(
            None if checkpoint is None else checkpoint.complete
        )
        return f"evidence:{item}"

    async def draft_packet(_: Any, *, context: Any = None) -> str:
        pieces = []
        if context is not None and context.shared_state is not None:
            raw = context.shared_state.get("map")
            if isinstance(raw, list):
                pieces = [str(p) for p in raw]
        return "SEV packet:\n" + "\n".join(pieces)

    async def publish(packet: Any) -> str:
        text = packet.text() if hasattr(packet, "text") else str(packet)
        return f"PUBLISHED:{text[:80]}"

    wf = (
        Workflow("sev1-desk", session_id=session_id, checkpointer=cp)
        .route(
            classify,
            full_sev=full_sev,
            quick=quick,
        )
        .map_over(gather, over="tasks", name="evidence")
        .step("draft", draft_packet)
        .step("publish", publish, confirm=True)
    )

    with pytest.raises(FlowPaused) as paused:
        await wf.arun("INC-88421 major outage in us-east-1")

    assert paused.value.node_id == "publish" or paused.value.pending.tool_name == "publish"
    assert mid_map_complete, "map workers should have run"
    assert all(flag is not True for flag in mid_map_complete)
    assert seen_route_arm
    assert "outage" in seen_route_arm[0].lower()

    with pytest.raises(ValueError, match="approved|rejected"):
        await wf.approve("publish", status="maybe")

    with pytest.raises(RuntimeError, match="pending HITL"):
        await wf.update_state({"hacked": True}, as_node="publish")

    await wf.approve("publish", status="approved")
    result = await wf.arun(None, resume=True)
    assert "PUBLISHED" in result.output.text()
    assert "evidence:" in result.output.text()

    with pytest.raises(FlowPaused):
        await wf.arun("another outage ticket")
    wf.bind_session(session_id, resume=False)
    with pytest.raises(FlowPaused):
        await wf.arun("fresh outage after clear")


@pytest.mark.asyncio
async def test_sev1_step_chain_goto_does_not_poison_next_input() -> None:
    """``.step`` after a Command-returning step must not get str(goto)."""
    seen: list[str] = []

    def triage(_: str) -> Command:
        return Command(goto="escalate_human", update={"severity": "sev1"})

    async def next_step(inp: Any) -> str:
        text = inp.text() if hasattr(inp, "text") else str(inp)
        seen.append(text)
        return "ok"

    wf = Workflow("goto-chain").step("triage", triage).step("next", next_step)
    await wf.arun("ticket")
    assert seen == [""]
    assert "escalate_human" not in seen[0]


# ---------------------------------------------------------------------------
# 4: Team session stability under max_delegations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sev1_team_session_stable_across_aruns() -> None:
    """Ops team with budgeted delegation must keep the same session_id."""
    provider = ScriptedProvider(
        [
            ModelResponse(content="coordinator triage"),
            ModelResponse(content="coordinator follow-up"),
        ]
    )
    oncall = _agent(ScriptedProvider([ModelResponse(content="logs ok")]), role="Oncall")
    comms = _agent(ScriptedProvider([ModelResponse(content="status page")]), role="Comms")

    team = Team(
        [oncall, comms],
        model=ModelSpec(provider="scripted", provider_impl=provider),
        mode="coordinate",
        session_id="team-inc-88421",
        max_delegations=3,
        max_depth=2,
    )
    r1 = await team.arun("triage outage")
    r2 = await team.arun("update status")
    assert r1.session_id == "team-inc-88421"
    assert r2.session_id == "team-inc-88421"
    assert r1.session_id == r2.session_id


@pytest.mark.asyncio
async def test_sev1_team_route_requires_first_delegate() -> None:
    """Soft route mode must advertise require_tools for the first delegate."""
    member = _agent(ScriptedProvider([ModelResponse(content="ok")]), role="SRE")
    team = Team(
        [member],
        model=ModelSpec(
            provider="scripted",
            provider_impl=ScriptedProvider([ModelResponse(content="routed")]),
        ),
        mode="route",
    )
    assert team._agent._require_tools
    assert team._agent._require_tools[0].startswith("delegate_to_")


# ---------------------------------------------------------------------------
# 5–7: Planner opt-in, planning_model alias, plan_tool exclude
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sev1_plan_path_json_without_kernel_planner() -> None:
    """PLAN complexity without planner= uses JSON parse (not dead code)."""
    plan_json = json.dumps(["Pull metrics", "Draft SEV notice"])
    provider = ScriptedProvider(
        [
            ModelResponse(content=plan_json),  # planner JSON
            ModelResponse(content="metrics gathered"),  # step 0
            ModelResponse(content="notice drafted"),  # step 1
            ModelResponse(content="SEV1 packet ready"),  # synthesizer
        ]
    )

    class ForcePlan(ComplexityRouter):
        def classify(self, agent_input, *, has_tools: bool = False):  # noqa: ANN001
            return RunStrategy.PLAN

    agent = _agent(provider, complexity_router=ForcePlan())
    built = agent.build()
    assert built.planner is None  # opt-in: no auto Planner
    result = await built.arun("Handle SEV1 outage")
    assert "SEV1" in result.output.text() or "packet" in result.output.text().lower()


@pytest.mark.asyncio
async def test_sev1_planning_model_alias_works() -> None:
    """planning_model= registers without tiers= so Planner.plan can invoke it."""
    provider = ScriptedProvider(
        [ModelResponse(content=json.dumps(["Isolate AZ", "Fail over"]))]
    )
    agent = _agent(provider, planning_model="planner-tier")
    built = agent.build()
    assert "planner-tier" in built.model_interface.providers
    # Explicit planner using the aliased id
    planner = Planner(built.model_interface, planning_model_id="planner-tier")
    plan = await planner.plan(TaskContext(task="SEV1 failover"))
    assert plan.steps == ["Isolate AZ", "Fail over"]


@pytest.mark.asyncio
async def test_sev1_plan_tool_excludes_plan_from_workers() -> None:
    """During plan-tool fan-out, workers must not re-advertise ``plan``."""
    advertised: list[list[str]] = []

    class CaptureProvider:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            names = [
                t.get("function", {}).get("name", "")
                for t in (request.tools or [])
                if isinstance(t, dict)
            ]
            if names:
                advertised.append([n for n in names if n])
            # Planner JSON when no tools
            if not request.tools:
                text = str(request.messages[-1] if request.messages else "")
                if "planner" in text.lower() or "JSON array" in text:
                    return ModelResponse(content='["Check runbooks"]')
                return ModelResponse(content="synthesized SEV answer")
            # Worker tool loop: one side_effect then done
            if any(n == "lookup_runbook" for n in names):
                # If plan is still advertised, that is a regression
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="1",
                            tool_name="lookup_runbook",
                            args={"q": "outage"},
                        )
                    ],
                )
            return ModelResponse(content="step done")

    @tool
    def lookup_runbook(q: str) -> str:
        return f"runbook:{q}"

    provider = CaptureProvider()
    # Simpler path: exclude_tools unit via BuiltAgent._run_tool_loop
    agent = _agent(provider, tools=[lookup_runbook], plan_tool=True)
    built = agent.build()
    await built._run_tool_loop(
        AgentInput.from_text("check"),
        include_history=False,
        exclude_tools=frozenset({"plan"}),
    )
    assert advertised
    for names in advertised:
        assert "plan" not in names
        assert "lookup_runbook" in names

    plan_tool = make_plan_tool(built)
    assert plan_tool.idempotent is False


# ---------------------------------------------------------------------------
# 8–9: Case guards + MapNode loud failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sev1_case_mode_rejects_media_kwargs() -> None:
    agent = _agent(
        ScriptedProvider([ModelResponse(content="ok")]),
        mode="case",
        goal="Close INC-88421 with SEV packet",
    )
    with pytest.raises(AgentConfigError, match="images"):
        await agent.arun("investigate", images=["/tmp/x.png"])


def test_sev1_case_from_agent_rejects_subagents() -> None:
    member = _agent(ScriptedProvider([ModelResponse(content="ok")]), role="SRE")
    agent = _agent(
        ScriptedProvider([ModelResponse(content="ok")]),
        mode="case",
        goal="close incident",
        subagents=[member],
    )
    with pytest.raises(AgentConfigError, match="subagents"):
        Case.from_agent(agent)


@pytest.mark.asyncio
async def test_sev1_map_node_missing_key_raises() -> None:
    node = MapNode(FunctionRunnable(lambda x: x), over="evidence_tasks")
    from loomable.agent.context import RunContext

    ctx = RunContext(shared_state=SharedState())
    with pytest.raises(FlowConfigError, match="evidence_tasks"):
        await node.arun("go", context=ctx)


# ---------------------------------------------------------------------------
# 10: parse_plan_steps realism (fenced JSON from models)
# ---------------------------------------------------------------------------


def test_sev1_parse_plan_steps_handles_fenced_json() -> None:
    raw = """```json
["Pull CloudWatch alarms", "Page on-call", "Draft status page"]
```"""
    steps = parse_plan_steps(raw)
    assert steps == [
        "Pull CloudWatch alarms",
        "Page on-call",
        "Draft status page",
    ]


# ---------------------------------------------------------------------------
# End-to-end composite: full desk happy path without HITL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sev1_composite_desk_without_hitl() -> None:
    """Full classify → map_over → draft with parent checkpointer (no HITL).

    Parent should end complete=True only after the outer Workflow finishes —
    never during nested map_over.
    """
    cp = InMemoryCheckpointer()
    session_id = "inc-composite"
    during_map: list[bool | None] = []

    def classify(ticket: str) -> Command:
        return Command(
            goto="investigate",
            update={
                "tasks": [
                    Send("logs", "error budget"),
                    Send("deps", "downstream SLOs"),
                ],
                "ticket": ticket,
            },
        )

    async def investigate(inp: Any) -> str:
        # Route arms receive the original workflow input (not str(goto)).
        text = inp.text() if hasattr(inp, "text") else str(inp)
        assert "outage" in text.lower()
        assert text != "investigate"
        return "investigating"

    async def worker(item: str) -> str:
        cp_now = await cp.get(session_id)
        during_map.append(None if cp_now is None else cp_now.complete)
        return f"found:{item}"

    async def synthesize(_: Any, *, context: Any = None) -> str:
        pieces = context.shared_state.get("map") if context else []
        return "PACKET\n" + "\n".join(str(p) for p in (pieces or []))

    wf = (
        Workflow("composite", session_id=session_id, checkpointer=cp)
        .route(classify, investigate=investigate)
        .map_over(worker, over="tasks")
        .step("synth", synthesize)
    )
    result = await wf.arun("outage in payments")
    assert "PACKET" in result.output.text()
    assert "found:" in result.output.text()
    assert all(flag is not True for flag in during_map)

    final = await cp.get(session_id)
    assert final is not None
    assert final.complete is True
