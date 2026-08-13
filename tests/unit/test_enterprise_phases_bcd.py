"""Vigorous unit tests for Phases B/C/D enterprise features."""

from __future__ import annotations

import pytest

from loomable import (
    Agent,
    ContextPolicy,
    InMemoryCheckpointer,
    Team,
    Workflow,
    spawn_specialist,
)
from loomable.agent import ModelSpec
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, Text
from loomable.kernel.models import ModelRequest, ModelResponse, Turn
from loomable.persist.checkpoint import Checkpoint


class _Echo:
    def __init__(self, label: str = "echo") -> None:
        self.label = label

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content=f"{self.label}")


@pytest.mark.asyncio
async def test_workflow_resume_skips_completed_nodes() -> None:
    cp = InMemoryCheckpointer()
    session = "t-resume"
    calls = {"a": 0, "b": 0}

    async def a(inp, *, context=None):
        calls["a"] += 1
        return RunResult(output=AgentOutput(parts=[Text("A")]), session_id=session)

    async def b(inp, *, context=None):
        calls["b"] += 1
        text = inp.text() if hasattr(inp, "text") else str(inp)
        return RunResult(output=AgentOutput(parts=[Text(f"B:{text}")]), session_id=session)

    # Seed incomplete checkpoint after step a
    from loomable.flow.state import SharedState

    state = SharedState()
    state.write("a", AgentOutput(parts=[Text("A")]))
    await cp.put(
        Checkpoint(
            thread_id=session,
            step=1,
            session_state={
                "shared_state": state.snapshot(),
                "completed_node_ids": ["a"],
            },
            complete=False,
        )
    )

    wf = Workflow("r", session_id=session, checkpointer=cp).step("a", a).step("b", b)
    result = await wf.arun("x", resume=True)
    assert calls["a"] == 0
    assert calls["b"] == 1
    assert result.metadata.get("resumed") is True
    assert result.output.text() == "B:A"


@pytest.mark.asyncio
async def test_resume_true_without_checkpoint_raises() -> None:
    cp = InMemoryCheckpointer()
    wf = Workflow("r", session_id="missing", checkpointer=cp).step(
        "a",
        lambda x: "ok",
    )
    with pytest.raises(RuntimeError, match="no incomplete checkpoint"):
        await wf.arun("x", resume=True)


def test_context_policy_compaction_and_spill() -> None:
    from loomable.kernel.summarizer import Summarizer

    policy = ContextPolicy(memory_window=2, compaction_threshold=3, token_budget=100)
    turns = [
        Turn(role="user", content=f"u{i}", tokens=1, step=i) for i in range(6)
    ]
    turns[0].content = "PIN:keep"
    new_l1, summaries, outcome = policy.compact_turns(
        turns,
        pinned_steps={0},
        summarizer=Summarizer(1),
    )
    assert outcome.compacted
    assert any("PIN:keep" in (t.content or "") for t in new_l1)
    assert summaries

    msgs = [
        {"role": "tool", "content": "Z" * 5000, "tool_call_id": "1"},
    ]
    # Force hard spill by tiny budget
    policy.token_budget = 10
    policy.hard_limit_ratio = 0.01
    spilled = policy.spill_bulky_tool_messages(msgs, max_tool_chars=100, force=True)
    assert len(spilled[0]["content"]) < 5000


@pytest.mark.asyncio
async def test_team_hard_broadcast_and_sequential() -> None:
    a = Agent(model=ModelSpec(provider="a", provider_impl=_Echo("A")), role="Alpha")
    b = Agent(model=ModelSpec(provider="b", provider_impl=_Echo("B")), role="Beta")
    team = Team(
        members=[a, b],
        model=ModelSpec(provider="m", provider_impl=_Echo("M")),
        mode="broadcast",
        hard=True,
    )
    result = await team.arun("task")
    assert "Alpha" in result.output.text() and "Beta" in result.output.text()
    assert "A" in result.output.text() and "B" in result.output.text()

    seq = Team(
        members=[a, b],
        model=ModelSpec(provider="m", provider_impl=_Echo("M")),
        mode="sequential",
        hard=True,
    )
    seq_result = await seq.arun("task")
    assert "B" in seq_result.output.text()


@pytest.mark.asyncio
async def test_spawn_specialist() -> None:
    text = await spawn_specialist(
        model=ModelSpec(provider="e", provider_impl=_Echo("spawned")),
        role="Auditor",
        task="check cert",
    )
    assert text == "spawned"


@pytest.mark.asyncio
async def test_delegation_budget() -> None:
    from loomable.agent.delegation import make_delegation_tools

    sub = Agent(model=ModelSpec(provider="s", provider_impl=_Echo("sub")), role="Worker")
    tools = make_delegation_tools([sub], max_delegations=1)
    tool = tools[0]
    first = await tool.invoke({"task": "one"})
    second = await tool.invoke({"task": "two"})
    assert "sub" in str(first.content).lower() or first.content == "sub"
    assert "budget" in str(second.content).lower() or "max_delegations" in str(second.content)
