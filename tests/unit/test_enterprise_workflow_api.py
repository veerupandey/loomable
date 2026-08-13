"""Unit tests for fluent Workflow API and ergonomic modalities."""

from __future__ import annotations

import pytest

from loomable import (
    Agent,
    InMemoryCheckpointer,
    Step,
    Workflow,
    parallel,
    sequential,
)
from loomable.content import AgentOutput, Modality, Text, capabilities_for
from loomable.kernel.models import ModelRequest, ModelResponse


def _text(value: object) -> str:
    if isinstance(value, AgentOutput):
        return value.text()
    if hasattr(value, "output") and hasattr(value.output, "text"):
        return value.output.text()
    return str(value)


class _Echo:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


@pytest.mark.asyncio
async def test_fluent_workflow_sequential() -> None:
    async def a(inp, *, context=None):
        from loomable.agent.run import RunResult

        return RunResult(output=AgentOutput(parts=[Text(f"A:{_text(inp)}")]), session_id="s")

    async def b(inp, *, context=None):
        from loomable.agent.run import RunResult

        return RunResult(output=AgentOutput(parts=[Text(f"B:{_text(inp)}")]), session_id="s")

    wf = Workflow("pipe").step("a", a).step("b", b)
    plan = wf.explain()
    assert "a" in plan.original_nodes
    assert "b" in plan.original_nodes

    result = await wf.arun("hi")
    assert result.output.text() == "B:A:hi"


@pytest.mark.asyncio
async def test_workflow_parallel_kwargs() -> None:
    async def left(inp, *, context=None):
        from loomable.agent.run import RunResult

        return RunResult(output=AgentOutput(parts=[Text("L")]), session_id="s")

    async def right(inp, *, context=None):
        from loomable.agent.run import RunResult

        return RunResult(output=AgentOutput(parts=[Text("R")]), session_id="s")

    wf = Workflow("fan").parallel(left=left, right=right)
    plan = wf.explain()
    assert any("parallel" in n or n in {"left", "right"} for n in plan.original_nodes)
    result = await wf.arun("x")
    # Parallel group merges / returns last branch text; just ensure it ran.
    assert result.output.text() in {"L", "R", "x"} or len(result.output.text()) > 0


@pytest.mark.asyncio
async def test_workflow_branch_then() -> None:
    async def then_step(inp, *, context=None):
        from loomable.agent.run import RunResult

        return RunResult(output=AgentOutput(parts=[Text("THEN")]), session_id="s")

    async def else_step(inp, *, context=None):
        from loomable.agent.run import RunResult

        return RunResult(output=AgentOutput(parts=[Text("ELSE")]), session_id="s")

    wf = Workflow("branchy").branch(
        when=lambda state: True,
        then=Step("go_then", then_step),
        else_=Step("go_else", else_step),
    )
    result = await wf.arun("seed")
    # Join may passthrough; prefer sub_results when present
    texts = [result.output.text()]
    if result.sub_results:
        texts.extend(r.output.text() for r in result.sub_results.values())
    assert any("THEN" in t for t in texts)


def test_capabilities_for_strings() -> None:
    caps = capabilities_for("text")
    assert caps.input == frozenset({Modality.TEXT})
    assert caps.output == frozenset({Modality.TEXT})

    caps2 = capabilities_for("text+image")
    assert Modality.IMAGE in caps2.input
    assert Modality.TEXT in caps2.input
    assert Modality.VIDEO not in caps2.input

    caps3 = capabilities_for(input="text+audio", output="text")
    assert Modality.AUDIO in caps3.input


def test_agent_modalities_and_text_only() -> None:
    a = Agent(model=_Echo(), text_only=True).build()
    assert a.capabilities.input == frozenset({Modality.TEXT})

    b = Agent(model=_Echo(), modalities="text+image").build()
    assert Modality.IMAGE in b.capabilities.input
    assert Modality.VIDEO not in b.capabilities.input

    c = Agent(model=_Echo(), capabilities="text").build()
    assert c.capabilities.input == frozenset({Modality.TEXT})


def test_agent_rejects_conflicting_modality_kwargs() -> None:
    with pytest.raises(Exception):
        Agent(model=_Echo(), text_only=True, modalities="text+image")


def test_top_level_imports() -> None:
    import loomable as L

    assert L.Agent is Agent
    assert L.Workflow is Workflow
    assert L.Step is Step
    assert L.InMemoryCheckpointer is InMemoryCheckpointer


def test_helpers_accept_checkpointer() -> None:
    cp = InMemoryCheckpointer()
    flow = sequential(lambda x: x, session_id="s1", checkpointer=cp)
    assert flow._checkpointer is cp
    flow2 = parallel(lambda x: x, session_id="s1", checkpointer=cp)
    assert flow2._checkpointer is cp


@pytest.mark.asyncio
async def test_workflow_wires_checkpointer() -> None:
    cp = InMemoryCheckpointer()

    async def only(inp, *, context=None):
        from loomable.agent.run import RunResult

        return RunResult(output=AgentOutput(parts=[Text(_text(inp))]), session_id="s")

    wf = Workflow("durable", session_id="thread-1", checkpointer=cp).step("only", only)
    flow = wf.flow
    assert flow._checkpointer is cp
    assert flow._session_id == "thread-1"
    result = await wf.arun("ok")
    assert result.output.text() == "ok"


@pytest.mark.asyncio
async def test_workflow_step_confirm_pauses_and_approves() -> None:
    from loomable.flow.hitl import FlowPaused

    cp = InMemoryCheckpointer()
    calls = {"scribe": 0}

    async def draft(inp, *, context=None):
        from loomable.agent.run import RunResult

        return RunResult(output=AgentOutput(parts=[Text("DRAFT")]), session_id="s")

    async def scribe(inp, *, context=None):
        from loomable.agent.run import RunResult

        calls["scribe"] += 1
        return RunResult(output=AgentOutput(parts=[Text("DONE")]), session_id="s")

    wf = (
        Workflow("hitl", session_id="t-hitl", checkpointer=cp)
        .step("draft", draft)
        .step("scribe", scribe, confirm=True)
    )

    with pytest.raises(FlowPaused) as exc:
        await wf.arun("go")
    assert exc.value.pending.tool_name == "scribe"
    assert calls["scribe"] == 0

    await wf.approve("scribe")
    result = await wf.arun("go", resume=True)
    assert calls["scribe"] == 1
    assert result.output.text() == "DONE"
    assert result.metadata.get("resumed") is True
    assert "draft" in (result.metadata.get("skipped_nodes") or [])


def test_step_confirm_alias() -> None:
    async def noop(inp, *, context=None):
        from loomable.agent.run import RunResult

        return RunResult(output=AgentOutput(parts=[Text("x")]), session_id="s")

    s = Step("gate", noop, confirm=True)
    assert s.require_confirmation is True
    node = Workflow("w").step("gate", noop, confirm=True).flow._nodes["gate"]
    assert node.require_confirmation is True


@pytest.mark.asyncio
async def test_workflow_confirm_approve_json_file_checkpointer(tmp_path) -> None:
    """JsonFileCheckpointer must persist approve() as the latest checkpoint."""
    from loomable.flow.hitl import FlowPaused
    from loomable.persist import JsonFileCheckpointer

    cp = JsonFileCheckpointer(str(tmp_path / "ckpts"))
    calls = {"scribe": 0}

    async def draft(inp, *, context=None):
        from loomable.agent.run import RunResult

        return RunResult(output=AgentOutput(parts=[Text("DRAFT")]), session_id="s")

    async def scribe(inp, *, context=None):
        from loomable.agent.run import RunResult

        calls["scribe"] += 1
        return RunResult(output=AgentOutput(parts=[Text("DONE")]), session_id="s")

    wf = (
        Workflow("hitl-json", session_id="t-json", checkpointer=cp)
        .step("draft", draft)
        .step("scribe", scribe, confirm=True)
    )

    with pytest.raises(FlowPaused):
        await wf.arun("go")

    await wf.approve("scribe")
    latest = await cp.get("t-json")
    assert latest is not None
    assert latest.pending[0].status == "approved"

    result = await wf.arun("go", resume=True)
    assert calls["scribe"] == 1
    assert result.output.text() == "DONE"
