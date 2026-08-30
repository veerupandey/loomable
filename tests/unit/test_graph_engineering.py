"""Unit tests for graph-engineering Workflow primitives.

Covers patterns from graph engineering practice:
- Local failure policies on Step (retry / skip / fallback / stop)
- Edge payload contracts via Step.reads=
- Inspectable route decisions on RouterNode / Workflow.branch
- Workflow.verify() bounded repair cycle
- Step.complexity cost hint
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.nodes import Edge, FlowConfigError, RouterNode
from loomable.flow.state import SharedState
from loomable.flow.step import Step, StepFailed
from loomable.flow.workflow import Workflow


def _text_result(text: str) -> RunResult:
    return RunResult(
        output=AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=text.encode("utf-8"),
                )
            ]
        ),
        session_id="",
    )


# ---------------------------------------------------------------------------
# Failure policies
# ---------------------------------------------------------------------------


class TestStepFailurePolicy:
    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        calls = {"n": 0}

        async def flaky(input, *, context=None):  # noqa: A002
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return _text_result("ok")

        step = Step("flaky", flaky, on_failure="retry", max_retries=3)
        result = await step.arun("x")
        assert result.output.text() == "ok"
        assert result.metadata["failure_retries"] == 2
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises_step_failed(self):
        async def always_fail(input, *, context=None):  # noqa: A002
            raise RuntimeError("boom")

        step = Step("bad", always_fail, on_failure="retry", max_retries=1)
        with pytest.raises(StepFailed, match="bad"):
            await step.arun("x")

    @pytest.mark.asyncio
    async def test_skip_returns_empty_result(self):
        async def boom(input, *, context=None):  # noqa: A002
            raise RuntimeError("optional branch failed")

        step = Step("optional", boom, on_failure="skip")
        result = await step.arun("x")
        assert result.metadata["step_skipped"] is True
        assert result.output.text() == ""

    @pytest.mark.asyncio
    async def test_fallback_runs_alternate(self):
        async def primary(input, *, context=None):  # noqa: A002
            raise RuntimeError("primary down")

        async def backup(input, *, context=None):  # noqa: A002
            return _text_result("from-fallback")

        step = Step("svc", primary, on_failure="fallback", fallback=backup)
        result = await step.arun("x")
        assert result.output.text() == "from-fallback"
        assert result.metadata["fallback_used"] is True

    @pytest.mark.asyncio
    async def test_stop_raises_step_failed(self):
        async def boom(input, *, context=None):  # noqa: A002
            raise RuntimeError("fatal")

        step = Step("gate", boom, on_failure="stop")
        with pytest.raises(StepFailed) as ei:
            await step.arun("x")
        assert ei.value.action == "stop"

    def test_fallback_requires_fallback_runnable(self):
        with pytest.raises(ValueError, match="fallback"):
            Step("x", lambda i: i, on_failure="fallback")

    @pytest.mark.asyncio
    async def test_workflow_skip_continues(self):
        async def boom(input, *, context=None):  # noqa: A002
            raise RuntimeError("skip me")

        async def next_step(input, *, context=None):  # noqa: A002
            return _text_result("continued")

        wf = (
            Workflow("pipe")
            .step("optional", boom, on_failure="skip")
            .step("next", next_step)
        )
        result = await wf.arun("in")
        assert "continued" in result.output.text()


# ---------------------------------------------------------------------------
# Edge payload contracts (reads=)
# ---------------------------------------------------------------------------


class TestEdgePayloadContract:
    @pytest.mark.asyncio
    async def test_reads_feeds_named_state_key(self):
        async def writer(input, *, context=None):  # noqa: A002
            if context and context.shared_state:
                context.shared_state.write(
                    "evidence",
                    AgentOutput(
                        parts=[
                            MediaPart(
                                modality=Modality.TEXT,
                                media_type="text/plain",
                                data=b"cited-facts",
                            )
                        ]
                    ),
                )
            return _text_result("noise-from-writer")

        captured: dict[str, str] = {}

        async def reader(input, *, context=None):  # noqa: A002
            text = input.text() if hasattr(input, "text") else str(input)
            captured["got"] = text
            return _text_result(f"draft:{text}")

        wf = (
            Workflow("article")
            .step("research", writer)
            .step("draft", reader, reads="evidence")
        )
        result = await wf.arun("topic")
        assert captured["got"] == "cited-facts"
        assert "draft:cited-facts" in result.output.text()

        # Compiler stamped payload_key on the edge
        flow = wf.build()._compiled_flow
        assert flow is not None
        payload_edges = [e for e in flow._edges if e.payload_key == "evidence"]
        assert len(payload_edges) == 1


# ---------------------------------------------------------------------------
# Route decision records
# ---------------------------------------------------------------------------


class TestRouteDecision:
    @pytest.mark.asyncio
    async def test_router_writes_route_decision(self):
        def chooser(input):  # noqa: A002
            return input

        router = RouterNode(chooser, choices=["quick", "full"])
        state = SharedState()
        ctx = RunContext(shared_state=state)
        result = await router.arun("quick", context=ctx)

        assert result.metadata["router_selected"] == "quick"
        decision = result.metadata["route_decision"]
        assert decision["selected"] == "quick"
        assert decision["choices"] == ["quick", "full"]
        assert state.get("_route_decision")["selected"] == "quick"

    @pytest.mark.asyncio
    async def test_router_captures_reason_from_metadata(self):
        class ReasonChooser:
            async def arun(self, input, *, context=None):  # noqa: A002
                return RunResult(
                    output=AgentOutput(
                        parts=[
                            MediaPart(
                                modality=Modality.TEXT,
                                media_type="text/plain",
                                data=b"",
                            )
                        ]
                    ),
                    session_id="",
                    metadata={"selection": "full", "reason": "high severity"},
                )

        router = RouterNode(ReasonChooser(), choices=["quick", "full"])
        result = await router.arun("x", context=RunContext(shared_state=SharedState()))
        assert result.metadata["route_decision"]["reason"] == "high severity"

    @pytest.mark.asyncio
    async def test_workflow_branch_records_condition_reason(self):
        async def left(input, *, context=None):  # noqa: A002
            return _text_result("THEN")

        async def right(input, *, context=None):  # noqa: A002
            return _text_result("ELSE")

        wf = Workflow("branch").branch(
            when=lambda state: True,
            then=Step("then_path", left),
            else_=Step("else_path", right),
        )
        result = await wf.arun("x")
        assert "THEN" in result.output.text()
        decision = wf.state.get("_route_decision")
        assert decision is not None
        assert decision["reason"] == "condition_true"


# ---------------------------------------------------------------------------
# Workflow.verify
# ---------------------------------------------------------------------------


class TestWorkflowVerify:
    @pytest.mark.asyncio
    async def test_verify_repairs_until_ok(self):
        calls = {"n": 0}

        async def body(input, *, context=None):  # noqa: A002
            calls["n"] += 1
            return _text_result("ready" if calls["n"] >= 2 else "draft")

        def check(output, ctx) -> bool:
            return "ready" in output.text()

        wf = Workflow("gate").verify(body, check=check, max_retries=3)
        result = await wf.arun("start")
        assert result.metadata.get("loop_verified") is True
        assert calls["n"] == 2

    def test_verify_requires_check(self):
        with pytest.raises(ValueError, match="check"):
            Workflow("x").verify(lambda i: i, check=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Complexity hint
# ---------------------------------------------------------------------------


class TestStepComplexity:
    def test_complexity_stored_on_step(self):
        step = Step("extract", lambda i: i, complexity="low")
        assert step.complexity == "low"

    def test_invalid_complexity_raises(self):
        with pytest.raises(ValueError, match="complexity"):
            Step("x", lambda i: i, complexity="medium")  # type: ignore[arg-type]

    def test_workflow_step_forwards_complexity(self):
        wf = Workflow("cost").step("simple", lambda i: "ok", complexity="low")
        assert wf._steps[0].complexity == "low"
