"""Hard-scenario audit tests for graph-engineering primitives.

Covers failure modes the happy-path suite does not:
- Parallel stop commits successful siblings before escalating
- Workflow.state preserved after StepFailed
- Cancel interrupts retry loops
- Fallback with None metadata / fallback that also fails
- max_retries honored for non-retry policies
- Missing reads= key falls back to predecessor
- verify budget exhaustion
- Branch else route decision
- Checkpoint durability across parallel stop
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.step import Step, StepFailed
from loomable.flow.workflow import Workflow
from loomable.persist.checkpoint import InMemoryCheckpointer


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


class TestParallelStopCommitsSiblings:
    @pytest.mark.asyncio
    async def test_successful_sibling_lands_in_state(self):
        async def ok(input, *, context=None):  # noqa: A002
            return _text_result("OK")

        async def bad(input, *, context=None):  # noqa: A002
            raise RuntimeError("fatal")

        async def after(input, *, context=None):  # noqa: A002
            return _text_result("SHOULD_NOT_RUN")

        wf = (
            Workflow("p")
            .parallel(
                Step("ok", ok),
                Step("bad", bad, on_failure="stop"),
            )
            .step("after", after)
        )
        with pytest.raises(StepFailed, match="bad"):
            await wf.arun("x")

        assert wf.state.get("ok") is not None
        assert wf.state.get("ok").text() == "OK"
        assert wf.state.get("after") is None

    @pytest.mark.asyncio
    async def test_parallel_stop_checkpoints_successful_siblings(self):
        from loomable.flow.flow import Flow

        cp = InMemoryCheckpointer()

        async def ok(input, *, context=None):  # noqa: A002
            return _text_result("OK")

        async def bad(input, *, context=None):  # noqa: A002
            raise RuntimeError("fatal")

        flow = Flow(
            nodes={
                "ok": Step("ok", ok),
                "bad": Step("bad", bad, on_failure="stop"),
            },
            edges=[],
            engine="parallel",
            checkpointer=cp,
            session_id="t-stop",
        )
        with pytest.raises(StepFailed):
            await flow.arun("x")

        saved = await cp.get("t-stop")
        assert saved is not None
        assert saved.complete is False
        assert "ok" in saved.session_state.get("completed_node_ids", [])
        assert "ok" in saved.session_state.get("shared_state", {})

    @pytest.mark.asyncio
    async def test_workflow_parallel_stop_checkpoints_via_scoped_id(self):
        cp = InMemoryCheckpointer()

        async def ok(input, *, context=None):  # noqa: A002
            return _text_result("OK")

        async def bad(input, *, context=None):  # noqa: A002
            raise RuntimeError("fatal")

        wf = Workflow(
            "p",
            session_id="outer",
            checkpointer=cp,
        ).parallel(
            Step("ok", ok),
            Step("bad", bad, on_failure="stop"),
        )
        with pytest.raises(StepFailed):
            await wf.arun("x")

        group_name = wf._steps[0].name
        scoped = await cp.get(f"outer::parallel::{group_name}")
        assert scoped is not None
        assert "ok" in scoped.session_state.get("completed_node_ids", [])
        assert wf.state.get("ok") is not None


class TestStatePreservedOnHardStop:
    @pytest.mark.asyncio
    async def test_sequential_stop_keeps_prior_state(self):
        async def s1(input, *, context=None):  # noqa: A002
            return _text_result("one")

        async def s2(input, *, context=None):  # noqa: A002
            raise RuntimeError("stop")

        async def s3(input, *, context=None):  # noqa: A002
            return _text_result("three")

        wf = (
            Workflow("seq")
            .step("s1", s1)
            .step("s2", s2, on_failure="stop")
            .step("s3", s3)
        )
        with pytest.raises(StepFailed):
            await wf.arun("x")

        assert wf.state.get("s1") is not None
        assert wf.state.get("s1").text() == "one"
        assert wf.state.get("s3") is None


class TestCancelInterruptsRetry:
    @pytest.mark.asyncio
    async def test_cancel_stops_further_retries(self):
        ctx = RunContext()
        calls = {"n": 0}

        async def flaky(input, *, context=None):  # noqa: A002
            calls["n"] += 1
            if calls["n"] == 1:
                ctx.cancel()
            raise RuntimeError("transient")

        result = await Step(
            "f", flaky, on_failure="retry", max_retries=5
        ).arun("x", context=ctx)

        # First attempt runs; cancel is seen before the second attempt.
        assert calls["n"] == 1
        assert result.metadata.get("step_cancelled") is True
        assert result.metadata.get("stop_reason") == "cancelled"


class TestFallbackEdgeCases:
    @pytest.mark.asyncio
    async def test_fallback_none_metadata_safe(self):
        async def primary(input, *, context=None):  # noqa: A002
            raise RuntimeError("primary down")

        class OddFallback:
            async def arun(self, input, *, context=None):  # noqa: A002
                r = _text_result("fb")
                r.metadata = None  # type: ignore[assignment]
                return r

        result = await Step(
            "s", primary, on_failure="fallback", fallback=OddFallback()
        ).arun("x")
        assert result.output.text() == "fb"
        assert result.metadata["fallback_used"] is True

    @pytest.mark.asyncio
    async def test_fallback_that_also_fails_propagates(self):
        async def primary(input, *, context=None):  # noqa: A002
            raise RuntimeError("primary")

        async def fb(input, *, context=None):  # noqa: A002
            raise RuntimeError("fallback dead")

        with pytest.raises(RuntimeError, match="fallback dead"):
            await Step(
                "s", primary, on_failure="fallback", fallback=fb
            ).arun("x")


class TestMaxRetriesAcrossPolicies:
    @pytest.mark.asyncio
    async def test_skip_honors_max_retries_then_skips(self):
        calls = {"n": 0}

        async def boom(input, *, context=None):  # noqa: A002
            calls["n"] += 1
            raise RuntimeError("x")

        result = await Step(
            "opt", boom, on_failure="skip", max_retries=2
        ).arun("x")
        assert calls["n"] == 3
        assert result.metadata["step_skipped"] is True
        assert result.metadata["failure_attempts"] == 3

    @pytest.mark.asyncio
    async def test_raise_with_max_retries_retries_then_raises(self):
        calls = {"n": 0}

        async def boom(input, *, context=None):  # noqa: A002
            calls["n"] += 1
            raise RuntimeError("x")

        with pytest.raises(RuntimeError, match="x"):
            await Step("s", boom, on_failure="raise", max_retries=2).arun("x")
        assert calls["n"] == 3


class TestReadsFallbacks:
    @pytest.mark.asyncio
    async def test_missing_reads_key_uses_predecessor(self):
        captured: dict[str, str] = {}

        async def writer(input, *, context=None):  # noqa: A002
            return _text_result("pred-out")

        async def reader(input, *, context=None):  # noqa: A002
            captured["got"] = input.text() if hasattr(input, "text") else str(input)
            return _text_result("done")

        await (
            Workflow("r")
            .step("w", writer)
            .step("r", reader, reads="does_not_exist")
            .arun("x")
        )
        assert captured["got"] == "pred-out"


class TestVerifyBudgets:
    @pytest.mark.asyncio
    async def test_verify_exhausted_returns_unverified(self):
        calls = {"n": 0}

        async def body(input, *, context=None):  # noqa: A002
            calls["n"] += 1
            return _text_result("never-good")

        result = await Workflow("v").verify(
            body, check=lambda o, c: False, max_retries=2
        ).arun("x")
        assert result.metadata.get("loop_verified") is False
        assert result.metadata.get("loop_stop") == "max_iterations"
        assert calls["n"] == 3  # max_retries=2 → 3 attempts

    @pytest.mark.asyncio
    async def test_verify_max_retries_zero_single_attempt(self):
        calls = {"n": 0}

        async def body(input, *, context=None):  # noqa: A002
            calls["n"] += 1
            return _text_result("x")

        result = await Workflow("v").verify(
            body, check=lambda o, c: False, max_retries=0
        ).arun("x")
        assert calls["n"] == 1
        assert result.metadata.get("loop_verified") is False


class TestBranchElseDecision:
    @pytest.mark.asyncio
    async def test_else_branch_reason(self):
        async def left(input, *, context=None):  # noqa: A002
            return _text_result("THEN")

        async def right(input, *, context=None):  # noqa: A002
            return _text_result("ELSE")

        wf = Workflow("b").branch(
            when=lambda state: False,
            then=Step("then_path", left),
            else_=Step("else_path", right),
        )
        result = await wf.arun("x")
        assert "ELSE" in result.output.text()
        decision = wf.state.get("_route_decision")
        assert decision["reason"] == "condition_false"


class TestParallelSkipIsolation:
    @pytest.mark.asyncio
    async def test_one_skip_does_not_block_siblings_or_join(self):
        order: list[str] = []

        async def a(input, *, context=None):  # noqa: A002
            order.append("a")
            return _text_result("A")

        async def b(input, *, context=None):  # noqa: A002
            order.append("b")
            raise RuntimeError("optional")

        async def join(input, *, context=None):  # noqa: A002
            order.append("join")
            return _text_result("JOIN")

        result = await (
            Workflow("ps")
            .parallel(Step("a", a), Step("b", b, on_failure="skip"))
            .step("join", join)
            .arun("x")
        )
        assert "join" in order
        assert "JOIN" in result.output.text()
        assert "a" in order and "b" in order


class TestCancelledErrorNotSwallowed:
    @pytest.mark.asyncio
    async def test_asyncio_cancelled_error_propagates(self):
        import asyncio

        async def boom(input, *, context=None):  # noqa: A002
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await Step("c", boom, on_failure="skip").arun("x")
