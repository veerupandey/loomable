"""Tests for flow-level HITL: pause before require_confirmation nodes, resume.

Validates: Requirements 16.1, 16.2, 16.3, 16.4, 16.5
- Node with require_confirmation raises FlowPaused before execution
- FlowPaused carries PendingAction with correct node info
- After resume with approval, the node executes
- After resume with rejection, the node is skipped
- Flow without confirmation nodes runs to completion (no pause)
- Checkpoint is written when FlowPaused is raised
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.engines.sequential import SequentialEngine
from loomable.flow.flow import Flow
from loomable.flow.hitl import FlowPaused
from loomable.flow.nodes import Edge, Node
from loomable.flow.state import SharedState
from loomable.persist.checkpoint import InMemoryCheckpointer, PendingAction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(text: str) -> AgentOutput:
    """Create an AgentOutput with a single text part."""
    return AgentOutput(
        parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=text.encode())]
    )


class TrackingRunnable:
    """A Runnable that records calls and returns a text output."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0
        self.inputs: list = []

    async def arun(self, input, *, context=None):  # noqa: A002
        self.call_count += 1
        self.inputs.append(input)
        output = _make_output(f"{self.name}:executed")
        return RunResult(output=output, session_id="")


# ---------------------------------------------------------------------------
# FlowPaused exception structure (Req 16.1, 16.4)
# ---------------------------------------------------------------------------


class TestFlowPausedException:
    """Verify FlowPaused carries the correct data."""

    def test_carries_pending_action_and_thread_id(self):
        """FlowPaused stores both the PendingAction and thread_id."""
        pending = PendingAction(
            tool_name="dangerous_node",
            call_id="abc123",
            args={"node_id": "dangerous_node"},
            status="pending",
        )
        exc = FlowPaused(pending=pending, thread_id="session-42")
        assert exc.pending is pending
        assert exc.thread_id == "session-42"
        assert "dangerous_node" in str(exc)
        assert "session-42" in str(exc)

    def test_inherits_from_loomable_error(self):
        """FlowPaused is a LoomableError."""
        from loomable.kernel.errors import LoomableError

        pending = PendingAction(tool_name="x", call_id="y")
        exc = FlowPaused(pending=pending, thread_id="t")
        assert isinstance(exc, LoomableError)

    def test_reuses_existing_pending_action(self):
        """FlowPaused reuses loomable.persist.checkpoint.PendingAction (Req 16.4)."""
        pending = PendingAction(
            tool_name="node_b",
            call_id="call-001",
            args={"node_id": "node_b"},
            status="pending",
        )
        exc = FlowPaused(pending=pending, thread_id="thread-1")
        assert exc.pending.tool_name == "node_b"
        assert exc.pending.call_id == "call-001"
        assert exc.pending.status == "pending"


# ---------------------------------------------------------------------------
# Pause before require_confirmation node (Req 16.1)
# ---------------------------------------------------------------------------


class TestHITLPause:
    """Verify the engine pauses before a require_confirmation node."""

    @pytest.mark.asyncio
    async def test_raises_flow_paused_before_confirmation_node(self):
        """Engine raises FlowPaused before executing a require_confirmation node (Req 16.1)."""
        node_a = TrackingRunnable("A")
        node_b = TrackingRunnable("B")  # requires confirmation
        node_c = TrackingRunnable("C")

        flow = Flow(
            {"a": Node(node_id="a", runnable=node_a),
             "b": Node(node_id="b", runnable=node_b, require_confirmation=True),
             "c": Node(node_id="c", runnable=node_c)},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
            engine="sequential",
        )

        engine = SequentialEngine()
        state = SharedState()
        ctx = RunContext()

        with pytest.raises(FlowPaused) as exc_info:
            await engine.run(flow, "input", state, ctx, session_id="s1")

        # Node A should have executed
        assert node_a.call_count == 1
        # Node B should NOT have executed (paused before)
        assert node_b.call_count == 0
        # Node C should NOT have executed
        assert node_c.call_count == 0

        # The exception carries correct info
        paused = exc_info.value
        assert paused.pending.tool_name == "b"
        assert paused.pending.status == "pending"
        assert paused.thread_id == "s1"

    @pytest.mark.asyncio
    async def test_flow_paused_carries_correct_pending_action(self):
        """The PendingAction in FlowPaused has the node's info (Req 16.1)."""
        node_risky = TrackingRunnable("risky")

        flow = Flow(
            {"risky": Node(node_id="risky", runnable=node_risky, require_confirmation=True)},
            engine="sequential",
        )

        engine = SequentialEngine()
        state = SharedState()
        ctx = RunContext()

        with pytest.raises(FlowPaused) as exc_info:
            await engine.run(flow, "hello", state, ctx, session_id="my-thread")

        paused = exc_info.value
        assert paused.pending.tool_name == "risky"
        assert paused.pending.args == {"node_id": "risky"}
        assert paused.pending.call_id  # non-empty UUID
        assert paused.thread_id == "my-thread"


# ---------------------------------------------------------------------------
# Resume with approval — node executes (Req 16.3)
# ---------------------------------------------------------------------------


class TestHITLResumeApproved:
    """Verify that resuming with approval executes the paused node."""

    @pytest.mark.asyncio
    async def test_approved_node_executes(self):
        """Approved node runs and flow continues past it (Req 16.3)."""
        node_a = TrackingRunnable("A")
        node_b = TrackingRunnable("B")  # requires confirmation
        node_c = TrackingRunnable("C")

        flow = Flow(
            {"a": Node(node_id="a", runnable=node_a),
             "b": Node(node_id="b", runnable=node_b, require_confirmation=True),
             "c": Node(node_id="c", runnable=node_c)},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
            engine="sequential",
        )

        engine = SequentialEngine()
        state = SharedState()
        # Pre-populate state with A's output (simulating resumed state)
        state.write("a", _make_output("A:executed"))
        ctx = RunContext()

        # Resume with 'a' already completed and 'b' approved
        result = await engine.run(
            flow, "input", state, ctx,
            completed_node_ids={"a"},
            pending_decisions={"b": "approved"},
            session_id="s1",
        )

        # Node A should NOT be re-run (already completed)
        assert node_a.call_count == 0
        # Node B should execute (approved)
        assert node_b.call_count == 1
        # Node C should execute (downstream)
        assert node_c.call_count == 1

    @pytest.mark.asyncio
    async def test_approved_does_not_rerun_completed_nodes(self):
        """Resuming with approval skips already-completed nodes (Req 16.3)."""
        node_a = TrackingRunnable("A")
        node_b = TrackingRunnable("B")
        node_c = TrackingRunnable("C")

        flow = Flow(
            {"a": Node(node_id="a", runnable=node_a),
             "b": Node(node_id="b", runnable=node_b, require_confirmation=True),
             "c": Node(node_id="c", runnable=node_c)},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
            engine="sequential",
        )

        engine = SequentialEngine()
        state = SharedState()
        state.write("a", _make_output("A:executed"))
        ctx = RunContext()

        await engine.run(
            flow, "input", state, ctx,
            completed_node_ids={"a"},
            pending_decisions={"b": "approved"},
        )

        # Only B and C should have run
        assert node_a.call_count == 0
        assert node_b.call_count == 1
        assert node_c.call_count == 1


# ---------------------------------------------------------------------------
# Resume with rejection — node is skipped (Req 16.3)
# ---------------------------------------------------------------------------


class TestHITLResumeRejected:
    """Verify that resuming with rejection skips the paused node."""

    @pytest.mark.asyncio
    async def test_rejected_node_is_skipped(self):
        """Rejected node is skipped, and downstream continues (Req 16.3)."""
        node_a = TrackingRunnable("A")
        node_b = TrackingRunnable("B")  # requires confirmation
        node_c = TrackingRunnable("C")

        flow = Flow(
            {"a": Node(node_id="a", runnable=node_a),
             "b": Node(node_id="b", runnable=node_b, require_confirmation=True),
             "c": Node(node_id="c", runnable=node_c)},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
            engine="sequential",
        )

        engine = SequentialEngine()
        state = SharedState()
        state.write("a", _make_output("A:executed"))
        ctx = RunContext()

        result = await engine.run(
            flow, "input", state, ctx,
            completed_node_ids={"a"},
            pending_decisions={"b": "rejected"},
            session_id="s1",
        )

        # Node A should NOT re-run
        assert node_a.call_count == 0
        # Node B should be SKIPPED (rejected)
        assert node_b.call_count == 0
        # Node C should still execute (downstream of B)
        assert node_c.call_count == 1

    @pytest.mark.asyncio
    async def test_rejected_does_not_rerun_completed_nodes(self):
        """Rejection also does not re-run previously completed nodes."""
        node_a = TrackingRunnable("A")
        node_b = TrackingRunnable("B")

        flow = Flow(
            {"a": Node(node_id="a", runnable=node_a),
             "b": Node(node_id="b", runnable=node_b, require_confirmation=True)},
            edges=[Edge(source="a", target="b")],
            engine="sequential",
        )

        engine = SequentialEngine()
        state = SharedState()
        state.write("a", _make_output("A:done"))
        ctx = RunContext()

        await engine.run(
            flow, "input", state, ctx,
            completed_node_ids={"a"},
            pending_decisions={"b": "rejected"},
        )

        assert node_a.call_count == 0
        assert node_b.call_count == 0


# ---------------------------------------------------------------------------
# No confirmation nodes — runs to completion (Req 16.5)
# ---------------------------------------------------------------------------


class TestHITLNoConfirmation:
    """Verify that flows without confirmation nodes run without pausing."""

    @pytest.mark.asyncio
    async def test_no_confirmation_runs_to_completion(self):
        """Flow with no require_confirmation nodes never pauses (Req 16.5)."""
        node_a = TrackingRunnable("A")
        node_b = TrackingRunnable("B")
        node_c = TrackingRunnable("C")

        flow = Flow(
            {"a": Node(node_id="a", runnable=node_a),
             "b": Node(node_id="b", runnable=node_b),
             "c": Node(node_id="c", runnable=node_c)},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
            engine="sequential",
        )

        engine = SequentialEngine()
        state = SharedState()
        ctx = RunContext()

        # Should NOT raise FlowPaused
        result = await engine.run(flow, "input", state, ctx)

        assert node_a.call_count == 1
        assert node_b.call_count == 1
        assert node_c.call_count == 1

    @pytest.mark.asyncio
    async def test_flow_arun_no_confirmation_completes(self):
        """Flow.arun with no confirmation nodes returns normally (Req 16.5)."""
        call_log = []

        async def step_a(input):  # noqa: A002
            call_log.append("a")
            return f"a:{input}"

        async def step_b(input):  # noqa: A002
            call_log.append("b")
            return f"b:{input}"

        flow = Flow([step_a, step_b], engine="sequential")
        result = await flow.arun("hello")

        assert call_log == ["a", "b"]
        assert result.output is not None


# ---------------------------------------------------------------------------
# Checkpoint written on HITL pause (Req 16.2)
# ---------------------------------------------------------------------------


class TestHITLCheckpoint:
    """Verify checkpoint is written when FlowPaused is raised."""

    @pytest.mark.asyncio
    async def test_checkpoint_written_on_pause(self):
        """Checkpoint with pending action is written before FlowPaused (Req 16.2)."""
        checkpointer = InMemoryCheckpointer()

        node_a = TrackingRunnable("A")
        node_b = TrackingRunnable("B")  # requires confirmation

        flow = Flow(
            {"a": Node(node_id="a", runnable=node_a),
             "b": Node(node_id="b", runnable=node_b, require_confirmation=True)},
            edges=[Edge(source="a", target="b")],
            engine="sequential",
        )

        engine = SequentialEngine()
        state = SharedState()
        ctx = RunContext()

        with pytest.raises(FlowPaused):
            await engine.run(
                flow, "input", state, ctx,
                checkpointer=checkpointer, session_id="hitl-session",
            )

        # A checkpoint should have been written
        cps = await checkpointer.list("hitl-session")
        # At least 2: one for node 'a' completing, one for the HITL pause
        assert len(cps) >= 2

        # The last checkpoint should contain the pending action
        hitl_cp = cps[-1]
        assert len(hitl_cp.pending) == 1
        assert hitl_cp.pending[0].tool_name == "b"
        assert hitl_cp.pending[0].status == "pending"
        assert hitl_cp.complete is False

        # Completed nodes in the checkpoint should include 'a'
        assert "a" in hitl_cp.session_state["completed_node_ids"]

    @pytest.mark.asyncio
    async def test_checkpoint_contains_state_snapshot_on_pause(self):
        """The HITL checkpoint includes the SharedState snapshot (Req 16.2)."""
        checkpointer = InMemoryCheckpointer()

        node_a = TrackingRunnable("A")
        node_b = TrackingRunnable("B")

        flow = Flow(
            {"a": Node(node_id="a", runnable=node_a),
             "b": Node(node_id="b", runnable=node_b, require_confirmation=True)},
            edges=[Edge(source="a", target="b")],
            engine="sequential",
        )

        engine = SequentialEngine()
        state = SharedState()
        ctx = RunContext()

        with pytest.raises(FlowPaused):
            await engine.run(
                flow, "input", state, ctx,
                checkpointer=checkpointer, session_id="state-check",
            )

        cp = await checkpointer.get("state-check")
        assert "shared_state" in cp.session_state
        # Node A's output should be in the state snapshot
        assert "a" in cp.session_state["shared_state"]


# ---------------------------------------------------------------------------
# Full Flow.arun HITL integration (Req 16.1–16.5)
# ---------------------------------------------------------------------------


class TestFlowHITLIntegration:
    """End-to-end HITL pause and resume through Flow.arun."""

    @pytest.mark.asyncio
    async def test_flow_arun_pauses_at_confirmation_node(self):
        """Flow.arun raises FlowPaused at a require_confirmation node."""
        checkpointer = InMemoryCheckpointer()

        node_a = TrackingRunnable("A")
        node_b = TrackingRunnable("B")

        flow = Flow(
            {"a": Node(node_id="a", runnable=node_a),
             "b": Node(node_id="b", runnable=node_b, require_confirmation=True)},
            edges=[Edge(source="a", target="b")],
            checkpointer=checkpointer,
            session_id="flow-hitl",
            engine="sequential",
        )

        with pytest.raises(FlowPaused) as exc_info:
            await flow.arun("start")

        assert exc_info.value.pending.tool_name == "b"
        assert exc_info.value.thread_id == "flow-hitl"
        assert node_a.call_count == 1
        assert node_b.call_count == 0

    @pytest.mark.asyncio
    async def test_flow_arun_resume_approved(self):
        """Flow.arun resumes correctly when the pending action is approved."""
        checkpointer = InMemoryCheckpointer()

        call_log = []

        async def step_a(input):  # noqa: A002
            call_log.append("a")
            return f"a:{input}"

        async def step_b(input):  # noqa: A002
            call_log.append("b")
            return f"b:{input}"

        # Build a flow using Node objects for require_confirmation
        from loomable.flow.runnable import FunctionRunnable

        runnable_a = FunctionRunnable(step_a)
        runnable_b = FunctionRunnable(step_b)

        flow = Flow(
            {"step_a": Node(node_id="step_a", runnable=runnable_a),
             "step_b": Node(node_id="step_b", runnable=runnable_b, require_confirmation=True)},
            edges=[Edge(source="step_a", target="step_b")],
            checkpointer=checkpointer,
            session_id="resume-approved",
            engine="sequential",
        )

        # First run — should pause
        with pytest.raises(FlowPaused):
            await flow.arun("hello")

        assert call_log == ["a"]

        # Simulate approval: update the pending action in the checkpoint
        cp = await checkpointer.get("resume-approved")
        assert cp is not None
        cp.pending[0].status = "approved"
        await checkpointer.put(cp)

        # Second run — should resume and execute step_b
        call_log.clear()
        result = await flow.arun("hello")

        # step_a should NOT re-run (completed in checkpoint)
        assert "a" not in call_log
        # step_b should execute (approved)
        assert "b" in call_log

    @pytest.mark.asyncio
    async def test_flow_arun_resume_rejected(self):
        """Flow.arun resumes correctly when the pending action is rejected."""
        checkpointer = InMemoryCheckpointer()

        call_log = []

        async def step_a(input):  # noqa: A002
            call_log.append("a")
            return f"a:{input}"

        async def step_b(input):  # noqa: A002
            call_log.append("b")
            return f"b:{input}"

        async def step_c(input):  # noqa: A002
            call_log.append("c")
            return f"c:{input}"

        from loomable.flow.runnable import FunctionRunnable

        runnable_a = FunctionRunnable(step_a)
        runnable_b = FunctionRunnable(step_b)
        runnable_c = FunctionRunnable(step_c)

        flow = Flow(
            {"step_a": Node(node_id="step_a", runnable=runnable_a),
             "step_b": Node(node_id="step_b", runnable=runnable_b, require_confirmation=True),
             "step_c": Node(node_id="step_c", runnable=runnable_c)},
            edges=[
                Edge(source="step_a", target="step_b"),
                Edge(source="step_b", target="step_c"),
            ],
            checkpointer=checkpointer,
            session_id="resume-rejected",
            engine="sequential",
        )

        # First run — should pause at step_b
        with pytest.raises(FlowPaused):
            await flow.arun("hello")

        assert call_log == ["a"]

        # Simulate rejection
        cp = await checkpointer.get("resume-rejected")
        cp.pending[0].status = "rejected"
        await checkpointer.put(cp)

        # Second run — step_b skipped, step_c runs
        call_log.clear()
        result = await flow.arun("hello")

        assert "a" not in call_log  # not re-run
        assert "b" not in call_log  # rejected/skipped
        assert "c" in call_log      # downstream continues
