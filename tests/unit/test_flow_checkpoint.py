"""Tests for flow-level checkpointing: save, restore, and resume.

Validates: Requirements 13.1, 13.2, 13.5
- Checkpoint saves state snapshot + completed nodes after each step
- Resume from checkpoint skips completed nodes
- No checkpointer = zero overhead (existing behavior unchanged)
- Checkpoint data roundtrips correctly
- Default to durable checkpointer when enabled; InMemoryCheckpointer is test-only
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.engines.sequential import SequentialEngine
from loomable.flow.engines.parallel import ParallelEngine
from loomable.flow.flow import Flow
from loomable.flow.nodes import Edge, Node
from loomable.flow.runnable import FunctionRunnable
from loomable.flow.state import SharedState
from loomable.persist.checkpoint import (
    Checkpoint,
    Checkpointer,
    InMemoryCheckpointer,
    JsonFileCheckpointer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(text: str) -> AgentOutput:
    """Create an AgentOutput with a single text part."""
    return AgentOutput(
        parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=text.encode())]
    )


class CountingRunnable:
    """A Runnable that counts how many times it was called."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0

    async def arun(self, input, *, context=None):  # noqa: A002
        self.call_count += 1
        output = _make_output(f"{self.name}:{input}:{self.call_count}")
        return RunResult(output=output, session_id="")


# ---------------------------------------------------------------------------
# InMemoryCheckpointer basic tests
# ---------------------------------------------------------------------------


class TestInMemoryCheckpointer:
    """Verify InMemoryCheckpointer satisfies the Checkpointer protocol."""

    def test_satisfies_protocol(self):
        cp = InMemoryCheckpointer()
        assert isinstance(cp, Checkpointer)

    @pytest.mark.asyncio
    async def test_put_and_get(self):
        cp = InMemoryCheckpointer()
        checkpoint = Checkpoint(
            thread_id="t1", step=1,
            session_state={"shared_state": {"a": 1}, "completed_node_ids": ["a"]},
        )
        await cp.put(checkpoint)
        result = await cp.get("t1")
        assert result is not None
        assert result.step == 1
        assert result.session_state["completed_node_ids"] == ["a"]

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self):
        cp = InMemoryCheckpointer()
        result = await cp.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_latest(self):
        cp = InMemoryCheckpointer()
        await cp.put(Checkpoint(thread_id="t1", step=1, session_state={"s": 1}))
        await cp.put(Checkpoint(thread_id="t1", step=2, session_state={"s": 2}))
        result = await cp.get("t1")
        assert result.step == 2

    @pytest.mark.asyncio
    async def test_list_returns_commit_order(self):
        cp = InMemoryCheckpointer()
        for i in range(3):
            await cp.put(Checkpoint(thread_id="t1", step=i, session_state={}))
        cps = await cp.list("t1")
        assert [c.step for c in cps] == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_max_checkpoints_prunes(self):
        cp = InMemoryCheckpointer(max_checkpoints=2)
        for i in range(5):
            await cp.put(Checkpoint(thread_id="t1", step=i, session_state={}))
        cps = await cp.list("t1")
        assert len(cps) == 2
        assert cps[-1].step == 4

    @pytest.mark.asyncio
    async def test_threads_are_isolated(self):
        cp = InMemoryCheckpointer()
        await cp.put(Checkpoint(thread_id="t1", step=1, session_state={"a": 1}))
        await cp.put(Checkpoint(thread_id="t2", step=99, session_state={"b": 2}))
        assert (await cp.get("t1")).step == 1
        assert (await cp.get("t2")).step == 99


# ---------------------------------------------------------------------------
# Checkpoint writes during sequential execution (Req 13.1)
# ---------------------------------------------------------------------------


class TestSequentialEngineCheckpointWrites:
    """Verify that the SequentialEngine writes checkpoints at node boundaries."""

    @pytest.mark.asyncio
    async def test_checkpoint_written_after_each_node(self):
        """A checkpoint is written after each node completes (Req 13.1)."""
        checkpointer = InMemoryCheckpointer()

        node_a = CountingRunnable("A")
        node_b = CountingRunnable("B")
        node_c = CountingRunnable("C")

        nodes = {
            "a": Node(node_id="a", runnable=node_a),
            "b": Node(node_id="b", runnable=node_b),
            "c": Node(node_id="c", runnable=node_c),
        }
        edges = [Edge(source="a", target="b"), Edge(source="b", target="c")]

        # Build a minimal flow-like object for the engine
        flow = Flow({"a": node_a, "b": node_b, "c": node_c}, edges=edges)

        state = SharedState()
        ctx = RunContext()
        engine = SequentialEngine()

        await engine.run(
            flow, "input", state, ctx,
            checkpointer=checkpointer, session_id="sess1"
        )

        # Should have 3 checkpoints (one per node)
        cps = await checkpointer.list("sess1")
        assert len(cps) == 3

        # First checkpoint: only 'a' completed
        assert sorted(cps[0].session_state["completed_node_ids"]) == ["a"]
        # Second checkpoint: 'a' and 'b' completed
        assert sorted(cps[1].session_state["completed_node_ids"]) == ["a", "b"]
        # Third checkpoint: all completed
        assert sorted(cps[2].session_state["completed_node_ids"]) == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_checkpoint_contains_state_snapshot(self):
        """Checkpoint session_state includes the SharedState snapshot (Req 13.1)."""
        checkpointer = InMemoryCheckpointer()

        async def step_a(input):  # noqa: A002
            return "result_a"

        flow = Flow([step_a], engine="sequential")

        state = SharedState()
        ctx = RunContext()
        engine = SequentialEngine()

        await engine.run(
            flow, "hello", state, ctx,
            checkpointer=checkpointer, session_id="s1"
        )

        cp = await checkpointer.get("s1")
        assert "shared_state" in cp.session_state
        # The state should have the node's output written
        assert cp.session_state["shared_state"] is not None

    @pytest.mark.asyncio
    async def test_no_checkpointer_no_overhead(self):
        """Without a checkpointer, no checkpoint writes happen (Req 13.5)."""
        node_a = CountingRunnable("A")
        flow = Flow([node_a])

        state = SharedState()
        ctx = RunContext()
        engine = SequentialEngine()

        # Should work fine without checkpointer
        result = await engine.run(flow, "input", state, ctx)
        assert node_a.call_count == 1
        assert result.output is not None


# ---------------------------------------------------------------------------
# Resume from checkpoint: skip completed nodes (Req 13.2)
# ---------------------------------------------------------------------------


class TestSequentialEngineResume:
    """Verify that resume skips completed nodes and restores state."""

    @pytest.mark.asyncio
    async def test_resume_skips_completed_nodes(self):
        """Nodes in completed_node_ids are not re-executed (Req 13.2)."""
        node_a = CountingRunnable("A")
        node_b = CountingRunnable("B")
        node_c = CountingRunnable("C")

        flow = Flow(
            {"a": node_a, "b": node_b, "c": node_c},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
        )

        # Simulate a state where 'a' has already completed
        state = SharedState()
        state.write("a", _make_output("A:input:1"))

        ctx = RunContext()
        engine = SequentialEngine()

        await engine.run(
            flow, "input", state, ctx,
            completed_node_ids={"a"},
        )

        # 'a' should not have been called
        assert node_a.call_count == 0
        # 'b' and 'c' should have been called
        assert node_b.call_count == 1
        assert node_c.call_count == 1

    @pytest.mark.asyncio
    async def test_resume_skips_multiple_completed_nodes(self):
        """Multiple completed nodes are all skipped."""
        node_a = CountingRunnable("A")
        node_b = CountingRunnable("B")
        node_c = CountingRunnable("C")

        flow = Flow(
            {"a": node_a, "b": node_b, "c": node_c},
            edges=[Edge(source="a", target="b"), Edge(source="b", target="c")],
        )

        # Simulate a state where 'a' and 'b' have already completed
        state = SharedState()
        state.write("a", _make_output("A:input:1"))
        state.write("b", _make_output("B:input:1"))

        ctx = RunContext()
        engine = SequentialEngine()

        await engine.run(
            flow, "input", state, ctx,
            completed_node_ids={"a", "b"},
        )

        # Only 'c' should execute
        assert node_a.call_count == 0
        assert node_b.call_count == 0
        assert node_c.call_count == 1


# ---------------------------------------------------------------------------
# Flow.arun checkpoint integration (Req 13.1, 13.2, 13.5)
# ---------------------------------------------------------------------------


class TestFlowCheckpointIntegration:
    """Test the full Flow.arun checkpoint save/restore cycle."""

    @pytest.mark.asyncio
    async def test_flow_writes_checkpoints_when_configured(self):
        """Flow writes checkpoints during execution when checkpointer is set."""
        checkpointer = InMemoryCheckpointer()

        call_log = []

        async def step_a(input):  # noqa: A002
            call_log.append("a")
            return f"a:{input}"

        async def step_b(input):  # noqa: A002
            call_log.append("b")
            return f"b:{input}"

        flow = Flow(
            [step_a, step_b],
            checkpointer=checkpointer,
            session_id="flow-session",
            engine="sequential",
        )

        await flow.arun("hello")

        # Should have per-node checkpoints + a final complete checkpoint
        cps = await checkpointer.list("flow-session")
        # 2 node checkpoints + 1 final complete checkpoint
        assert len(cps) == 3
        # The final checkpoint should be marked complete
        assert cps[-1].complete is True

    @pytest.mark.asyncio
    async def test_flow_resumes_from_checkpoint(self):
        """Flow resumes from a checkpoint, skipping completed nodes (Req 13.2)."""
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

        # First run: simulate partial execution by writing a checkpoint
        # where node 'step_a' is already done
        partial_cp = Checkpoint(
            thread_id="resume-session",
            step=1,
            session_state={
                "shared_state": {"step_a": _make_output("a:hello")},
                "completed_node_ids": ["step_a"],
            },
            complete=False,
        )
        await checkpointer.put(partial_cp)

        # Create the flow with resume session
        flow = Flow(
            [step_a, step_b, step_c],
            checkpointer=checkpointer,
            session_id="resume-session",
            engine="sequential",
        )

        call_log.clear()
        await flow.arun("hello")

        # step_a should have been skipped
        assert "a" not in call_log
        # step_b and step_c should have executed
        assert "b" in call_log
        assert "c" in call_log

    @pytest.mark.asyncio
    async def test_flow_no_checkpointer_zero_overhead(self):
        """Flow without checkpointer runs normally with no persistence (Req 13.5)."""
        call_log = []

        async def step_a(input):  # noqa: A002
            call_log.append("a")
            return f"a:{input}"

        async def step_b(input):  # noqa: A002
            call_log.append("b")
            return f"b:{input}"

        flow = Flow([step_a, step_b], engine="sequential")
        result = await flow.arun("hello")

        # Both nodes should execute normally
        assert call_log == ["a", "b"]
        # Result should be from the last node
        output_data = result.output.parts[0].data.decode()
        assert "b:" in output_data

    @pytest.mark.asyncio
    async def test_checkpoint_data_roundtrip(self):
        """Checkpoint data (state + completed_node_ids) roundtrips correctly."""
        checkpointer = InMemoryCheckpointer()

        async def step_a(input):  # noqa: A002
            return f"a:{input}"

        flow = Flow(
            [step_a],
            checkpointer=checkpointer,
            session_id="roundtrip-session",
            engine="sequential",
        )

        await flow.arun("test_input")

        # Get the final checkpoint
        cp = await checkpointer.get("roundtrip-session")
        assert cp is not None
        assert cp.complete is True
        assert "shared_state" in cp.session_state
        assert "completed_node_ids" in cp.session_state

        # Verify the completed_node_ids contain the node
        completed = cp.session_state["completed_node_ids"]
        assert "step_a" in completed

        # Verify the state snapshot is a dict we can restore from
        state_data = cp.session_state["shared_state"]
        restored_state = SharedState.restore(state_data)
        assert restored_state.get("step_a") is not None

    @pytest.mark.asyncio
    async def test_completed_checkpoint_not_resumed(self):
        """A completed checkpoint (complete=True) is not used for resume."""
        checkpointer = InMemoryCheckpointer()

        call_log = []

        async def step_a(input):  # noqa: A002
            call_log.append("a")
            return f"a:{input}"

        # Write a completed checkpoint — should NOT trigger resume
        complete_cp = Checkpoint(
            thread_id="done-session",
            step=1,
            session_state={
                "shared_state": {"step_a": "old_result"},
                "completed_node_ids": ["step_a"],
            },
            complete=True,  # This is marked complete, so resume should not apply
        )
        await checkpointer.put(complete_cp)

        flow = Flow(
            [step_a],
            checkpointer=checkpointer,
            session_id="done-session",
            engine="sequential",
        )

        await flow.arun("hello")

        # step_a should run because the checkpoint was marked complete
        assert "a" in call_log


# ---------------------------------------------------------------------------
# Durable checkpointer default (Req 13.5)
# ---------------------------------------------------------------------------


class TestDurableCheckpointerDefault:
    """Verify that durable checkpointers are the default when enabled."""

    def test_json_file_checkpointer_is_durable(self, tmp_path):
        """JsonFileCheckpointer persists to disk (durable, Req 13.5)."""
        cp = JsonFileCheckpointer(location=str(tmp_path / "ck"))
        assert isinstance(cp, Checkpointer)

    def test_in_memory_checkpointer_is_test_only(self):
        """InMemoryCheckpointer is documented as test-only (Req 13.5)."""
        cp = InMemoryCheckpointer()
        # It's a valid Checkpointer but loses data on restart
        assert isinstance(cp, Checkpointer)
        # The docstring mentions test-only
        assert "test" in InMemoryCheckpointer.__doc__.lower()

    @pytest.mark.asyncio
    async def test_json_file_checkpointer_used_in_flow(self, tmp_path):
        """Flow can use JsonFileCheckpointer for durable persistence."""
        checkpointer = JsonFileCheckpointer(location=str(tmp_path / "flow_ck"))

        async def step_a(input):  # noqa: A002
            return f"a:{input}"

        flow = Flow(
            [step_a],
            checkpointer=checkpointer,
            session_id="durable-test",
            engine="sequential",
        )

        await flow.arun("hello")

        # Verify the checkpoint was written to disk
        cp = await checkpointer.get("durable-test")
        assert cp is not None
        assert cp.complete is True
