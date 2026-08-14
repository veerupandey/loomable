"""Tests for event-driven checkpoint listener and durable HITL."""

from __future__ import annotations

import asyncio

import pytest

from loomable.agent.events import Event, JSONTracer, NoOpEvents
from loomable.persist.checkpoint import (
    Checkpoint,
    CheckpointConfig,
    JsonFileCheckpointer,
    PendingAction,
    SQLiteCheckpointer,
)
from loomable.persist.listener import CheckpointListener


# ---------------------------------------------------------------------------
# CheckpointListener event-driven tests
# ---------------------------------------------------------------------------


class TestCheckpointListener:
    """Test that CheckpointListener auto-checkpoints on configured events."""

    @pytest.fixture
    def sqlite_checkpointer(self):
        return SQLiteCheckpointer(":memory:")

    async def test_checkpoints_on_matching_event(self, sqlite_checkpointer):
        """Listener should write a checkpoint when event kind matches on_events."""
        config = CheckpointConfig(on_events=["run_end"])
        listener = CheckpointListener(
            checkpointer=sqlite_checkpointer,
            config=config,
            thread_id="t1",
            step_fn=lambda: 5,
            session_state_fn=lambda: {"turns": ["hi"]},
        )

        # Emit a non-matching event — should NOT checkpoint
        listener.emit(Event(kind="model_call", t=1.0))
        await asyncio.sleep(0.05)  # let task complete
        assert await sqlite_checkpointer.get("t1") is None

        # Emit a matching event — SHOULD checkpoint
        listener.emit(Event(kind="run_end", t=2.0))
        await asyncio.sleep(0.05)
        cp = await sqlite_checkpointer.get("t1")
        assert cp is not None
        assert cp.step == 5
        assert cp.session_state == {"turns": ["hi"]}
        assert cp.event_kind == "run_end"

    async def test_wildcard_matches_everything(self, sqlite_checkpointer):
        """on_events=["*"] should checkpoint on every event."""
        config = CheckpointConfig(on_events=["*"])
        listener = CheckpointListener(
            checkpointer=sqlite_checkpointer,
            config=config,
            thread_id="t1",
        )

        listener.emit(Event(kind="model_call", t=1.0))
        listener.emit(Event(kind="tool_call", t=2.0))
        listener.emit(Event(kind="run_end", t=3.0))
        await asyncio.sleep(0.05)

        cps = await sqlite_checkpointer.list("t1")
        assert len(cps) == 3

    async def test_forwards_to_inner_emitter(self, sqlite_checkpointer):
        """Events should still flow through to the inner emitter."""
        inner = JSONTracer(stream=None)
        # JSONTracer writes to a stream; use a custom one that doesn't print
        import io
        inner = JSONTracer(stream=io.StringIO())

        listener = CheckpointListener(
            checkpointer=sqlite_checkpointer,
            config=CheckpointConfig(on_events=["run_end"]),
            thread_id="t1",
            inner=inner,
        )

        listener.emit(Event(kind="model_call", t=1.0))
        listener.emit(Event(kind="run_end", t=2.0))

        # Inner tracer should see both events
        assert len(inner.trace) == 2
        assert inner.trace[0].kind == "model_call"

    async def test_pending_actions_checkpointed(self, sqlite_checkpointer):
        """Pending actions should be included in the checkpoint."""
        listener = CheckpointListener(
            checkpointer=sqlite_checkpointer,
            config=CheckpointConfig(on_events=["tool_call"]),
            thread_id="t1",
        )

        # Set pending actions (e.g., tools awaiting approval)
        listener.set_pending([
            PendingAction(tool_name="deploy", call_id="c1", args={"env": "prod"}),
        ])

        listener.emit(Event(kind="tool_call", t=1.0))
        await asyncio.sleep(0.05)

        cp = await sqlite_checkpointer.get("t1")
        assert len(cp.pending) == 1
        assert cp.pending[0].tool_name == "deploy"
        assert cp.pending[0].args == {"env": "prod"}

    async def test_trace_accumulates(self, sqlite_checkpointer):
        """Listener should accumulate a trace list like JSONTracer."""
        listener = CheckpointListener(
            checkpointer=sqlite_checkpointer,
            config=CheckpointConfig(on_events=[]),
            thread_id="t1",
        )

        listener.emit(Event(kind="a", t=1.0))
        listener.emit(Event(kind="b", t=2.0))

        assert len(listener.trace) == 2
        assert listener.trace[0].kind == "a"


# ---------------------------------------------------------------------------
# Durable HITL flow tests
# ---------------------------------------------------------------------------


class TestDurableHITL:
    """Test the durable HITL pattern: pause → checkpoint → resume."""

    async def test_pending_survives_checkpoint_roundtrip(self, tmp_path):
        """PendingActions should survive write → read via file checkpointer."""
        ckpt = JsonFileCheckpointer(location=str(tmp_path / "ck"))

        # Simulate: agent proposes a dangerous tool call, we checkpoint the pending state
        cp = Checkpoint(
            thread_id="run-1",
            step=3,
            session_state={"messages": ["user: deploy to prod"]},
            pending=[
                PendingAction(
                    tool_name="deploy_to_production",
                    call_id="call-abc",
                    args={"service": "api", "version": "2.1.0"},
                    status="pending",
                ),
            ],
            complete=False,  # run is NOT complete — it's paused
            event_kind="hitl_pause",
        )
        await ckpt.put(cp)

        # Simulate: process restarts, we load the checkpoint
        restored = await ckpt.get("run-1")
        assert restored is not None
        assert not restored.complete
        assert len(restored.pending) == 1
        assert restored.pending[0].tool_name == "deploy_to_production"
        assert restored.pending[0].status == "pending"
        assert restored.pending[0].args == {"service": "api", "version": "2.1.0"}

    async def test_approve_and_resume_pattern(self, tmp_path):
        """After approval, update pending status and checkpoint as complete."""
        ckpt = JsonFileCheckpointer(location=str(tmp_path / "ck"))

        # Phase 1: Agent pauses with pending
        cp = Checkpoint(
            thread_id="run-1",
            step=3,
            session_state={},
            pending=[
                PendingAction(tool_name="deploy", call_id="c1", args={}, status="pending"),
            ],
            complete=False,
        )
        await ckpt.put(cp)

        # Phase 2: External process grants approval
        restored = await ckpt.get("run-1")
        restored.pending[0].status = "approved"

        # Phase 3: Agent resumes, executes, and checkpoints as complete
        final = Checkpoint(
            thread_id="run-1",
            step=4,
            session_state={"result": "deployed successfully"},
            pending=[],  # cleared after execution
            complete=True,
        )
        await ckpt.put(final)

        # Verify final state
        latest = await ckpt.get("run-1")
        assert latest.complete
        assert latest.pending == []
        assert latest.step == 4
