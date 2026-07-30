"""Tests for flow observability: node events and context snapshots.

Covers:
- node_start and node_end events carry node_id (Req 13.3)
- context_snapshot records dispositions when enabled (Req 15.1, 15.3)
- disabled = no events emitted, zero overhead (Req 15.4)
- metadata-only mode omits text (Req 15.6)
- node events are emitted by the SequentialEngine during execution
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.events import Event, NoOpEvents
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow import (
    ContextSnapshotConfig,
    Flow,
    MessageDisposition,
    MessageSnapshot,
    emit_context_snapshot,
    emit_node_end,
    emit_node_start,
)
from loomable.flow.runnable import FunctionRunnable, Runnable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingEvents:
    """A test emitter that collects all emitted events."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


def _make_result(text: str = "ok") -> RunResult:
    """Create a minimal RunResult."""
    output = AgentOutput(
        parts=[MediaPart(modality=Modality.TEXT, media_type="text/plain", data=text.encode())]
    )
    return RunResult(output=output, session_id="")


# ---------------------------------------------------------------------------
# Unit tests for emit helpers
# ---------------------------------------------------------------------------


class TestEmitNodeStart:
    def test_emits_event_with_node_id(self) -> None:
        recorder = RecordingEvents()
        t = emit_node_start(recorder, "summarize")

        assert len(recorder.events) == 1
        ev = recorder.events[0]
        assert ev.kind == "node_start"
        assert ev.attributes["node_id"] == "summarize"
        assert t > 0

    def test_no_op_events_does_nothing(self) -> None:
        noop = NoOpEvents()
        # Should not raise
        t = emit_node_start(noop, "test_node")
        assert t > 0


class TestEmitNodeEnd:
    def test_emits_event_with_node_id_and_duration(self) -> None:
        recorder = RecordingEvents()
        start_t = emit_node_start(recorder, "analyze")
        emit_node_end(recorder, "analyze", start_t)

        assert len(recorder.events) == 2
        end_ev = recorder.events[1]
        assert end_ev.kind == "node_end"
        assert end_ev.attributes["node_id"] == "analyze"
        assert end_ev.duration_ms is not None
        assert end_ev.duration_ms >= 0


class TestEmitContextSnapshot:
    def test_disabled_emits_nothing(self) -> None:
        """When config is None or disabled, no event is emitted (Req 15.4)."""
        recorder = RecordingEvents()
        messages = [
            MessageSnapshot(
                role="user", token_estimate=10, disposition=MessageDisposition.ADMITTED
            )
        ]

        # None config
        emit_context_snapshot(recorder, "node_a", messages, config=None)
        assert len(recorder.events) == 0

        # Disabled config
        emit_context_snapshot(
            recorder, "node_a", messages, config=ContextSnapshotConfig(enabled=False)
        )
        assert len(recorder.events) == 0

    def test_enabled_emits_snapshot_with_dispositions(self) -> None:
        """Enabled config emits snapshot with roles, tokens, dispositions (Req 15.1, 15.2, 15.3)."""
        recorder = RecordingEvents()
        messages = [
            MessageSnapshot(
                role="system",
                token_estimate=50,
                disposition=MessageDisposition.ADMITTED,
                text="You are a helper.",
            ),
            MessageSnapshot(
                role="user",
                token_estimate=20,
                disposition=MessageDisposition.ADMITTED,
                text="Hello",
            ),
            MessageSnapshot(
                role="assistant",
                token_estimate=100,
                disposition=MessageDisposition.COMPACTED,
                text="[compacted]",
            ),
        ]

        config = ContextSnapshotConfig(enabled=True, metadata_only=False)
        emit_context_snapshot(recorder, "chat_node", messages, config=config)

        assert len(recorder.events) == 1
        ev = recorder.events[0]
        assert ev.kind == "context_snapshot"
        assert ev.attributes["node_id"] == "chat_node"
        assert ev.attributes["roles"] == ["system", "user", "assistant"]
        assert ev.attributes["token_estimates"] == [50, 20, 100]
        assert ev.attributes["dispositions"] == ["admitted", "admitted", "compacted"]
        assert ev.tokens_in == 170

        # Full mode includes text
        msgs_data = ev.attributes["messages"]
        assert msgs_data[0]["text"] == "You are a helper."
        assert msgs_data[2]["text"] == "[compacted]"

    def test_metadata_only_mode_omits_text(self) -> None:
        """Metadata-only mode records roles/counts/dispositions without text (Req 15.6)."""
        recorder = RecordingEvents()
        messages = [
            MessageSnapshot(
                role="user",
                token_estimate=30,
                disposition=MessageDisposition.EVICTED,
                text="This should be hidden",
            ),
        ]

        config = ContextSnapshotConfig(enabled=True, metadata_only=True)
        emit_context_snapshot(recorder, "node_b", messages, config=config)

        assert len(recorder.events) == 1
        ev = recorder.events[0]
        msg_data = ev.attributes["messages"][0]
        assert "text" not in msg_data
        assert msg_data["role"] == "user"
        assert msg_data["token_estimate"] == 30
        assert msg_data["disposition"] == "evicted"


# ---------------------------------------------------------------------------
# Integration test: SequentialEngine emits node events
# ---------------------------------------------------------------------------


class TestSequentialEngineNodeEvents:
    @pytest.mark.asyncio
    async def test_node_start_and_end_emitted_for_each_node(self) -> None:
        """The engine emits node_start/node_end for each node it executes."""

        async def step_a(inp, *, context=None):
            return "a_result"

        async def step_b(inp, *, context=None):
            return "b_result"

        recorder = RecordingEvents()
        ctx = RunContext(events=recorder)

        flow = Flow([step_a, step_b])
        await flow.arun("start", context=ctx)

        # Should have node_start + node_end for each node (4 events total)
        kinds = [ev.kind for ev in recorder.events]
        assert kinds.count("node_start") == 2
        assert kinds.count("node_end") == 2

        # Verify ordering: start_a, end_a, start_b, end_b
        node_events = [
            (ev.kind, ev.attributes["node_id"])
            for ev in recorder.events
            if ev.kind in ("node_start", "node_end")
        ]
        assert node_events[0] == ("node_start", "step_a")
        assert node_events[1] == ("node_end", "step_a")
        assert node_events[2] == ("node_start", "step_b")
        assert node_events[3] == ("node_end", "step_b")

    @pytest.mark.asyncio
    async def test_no_events_when_noop_emitter(self) -> None:
        """With NoOpEvents (default), no events recorded — zero overhead (Req 15.4)."""

        async def step(inp, *, context=None):
            return "done"

        # Default RunContext uses NoOpEvents
        ctx = RunContext()

        flow = Flow([step])
        result = await flow.arun("input", context=ctx)

        # NoOpEvents doesn't record anything — just verify no error
        assert result is not None

    @pytest.mark.asyncio
    async def test_node_end_has_duration(self) -> None:
        """node_end events include a non-negative duration_ms."""
        import asyncio

        async def slow_step(inp, *, context=None):
            await asyncio.sleep(0.01)
            return "done"

        recorder = RecordingEvents()
        ctx = RunContext(events=recorder)

        flow = Flow([slow_step])
        await flow.arun("go", context=ctx)

        end_events = [ev for ev in recorder.events if ev.kind == "node_end"]
        assert len(end_events) == 1
        assert end_events[0].duration_ms is not None
        assert end_events[0].duration_ms >= 5  # at least ~10ms sleep
