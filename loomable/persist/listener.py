"""loomable.persist.listener - Event-driven checkpoint trigger.

A ``CheckpointListener`` subscribes to the ``AgentEvents`` emitter and writes
a checkpoint whenever a matching event fires. This is the wiring that makes
``CheckpointConfig.on_events`` actually work — you configure which events
trigger checkpoints, and this listener does the rest automatically.

Usage:

    from loomable.persist import CheckpointConfig, JsonFileCheckpointer
    from loomable.persist.listener import CheckpointListener

    checkpointer = JsonFileCheckpointer()
    listener = CheckpointListener(
        checkpointer=checkpointer,
        config=CheckpointConfig(on_events=["run_end", "tool_call"]),
        thread_id="session-123",
    )

    # Wire as the events emitter on BuiltAgent
    built_agent.events = listener

The listener wraps any inner ``AgentEvents`` emitter (e.g. a JSONTracer)
so events still flow through for tracing while also triggering checkpoints.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from loomable.agent.events import AgentEvents, Event, NoOpEvents

from .checkpoint import Checkpoint, CheckpointConfig, Checkpointer, PendingAction


class CheckpointListener:
    """An AgentEvents wrapper that auto-checkpoints on configured event kinds.

    Wraps an optional inner emitter so observability tracing continues
    while checkpoints are written on matching events.
    """

    def __init__(
        self,
        checkpointer: Checkpointer,
        config: CheckpointConfig | None = None,
        thread_id: str = "",
        *,
        inner: AgentEvents | None = None,
        session_state_fn: Callable[[], dict[str, Any]] | None = None,
        step_fn: Callable[[], int] | None = None,
    ) -> None:
        """Initialize the listener.

        Parameters
        ----------
        checkpointer:
            The checkpoint store to write to.
        config:
            Checkpoint configuration (which events trigger writes, pruning).
            Defaults to CheckpointConfig() (checkpoint on "run_end").
        thread_id:
            The thread identifier for checkpoints.
        inner:
            Optional wrapped AgentEvents emitter (events are forwarded to it).
        session_state_fn:
            Callable returning the current session state dict for checkpointing.
            If not provided, checkpoints will have empty session_state.
        step_fn:
            Callable returning the current step counter.
        """
        self._checkpointer = checkpointer
        self._config = config or CheckpointConfig()
        self._thread_id = thread_id
        self._inner = inner or NoOpEvents()
        self._session_state_fn = session_state_fn
        self._step_fn = step_fn
        self._pending: list[PendingAction] = []

        # Accumulated trace (mirrors JSONTracer interface)
        self._trace: list[Event] = []

    @property
    def trace(self) -> list[Event]:
        """All accumulated events in emission order."""
        return list(self._trace)

    def set_pending(self, pending: list[PendingAction]) -> None:
        """Update the pending actions (for durable HITL checkpointing)."""
        self._pending = list(pending)

    def clear_pending(self) -> None:
        """Clear all pending actions after they've been resolved."""
        self._pending = []

    def emit(self, event: Event) -> None:
        """Process an event: forward to inner emitter, then checkpoint if matching."""
        # Always forward to inner emitter
        self._inner.emit(event)
        self._trace.append(event)

        # Check if this event kind triggers a checkpoint
        if self._should_checkpoint(event.kind):
            self._write_checkpoint(event.kind)

    def _should_checkpoint(self, event_kind: str) -> bool:
        """Check if an event kind matches the configured triggers."""
        on_events = self._config.on_events
        if "*" in on_events:
            return True
        return event_kind in on_events

    def _write_checkpoint(self, event_kind: str) -> None:
        """Write a checkpoint synchronously (fire-and-forget async put)."""
        session_state = self._session_state_fn() if self._session_state_fn else {}
        step = self._step_fn() if self._step_fn else 0

        cp = Checkpoint(
            thread_id=self._thread_id,
            step=step,
            session_state=session_state,
            pending=list(self._pending),
            event_kind=event_kind,
        )

        # Schedule the async put without blocking emit()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._checkpointer.put(cp))
        except RuntimeError:
            # No running loop — run synchronously
            asyncio.run(self._checkpointer.put(cp))
