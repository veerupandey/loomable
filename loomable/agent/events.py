"""Structured observability for the agent run path.

Provides a lightweight, zero-dependency event protocol and two in-box
implementations:

- ``NoOpEvents`` — default, zero-overhead (emit is a no-op).
- ``JSONTracer`` — appends one JSON line per event to a stream and accumulates
  a step-by-step ``trace`` list for attachment to ``RunResult``.

``Event.attributes`` keys use OpenTelemetry GenAI semantic-convention names so
downstream adapters can forward them without translation. Core has no OTel
dependency; names are mapped, not imported.

OTel GenAI attribute key reference (used in attributes dict):
- ``gen_ai.request.model``         — model identifier
- ``gen_ai.usage.input_tokens``    — prompt/input tokens
- ``gen_ai.usage.output_tokens``   — completion/output tokens
- ``gen_ai.tool.name``             — tool name
- ``gen_ai.operation.name``        — high-level operation (e.g. "chat")
"""

from __future__ import annotations

__all__ = ["Event", "AgentEvents", "NoOpEvents", "JSONTracer"]

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import IO, Protocol


@dataclass
class Event:
    """A single observability event emitted during a run.

    Parameters
    ----------
    kind:
        One of ``run_start``, ``model_call``, ``tool_call``, ``compaction``,
        ``tier_substitution``, ``loop_stop``, ``run_end``.
    t:
        Monotonic timestamp (from ``time.monotonic()``).
    duration_ms:
        Duration in milliseconds (for events that span a period).
    tokens_in:
        Input/prompt token count (when available).
    tokens_out:
        Output/completion token count (when available).
    attributes:
        Arbitrary metadata; keys follow OTel GenAI semantic conventions
        (e.g. ``gen_ai.request.model``, ``gen_ai.tool.name``).
    """

    kind: str
    t: float
    duration_ms: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    attributes: dict = field(default_factory=dict)


class AgentEvents(Protocol):
    """Protocol for event receivers.

    Any object with an ``emit(event: Event) -> None`` method satisfies
    this protocol.
    """

    def emit(self, event: Event) -> None: ...


class NoOpEvents:
    """Default event emitter — zero overhead.

    Used when no tracer is configured so the run path pays no recording cost.
    """

    __slots__ = ()

    def emit(self, event: Event) -> None:  # noqa: D102
        pass


class JSONTracer:
    """In-box recording tracer.

    Appends one JSON line per event to a stream (defaults to ``sys.stdout``)
    and accumulates a step-by-step ``trace`` list exposed as a property for
    attachment to ``RunResult``.
    """

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream: IO[str] = stream if stream is not None else sys.stdout
        self._trace: list[Event] = []

    def emit(self, event: Event) -> None:
        """Record the event and write a JSON line to the stream."""
        self._trace.append(event)
        line = json.dumps(asdict(event), default=str)
        self._stream.write(line + "\n")

    @property
    def trace(self) -> list[Event]:
        """All accumulated events in emission order."""
        return list(self._trace)
