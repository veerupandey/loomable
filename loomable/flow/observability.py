"""Node events and ContextSnapshot event helpers.

Emits node_start/node_end events and opt-in context snapshots through the
existing AgentEvents emitter with zero overhead when disabled (Req 13.3, 15.1–15.6).

Usage
-----
The engine calls ``emit_node_start`` before running a node and ``emit_node_end``
after. When ``ContextSnapshotConfig.enabled`` is True, model-calling nodes
additionally emit a ``context_snapshot`` event via ``emit_context_snapshot``.

All events flow through the existing ``AgentEvents`` protocol (Req 15.5). When
the emitter is ``NoOpEvents`` (the default), emit is a no-op and the snapshot
helpers short-circuit immediately (Req 15.4 — zero overhead when disabled).
"""

from __future__ import annotations

__all__ = [
    "ContextSnapshotConfig",
    "MessageDisposition",
    "MessageSnapshot",
    "emit_node_start",
    "emit_node_end",
    "emit_context_snapshot",
]

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from loomable.agent.events import AgentEvents, Event


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ContextSnapshotConfig:
    """Configuration for context-snapshot observability (Req 15.4, 15.6).

    Parameters
    ----------
    enabled:
        Whether context-snapshot events are emitted. When ``False`` (the
        default), no snapshot events are emitted and no recording overhead
        is incurred.
    metadata_only:
        When ``True``, snapshot events include roles, counts, and dispositions
        but omit raw message text (Req 15.6). Useful in production where
        recording raw content is undesirable.
    """

    enabled: bool = False
    metadata_only: bool = False


# ---------------------------------------------------------------------------
# Data models for context snapshots
# ---------------------------------------------------------------------------


class MessageDisposition(str, Enum):
    """Disposition of a message in the assembled context (Req 15.3).

    Indicates whether the context bounder admitted, evicted, or compacted
    the message so silent context loss is observable.
    """

    ADMITTED = "admitted"
    EVICTED = "evicted"
    COMPACTED = "compacted"


@dataclass
class MessageSnapshot:
    """Per-message record within a context snapshot (Req 15.1, 15.2, 15.3).

    Parameters
    ----------
    role:
        The message role (e.g. "system", "user", "assistant", "tool").
    token_estimate:
        Estimated token count for this message.
    disposition:
        Whether the message was admitted, evicted, or compacted by the
        context bounder.
    text:
        The raw message text. ``None`` in metadata-only mode (Req 15.6).
    """

    role: str
    token_estimate: int
    disposition: MessageDisposition
    text: str | None = None


# ---------------------------------------------------------------------------
# Event emission helpers
# ---------------------------------------------------------------------------


def emit_node_start(events: "AgentEvents", node_id: str) -> float:
    """Emit a ``node_start`` event carrying the node_id (Req 13.3).

    Parameters
    ----------
    events:
        The AgentEvents emitter from the RunContext.
    node_id:
        The unique identifier of the node about to execute.

    Returns
    -------
    float
        The monotonic timestamp at emission (used to compute duration in
        ``emit_node_end``).
    """
    from loomable.agent.events import Event

    t = time.monotonic()
    events.emit(
        Event(
            kind="node_start",
            t=t,
            attributes={"node_id": node_id},
        )
    )
    return t


def emit_node_end(
    events: "AgentEvents", node_id: str, start_t: float
) -> None:
    """Emit a ``node_end`` event carrying the node_id and duration (Req 13.3).

    Parameters
    ----------
    events:
        The AgentEvents emitter from the RunContext.
    node_id:
        The unique identifier of the node that just completed.
    start_t:
        The monotonic timestamp returned by the corresponding
        ``emit_node_start`` call.
    """
    from loomable.agent.events import Event

    t = time.monotonic()
    duration_ms = (t - start_t) * 1000.0
    events.emit(
        Event(
            kind="node_end",
            t=t,
            duration_ms=duration_ms,
            attributes={"node_id": node_id},
        )
    )


def emit_context_snapshot(
    events: "AgentEvents",
    node_id: str,
    messages: list[MessageSnapshot],
    *,
    config: ContextSnapshotConfig | None = None,
) -> None:
    """Emit a ``context_snapshot`` event if snapshots are enabled (Req 15.1–15.6).

    When ``config`` is ``None`` or ``config.enabled`` is ``False``, this
    function returns immediately with zero overhead (Req 15.4).

    Parameters
    ----------
    events:
        The AgentEvents emitter from the RunContext.
    node_id:
        The unique identifier of the model-calling node.
    messages:
        The ordered list of message snapshots reflecting what was assembled
        for the model call.
    config:
        The snapshot configuration. ``None`` or disabled → no-op.
    """
    # Zero-overhead guard (Req 15.4)
    if config is None or not config.enabled:
        return

    from loomable.agent.events import Event

    # Build per-message data respecting metadata_only mode (Req 15.6)
    snapshot_messages: list[dict] = []
    for msg in messages:
        entry: dict = {
            "role": msg.role,
            "token_estimate": msg.token_estimate,
            "disposition": msg.disposition.value,
        }
        if not config.metadata_only and msg.text is not None:
            entry["text"] = msg.text
        snapshot_messages.append(entry)

    # Compute summary counts
    roles = [msg.role for msg in messages]
    total_tokens = sum(msg.token_estimate for msg in messages)

    events.emit(
        Event(
            kind="context_snapshot",
            t=time.monotonic(),
            tokens_in=total_tokens,
            attributes={
                "node_id": node_id,
                "roles": roles,
                "token_estimates": [msg.token_estimate for msg in messages],
                "dispositions": [msg.disposition.value for msg in messages],
                "messages": snapshot_messages,
            },
        )
    )
