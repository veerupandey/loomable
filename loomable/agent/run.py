"""loomable.agent.run - Run result and streaming chunk models.

These are the return shapes of the high-level run flow:

- :class:`RunResult` is what :meth:`BuiltAgent.arun` returns: the produced
  :class:`~loomable.content.AgentOutput` plus run metadata (session id, token
  usage, and tool activity). ``sub_results`` and ``structured`` are placeholders
  for later tasks (orchestration / structured output) and default to ``None``.
- :class:`RunChunk` is what :meth:`BuiltAgent.astream` yields: an incremental
  output part plus a ``done`` flag marking the final chunk (Req 1.5).

Depends only on the standard library and ``loomable.content``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loomable.agent.events import Event
from loomable.content import AgentOutput, MediaPart

if TYPE_CHECKING:
    from loomable.flow.loop import VerdictResult


@dataclass
class RunResult:
    """The result of a single agent run (Req 1.4).

    Attributes
    ----------
    output:
        The produced :class:`~loomable.content.AgentOutput`.
    session_id:
        The session the run belongs to.
    usage:
        Token usage reported by the provider (input/output token counts).
    tool_activity:
        Tool outcomes observed during the run (empty for a plain single turn).
    sub_results:
        Per-sub-agent results for multi-agent orchestration (task 8); ``None`` here.
    structured:
        The parsed/validated structured object for structured output (task 9.1);
        ``None`` here.
    metadata:
        Optional run metadata dict. When tiered routing is active and a fallback
        is used, contains a ``"tier_substitution"`` key with the
        :class:`~loomable.kernel.model_router.TierSubstitution` record (Req 7.1–7.3).
    """

    output: AgentOutput
    session_id: str
    usage: dict[str, int] = field(default_factory=dict)
    tool_activity: list[Any] = field(default_factory=list)
    sub_results: dict[str, Any] | None = None
    structured: object | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: list[Event] = field(default_factory=list)
    verification: "VerdictResult | None" = None


@dataclass
class RunChunk:
    """A single incremental output part yielded while streaming (Req 1.5).

    ``done`` marks the final chunk of a stream.
    """

    delta: MediaPart
    done: bool = False
