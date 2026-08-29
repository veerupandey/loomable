"""Command — control-plane return value for graph nodes (LangGraph-style).

A step or chooser may return a :class:`Command` to combine state updates with
routing, instead of relying only on ambient SharedState chaining::

    async def classify(change, *, context=None):
        severity = score(change)
        if severity == "high":
            return Command(goto="full_audit", update={"severity": severity})
        return Command(goto="quick_path", update={"severity": severity})

    wf = Workflow("review").route(classify, quick_path=quick, full_audit=full)

``update`` is merged into SharedState (respecting Workflow reducers).
``goto`` selects the next route target (used by ``Workflow.route`` / RouterNode).
``resume`` is reserved for HITL resume payloads (passed to ``arun``).
"""

from __future__ import annotations

__all__ = ["Command"]

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Command:
    """Combine a state patch with optional routing / resume control.

    Parameters
    ----------
    update:
        Key/value patches written into SharedState after the node runs.
    goto:
        Target node id or route choice name. For ``Workflow.route``, this is
        the selected branch. For sequential graphs, engines may skip ahead
        when the target appears later in topological order.
    resume:
        Value supplied when continuing after an interrupt / HITL pause.
    """

    update: dict[str, Any] = field(default_factory=dict)
    goto: str | list[str] | None = None
    resume: Any = None

    def to_metadata(self) -> dict[str, Any]:
        """Serialize into RunResult.metadata under ``command``."""
        payload: dict[str, Any] = {}
        if self.update:
            payload["update"] = dict(self.update)
        if self.goto is not None:
            payload["goto"] = self.goto
        if self.resume is not None:
            payload["resume"] = self.resume
        return {"command": payload}

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> "Command | None":
        """Rebuild a Command from RunResult.metadata if present."""
        if not metadata:
            return None
        raw = metadata.get("command")
        if not isinstance(raw, dict):
            return None
        return cls(
            update=dict(raw.get("update") or {}),
            goto=raw.get("goto"),
            resume=raw.get("resume"),
        )
