"""LangGraph-style dynamic fan-out targets for Workflow graphs.

Use :class:`Send` in a chooser ``Command.update`` list value, then fan out with
``Workflow.map_over(..., over="tasks")`` or ``MapNode(over="tasks")``.

``Send.node`` is **metadata only** (route-arm label for logging/inspection).
Only ``Send.arg`` is passed to the worker — multi-arm routing by ``node`` is
not implemented in this release.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

__all__ = ["Send", "send_args"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Send:
    """Dynamic map target — ``node`` is metadata; ``arg`` is worker input."""

    node: str
    arg: Any

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node, "arg": self.arg}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Send":
        return cls(node=str(raw.get("node") or ""), arg=raw.get("arg"))


def send_args(items: list[Any]) -> list[Any]:
    """Normalize a SharedState list that may contain :class:`Send` instances."""
    nodes: set[str] = set()
    out: list[Any] = []
    for item in items:
        if isinstance(item, Send):
            if item.node:
                nodes.add(item.node)
            out.append(item.arg)
        elif isinstance(item, dict) and "arg" in item and "node" in item:
            node = str(item.get("node") or "")
            if node:
                nodes.add(node)
            out.append(item["arg"])
        else:
            out.append(item)
    if len(nodes) > 1:
        logger.warning(
            "Send.node values %s differ within one batch; only Send.arg is used "
            "(multi-arm routing not implemented)",
            sorted(nodes),
        )
    return out
