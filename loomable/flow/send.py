"""LangGraph-style dynamic fan-out targets for Workflow graphs.

Use :class:`Send` in a chooser ``Command.update`` list value, then fan out with
``Workflow.map_over(..., over="tasks")`` or ``MapNode(over="tasks")``.

Example::

    def classify(state):
        return Command(
            update={
                "tasks": [
                    Send("research", "gather logs"),
                    Send("research", "check metrics"),
                ]
            },
            goto="map",
        )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Send", "send_args"]


@dataclass(frozen=True)
class Send:
    """Dynamic map target — ``node`` names the route arm; ``arg`` is worker input."""

    node: str
    arg: Any

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node, "arg": self.arg}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Send":
        return cls(node=str(raw.get("node") or ""), arg=raw.get("arg"))


def send_args(items: list[Any]) -> list[Any]:
    """Normalize a SharedState list that may contain :class:`Send` instances."""
    out: list[Any] = []
    for item in items:
        if isinstance(item, Send):
            out.append(item.arg)
        elif isinstance(item, dict) and "arg" in item and "node" in item:
            out.append(item["arg"])
        else:
            out.append(item)
    return out
