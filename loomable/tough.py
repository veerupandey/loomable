"""Legacy ToughTask module — re-exports from :mod:`loomable.case`.

Prefer::

    from loomable import Case

``ToughTask``, ``mode="tough"``, ``fan_out``, and ``verify`` remain available
for one release as aliases.
"""

from __future__ import annotations

from loomable.case import (  # noqa: F401
    Board,
    Case,
    ToughTask,
    WorkItem,
    WorkItems,
    board_tools,
    map_specialists,
    parse_plan_steps,
    plan_act_verify,
)

__all__ = [
    "ToughTask",
    "Case",
    "Board",
    "WorkItem",
    "WorkItems",
    "plan_act_verify",
    "map_specialists",
    "parse_plan_steps",
    "board_tools",
]
