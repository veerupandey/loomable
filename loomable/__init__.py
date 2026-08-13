"""loomable - Enterprise-grade AI agent framework.

High-level imports (preferred)::

    from loomable import Agent, Team, Workflow, Case, Step, Loop, tool

Progressive disclosure:
  Agent → Team → Workflow (+ Step / Loop / branch / parallel)
  Case / plan_act_verify for goal + WorkItems + dispatch + accept
  Flow / engines / edges remain available for advanced graph control.
"""

from __future__ import annotations

__version__ = "0.1.0"

from loomable.agent import (
    Agent,
    BuiltAgent,
    ContextPolicy,
    RunResult,
    Team,
    spawn_specialist,
    tool,
)
from loomable.case import (
    Board,
    Case,
    ToughTask,
    WorkItem,
    WorkItems,
    map_specialists,
    plan_act_verify,
)
from loomable.flow import (
    Condition,
    Flow,
    FlowPaused,
    Loop,
    Parallel_Group,
    Step,
    VerdictResult,
    Verifier,
    Workflow,
    coordinate,
    parallel,
    plan_and_execute,
    route,
    sequential,
)
from loomable.persist import (
    InMemoryCheckpointer,
    JsonFileCheckpointer,
    SQLiteCheckpointer,
)

__all__ = [
    "__version__",
    # Core primitives
    "Agent",
    "BuiltAgent",
    "Team",
    "Workflow",
    "Case",
    "Board",
    "WorkItem",
    "WorkItems",
    "Step",
    "Loop",
    "Condition",
    "Parallel_Group",
    "tool",
    "RunResult",
    "ContextPolicy",
    "spawn_specialist",
    "FlowPaused",
    # Case (plan → dispatch → accept); ToughTask alias one release
    "ToughTask",
    "plan_act_verify",
    "map_specialists",
    "Verifier",
    "VerdictResult",
    # Helpers (aliases that return Flow)
    "sequential",
    "parallel",
    "route",
    "coordinate",
    "plan_and_execute",
    # Durability
    "JsonFileCheckpointer",
    "SQLiteCheckpointer",
    "InMemoryCheckpointer",
    # Advanced escape hatch
    "Flow",
]
