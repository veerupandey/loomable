"""loomable - Enterprise-grade AI agent framework.

High-level imports (preferred)::

    from loomable import Agent, Team, Workflow, ToughTask, Step, Loop, tool

Progressive disclosure:
  Agent → Team → Workflow (+ Step / Loop / branch / parallel)
  ToughTask / plan_act_verify for plan → fan-out → verify
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
from loomable.tough import ToughTask, map_specialists, plan_act_verify

__all__ = [
    "__version__",
    # Core primitives
    "Agent",
    "BuiltAgent",
    "Team",
    "Workflow",
    "Step",
    "Loop",
    "Condition",
    "Parallel_Group",
    "tool",
    "RunResult",
    "ContextPolicy",
    "spawn_specialist",
    "FlowPaused",
    # Tough tasks (plan → fan-out → verify)
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
