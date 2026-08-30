"""loomable — enterprise AI agent framework.

High-level imports::

    from loomable import Agent, Team, Workflow, Case, Step, Loop, tool

Progressive disclosure:
  Agent → Team → Workflow (+ Step / Loop / branch / parallel)
  Case for goal + WorkItems board + dispatch + accept
  Flow / engines / edges for advanced graph control

Package tiers: facade (``loomable``) → product (``agent``, ``memory``, ``flow``,
``case``, …) → runtime (``kernel``, ``providers``, ``persist``). There is no
separate ``loomable.core`` — ``kernel`` is the runtime core.
"""

from __future__ import annotations

__version__ = "0.2.0b0"

from loomable.agent import (
    Agent,
    BuiltAgent,
    ContextPolicy,
    RunResult,
    Team,
    create_deep_agent,
    spawn_specialist,
    tool,
)
from loomable.memory import (
    ConversationMemory,
    KnowledgeMemory,
    Memory,
    MemoryScope,
    UserMemory,
    WorkingMemory,
    open_session_store,
)
from loomable.providers.vector_store import open_vector_store
from loomable.skills import list_bundled_skills, resolve_skills
from loomable.case import (
    Board,
    Case,
    WorkItem,
    build_case_workflow,
    map_specialists,
)
from loomable.flow import (
    Command,
    Condition,
    Flow,
    FlowPaused,
    Loop,
    Parallel_Group,
    Route,
    Send,
    Step,
    StepFailed,
    VerdictResult,
    Verifier,
    Workflow,
    plan_and_execute,
)
from loomable.persist import (
    InMemoryCheckpointer,
    JsonFileCheckpointer,
    PostgresCheckpointer,
    SQLiteCheckpointer,
)

__all__ = [
    "__version__",
    "Agent",
    "BuiltAgent",
    "Team",
    "Workflow",
    "Case",
    "Board",
    "WorkItem",
    "Step",
    "StepFailed",
    "Command",
    "Send",
    "Route",
    "Loop",
    "Condition",
    "Parallel_Group",
    "tool",
    "RunResult",
    "ContextPolicy",
    "spawn_specialist",
    "create_deep_agent",
    "resolve_skills",
    "list_bundled_skills",
    "Memory",
    "MemoryScope",
    "ConversationMemory",
    "UserMemory",
    "KnowledgeMemory",
    "WorkingMemory",
    "open_session_store",
    "open_vector_store",
    "FlowPaused",
    "build_case_workflow",
    "map_specialists",
    "Verifier",
    "VerdictResult",
    "plan_and_execute",
    "JsonFileCheckpointer",
    "SQLiteCheckpointer",
    "InMemoryCheckpointer",
    "PostgresCheckpointer",
    "Flow",
]
