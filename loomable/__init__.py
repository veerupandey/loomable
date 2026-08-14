"""loomable — enterprise AI agent framework.

High-level imports::

    from loomable import Agent, Team, Workflow, Case, Step, Loop, tool

Progressive disclosure:
  Agent → Team → Workflow (+ Step / Loop / branch / parallel)
  Case for goal + WorkItems board + dispatch + accept
  Flow / engines / edges for advanced graph control
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
    create_research_agent,
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
    "Loop",
    "Condition",
    "Parallel_Group",
    "tool",
    "RunResult",
    "ContextPolicy",
    "spawn_specialist",
    "create_deep_agent",
    "create_research_agent",
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
    "sequential",
    "parallel",
    "route",
    "coordinate",
    "plan_and_execute",
    "JsonFileCheckpointer",
    "SQLiteCheckpointer",
    "InMemoryCheckpointer",
    "PostgresCheckpointer",
    "Flow",
]
