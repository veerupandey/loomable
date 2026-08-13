"""loomable.flow - Unified composition model for loomable.

Public high-level API (prefer these)::

    from loomable import Agent, Team, Workflow, Step, Loop

    wf = (
        Workflow("job", session_id="t1", checkpointer=cp)
        .step("research", researcher)
        .parallel(analyst=analyst, writer=writer)
        .branch(when=needs_review, then=reviewer, else_=publisher)
        .step("publish", publisher)
    )

Three primitives:
  - **Agent** — one model + tools + session
  - **Team** — multi-agent LLM-driven coordination
  - **Workflow** — deterministic process (seq / parallel / branch / loop / map)

``Flow``, engines, and ``Edge`` remain available as an advanced escape hatch.
Helpers (``sequential``, ``parallel``, …) are thin aliases that return ``Flow``.
"""

from .engines import (
    ExecutionEngine,
    HierarchicalEngine,
    ParallelEngine,
    SequentialEngine,
)
from .flow import Flow, FlowPlan
from .helpers import coordinate, parallel, plan_and_execute, route, sequential
from .hitl import FlowPaused
from .loop import AlwaysOkVerifier, CallableVerifier, Loop, VerdictResult, Verifier
from .memory import MemoryStore, Tier, TieredMemoryStore
from .nodes import Edge, FlowConfigError, MapNode, Node, RouterNode
from .observability import (
    ContextSnapshotConfig,
    MessageDisposition,
    MessageSnapshot,
    emit_context_snapshot,
    emit_node_end,
    emit_node_start,
)
from .optimizer import Optimizer, OptimizationRule
from .runnable import FunctionRunnable, Runnable
from .state import Reducer, SharedState, append, merge, overwrite

# Workflow ergonomics: high-level composable classes
from .step import Step
from .workflow import Workflow
from .condition import Condition, ComposableElement
from .parallel_group import Parallel_Group
from .flow_class import FlowClass, start, listen, router

# Convenience aliases for progressive-disclosure naming (design spec)
Map = MapNode
Router = RouterNode

__all__ = [
    # Core
    "Runnable",
    "FunctionRunnable",
    # Tier 2: Loop + Verifier
    "VerdictResult",
    "Verifier",
    "AlwaysOkVerifier",
    "CallableVerifier",
    "Loop",
    # Tier 3: Flow graph (advanced)
    "Flow",
    "FlowPlan",
    "FlowPaused",
    "Edge",
    "Node",
    "MapNode",
    "Map",
    "RouterNode",
    "Router",
    "FlowConfigError",
    # State
    "Reducer",
    "SharedState",
    "overwrite",
    "append",
    "merge",
    # Engines
    "ExecutionEngine",
    "SequentialEngine",
    "ParallelEngine",
    "HierarchicalEngine",
    # Optimizer
    "Optimizer",
    "OptimizationRule",
    # Memory
    "MemoryStore",
    "Tier",
    "TieredMemoryStore",
    # Observability
    "ContextSnapshotConfig",
    "MessageDisposition",
    "MessageSnapshot",
    "emit_context_snapshot",
    "emit_node_end",
    "emit_node_start",
    # Helpers (→ Flow)
    "sequential",
    "parallel",
    "route",
    "coordinate",
    "plan_and_execute",
    # High-level Workflow API (preferred)
    "Step",
    "Workflow",
    "Condition",
    "ComposableElement",
    "Parallel_Group",
    "FlowClass",
    "start",
    "listen",
    "router",
]
