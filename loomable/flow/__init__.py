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
Prefer ``Workflow`` for new process code.
"""

from .engines import (
    ExecutionEngine,
    HierarchicalEngine,
    ParallelEngine,
    SequentialEngine,
)
from .flow import Flow, FlowPlan
from .helpers import plan_and_execute
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
from .state import Reducer, SharedState, append, extend, merge, overwrite

# Workflow ergonomics: high-level composable classes
from .command import Command
from .step import Step, StepFailed, FAILURE_ACTIONS
from .workflow import Workflow
from .condition import Condition, ComposableElement
from .parallel_group import Parallel_Group
from .route import Route
from .flow_class import FlowClass, start, listen, router

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
    "RouterNode",
    "FlowConfigError",
    # State
    "Reducer",
    "SharedState",
    "overwrite",
    "append",
    "extend",
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
    # Used by Workflow.map
    "plan_and_execute",
    # High-level Workflow API (preferred)
    "Step",
    "StepFailed",
    "FAILURE_ACTIONS",
    "Command",
    "Route",
    "Workflow",
    "Condition",
    "ComposableElement",
    "Parallel_Group",
    "FlowClass",
    "start",
    "listen",
    "router",
]
