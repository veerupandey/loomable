"""loomable.flow - Unified composition model for loomable.

This package provides the three-tier execution model:

- **Tier 1 — Agent** (unchanged): the 3-line entry point.
- **Tier 2 — Loop**: repeat a Runnable until a Verifier passes or a cap is hit.
- **Tier 3 — Flow**: a directed graph of Runnables with shared state, pluggable
  engines, optional optimization, memory, observability, and HITL.

Everything executable satisfies the :class:`Runnable` protocol so agents,
functions, loops, and flows compose interchangeably as graph nodes.

Progressive-disclosure exports (from simplest to most advanced):

- Core: ``Runnable``, ``FunctionRunnable``
- Tier 2: ``Loop``, ``Verifier``, ``VerdictResult``, ``AlwaysOkVerifier``, ``CallableVerifier``
- Tier 3: ``Flow``, ``FlowPlan``, ``Node``, ``Edge``, ``Map`` (MapNode), ``Router`` (RouterNode)
- State: ``SharedState``, ``Reducer``, ``overwrite``, ``append``, ``merge``
- Engines: ``SequentialEngine``, ``ParallelEngine``, ``HierarchicalEngine``, ``ExecutionEngine``
- Optimizer: ``Optimizer``, ``OptimizationRule``
- Memory: ``MemoryStore``, ``Tier``, ``TieredMemoryStore``
- HITL: ``FlowPaused``
- Observability: ``ContextSnapshotConfig``, ``MessageDisposition``, ``MessageSnapshot``
- Helpers: ``sequential``, ``parallel``, ``route``, ``coordinate``, ``plan_and_execute``
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

# Convenience aliases for progressive-disclosure naming (design spec)
Map = MapNode
Router = RouterNode

__all__ = [
    # Core (Tier 1 primitives)
    "Runnable",
    "FunctionRunnable",
    # Tier 2: Loop + Verifier
    "VerdictResult",
    "Verifier",
    "AlwaysOkVerifier",
    "CallableVerifier",
    "Loop",
    # Tier 3: Flow graph
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
    # Helpers (convenience constructors)
    "sequential",
    "parallel",
    "route",
    "coordinate",
    "plan_and_execute",
]
