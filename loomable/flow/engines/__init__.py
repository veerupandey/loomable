"""Execution engines for the Flow runtime.

Engines implement the physical plan: how nodes are driven to completion.
Ships Sequential, Parallel (BSP), and Hierarchical, plus an auto-selector.
"""

from .base import ExecutionEngine, detect_cycle, level_sets, toposort
from .hierarchical import HierarchicalEngine
from .parallel import ParallelEngine
from .selector import EngineSelector
from .sequential import SequentialEngine

__all__ = [
    "EngineSelector",
    "ExecutionEngine",
    "HierarchicalEngine",
    "ParallelEngine",
    "SequentialEngine",
    "detect_cycle",
    "level_sets",
    "toposort",
]
