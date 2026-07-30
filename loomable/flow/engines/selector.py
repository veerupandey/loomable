"""EngineSelector: automatic engine selection from flow topology.

Implements Req 9: pure topology inspection to select the appropriate
ExecutionEngine when engine="auto".

Algorithm:
1. If any node has manager=True → HierarchicalEngine (Req 9.4)
2. Compute level_sets. If any level has ≥2 nodes → ParallelEngine (Req 9.3)
3. Otherwise (all levels have exactly 1 node = linear chain) → SequentialEngine (Req 9.2)
"""

from __future__ import annotations

__all__ = ["EngineSelector"]

from loomable.flow.engines.base import ExecutionEngine, level_sets
from loomable.flow.nodes import Edge, Node


class EngineSelector:
    """Select an ExecutionEngine based on flow topology (Req 9.1).

    This is a pure topology inspector — it examines nodes and edges to
    determine the best engine without executing anything.

    Selection rules (applied in priority order):
    1. Manager present → HierarchicalEngine (Req 9.4)
    2. Independent branches (any level has ≥2 nodes) → ParallelEngine (Req 9.3)
    3. Linear chain (all levels have exactly 1 node) → SequentialEngine (Req 9.2)
    """

    @staticmethod
    def select(nodes: dict[str, Node], edges: list[Edge]) -> ExecutionEngine:
        """Analyze topology and return the appropriate engine instance.

        Parameters
        ----------
        nodes:
            Mapping of node_id to Node instances.
        edges:
            List of Edge instances defining directed dependencies.

        Returns
        -------
        An instance of the selected ExecutionEngine.
        """
        from loomable.flow.engines.hierarchical import HierarchicalEngine
        from loomable.flow.engines.parallel import ParallelEngine
        from loomable.flow.engines.sequential import SequentialEngine

        # Rule 1: If any node has manager=True → Hierarchical
        for node in nodes.values():
            if node.manager:
                return HierarchicalEngine()

        # Rule 2: Compute level sets. If any level has ≥2 nodes → Parallel
        if nodes:
            levels = level_sets(nodes, edges)
            for level in levels:
                if len(level) >= 2:
                    return ParallelEngine()

        # Rule 3: Linear chain (or empty/single node) → Sequential
        return SequentialEngine()
