"""Tests for ExecutionEngine protocol and shared topology utilities.

Covers:
- toposort produces correct topological order
- detect_cycle raises FlowConfigError naming the cycle
- detect_cycle allows LoopNode (a node whose runnable is a Loop)
- level_sets groups independent nodes at the same level
"""

from __future__ import annotations

import pytest

from loomable.flow.engines.base import (
    ExecutionEngine,
    detect_cycle,
    level_sets,
    toposort,
)
from loomable.flow.loop import AlwaysOkVerifier, Loop
from loomable.flow.nodes import Edge, FlowConfigError, Node
from loomable.flow.runnable import FunctionRunnable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(node_id: str) -> Node:
    """Create a simple Node wrapping a no-op function."""
    return Node(node_id=node_id, runnable=FunctionRunnable(lambda x: x))


def _make_loop_node(node_id: str) -> Node:
    """Create a Node wrapping a Loop (LoopNode)."""
    body = FunctionRunnable(lambda x: x)
    loop = Loop(body=body, max_iterations=3)
    return Node(node_id=node_id, runnable=loop)


# ---------------------------------------------------------------------------
# toposort tests
# ---------------------------------------------------------------------------


class TestToposort:
    """Tests for the toposort utility."""

    def test_linear_chain(self):
        """A→B→C produces [A, B, C]."""
        nodes = {
            "A": _make_node("A"),
            "B": _make_node("B"),
            "C": _make_node("C"),
        }
        edges = [Edge(source="A", target="B"), Edge(source="B", target="C")]

        result = toposort(nodes, edges)

        assert result == ["A", "B", "C"]

    def test_diamond_dag(self):
        """Diamond: A→B, A→C, B→D, C→D produces valid order."""
        nodes = {
            "A": _make_node("A"),
            "B": _make_node("B"),
            "C": _make_node("C"),
            "D": _make_node("D"),
        }
        edges = [
            Edge(source="A", target="B"),
            Edge(source="A", target="C"),
            Edge(source="B", target="D"),
            Edge(source="C", target="D"),
        ]

        result = toposort(nodes, edges)

        # A must come before B and C; B and C must come before D
        assert result.index("A") < result.index("B")
        assert result.index("A") < result.index("C")
        assert result.index("B") < result.index("D")
        assert result.index("C") < result.index("D")

    def test_no_edges(self):
        """Disconnected nodes: all returned (sorted lexicographically)."""
        nodes = {
            "X": _make_node("X"),
            "Y": _make_node("Y"),
            "Z": _make_node("Z"),
        }
        edges: list[Edge] = []

        result = toposort(nodes, edges)

        assert sorted(result) == ["X", "Y", "Z"]
        # With deterministic sort, expect alphabetical
        assert result == ["X", "Y", "Z"]

    def test_single_node(self):
        """Single node with no edges returns that node."""
        nodes = {"solo": _make_node("solo")}
        result = toposort(nodes, [])
        assert result == ["solo"]

    def test_cycle_raises(self):
        """A cycle raises FlowConfigError."""
        nodes = {
            "A": _make_node("A"),
            "B": _make_node("B"),
        }
        edges = [Edge(source="A", target="B"), Edge(source="B", target="A")]

        with pytest.raises(FlowConfigError, match="cycle"):
            toposort(nodes, edges)


# ---------------------------------------------------------------------------
# detect_cycle tests
# ---------------------------------------------------------------------------


class TestDetectCycle:
    """Tests for the detect_cycle utility."""

    def test_no_cycle_passes_silently(self):
        """A DAG does not raise."""
        nodes = {
            "A": _make_node("A"),
            "B": _make_node("B"),
            "C": _make_node("C"),
        }
        edges = [Edge(source="A", target="B"), Edge(source="B", target="C")]

        # Should not raise
        detect_cycle(nodes, edges)

    def test_cycle_raises_naming_nodes(self):
        """A cycle raises FlowConfigError that names the nodes."""
        nodes = {
            "A": _make_node("A"),
            "B": _make_node("B"),
            "C": _make_node("C"),
        }
        edges = [
            Edge(source="A", target="B"),
            Edge(source="B", target="C"),
            Edge(source="C", target="A"),
        ]

        with pytest.raises(FlowConfigError, match="cycle") as exc_info:
            detect_cycle(nodes, edges)

        error_msg = str(exc_info.value)
        # The error should name nodes in the cycle
        assert "A" in error_msg
        assert "B" in error_msg
        assert "C" in error_msg

    def test_self_loop_raises(self):
        """A self-loop (A→A) raises FlowConfigError."""
        nodes = {"A": _make_node("A")}
        edges = [Edge(source="A", target="A")]

        with pytest.raises(FlowConfigError, match="cycle"):
            detect_cycle(nodes, edges)

    def test_loop_node_cycle_allowed(self):
        """A cycle among LoopNodes (explicit Loop used as node) is permitted."""
        nodes = {
            "loop_a": _make_loop_node("loop_a"),
            "loop_b": _make_loop_node("loop_b"),
        }
        edges = [
            Edge(source="loop_a", target="loop_b"),
            Edge(source="loop_b", target="loop_a"),
        ]

        # Should NOT raise — all cycle participants are LoopNodes
        detect_cycle(nodes, edges)

    def test_mixed_cycle_with_loop_node_raises(self):
        """A cycle with a mix of LoopNodes and regular nodes raises."""
        nodes = {
            "loop_a": _make_loop_node("loop_a"),
            "regular_b": _make_node("regular_b"),
        }
        edges = [
            Edge(source="loop_a", target="regular_b"),
            Edge(source="regular_b", target="loop_a"),
        ]

        with pytest.raises(FlowConfigError, match="cycle"):
            detect_cycle(nodes, edges)

    def test_disconnected_graph_no_cycle(self):
        """Disconnected components without cycles pass."""
        nodes = {
            "A": _make_node("A"),
            "B": _make_node("B"),
            "C": _make_node("C"),
            "D": _make_node("D"),
        }
        edges = [Edge(source="A", target="B"), Edge(source="C", target="D")]

        detect_cycle(nodes, edges)


# ---------------------------------------------------------------------------
# level_sets tests
# ---------------------------------------------------------------------------


class TestLevelSets:
    """Tests for the level_sets utility."""

    def test_linear_chain(self):
        """A→B→C produces three levels: [[A], [B], [C]]."""
        nodes = {
            "A": _make_node("A"),
            "B": _make_node("B"),
            "C": _make_node("C"),
        }
        edges = [Edge(source="A", target="B"), Edge(source="B", target="C")]

        result = level_sets(nodes, edges)

        assert result == [["A"], ["B"], ["C"]]

    def test_diamond_dag(self):
        """Diamond: A→B,C→D produces [[A], [B,C], [D]]."""
        nodes = {
            "A": _make_node("A"),
            "B": _make_node("B"),
            "C": _make_node("C"),
            "D": _make_node("D"),
        }
        edges = [
            Edge(source="A", target="B"),
            Edge(source="A", target="C"),
            Edge(source="B", target="D"),
            Edge(source="C", target="D"),
        ]

        result = level_sets(nodes, edges)

        assert result == [["A"], ["B", "C"], ["D"]]

    def test_independent_nodes(self):
        """Nodes with no edges all land in level 0."""
        nodes = {
            "X": _make_node("X"),
            "Y": _make_node("Y"),
            "Z": _make_node("Z"),
        }
        edges: list[Edge] = []

        result = level_sets(nodes, edges)

        assert result == [["X", "Y", "Z"]]

    def test_wide_fan_out(self):
        """A→B, A→C, A→D produces [[A], [B, C, D]]."""
        nodes = {
            "A": _make_node("A"),
            "B": _make_node("B"),
            "C": _make_node("C"),
            "D": _make_node("D"),
        }
        edges = [
            Edge(source="A", target="B"),
            Edge(source="A", target="C"),
            Edge(source="A", target="D"),
        ]

        result = level_sets(nodes, edges)

        assert result == [["A"], ["B", "C", "D"]]

    def test_complex_dag(self):
        """More complex DAG with multiple levels.

        A → B → D
        A → C → D
        C → E
        """
        nodes = {
            "A": _make_node("A"),
            "B": _make_node("B"),
            "C": _make_node("C"),
            "D": _make_node("D"),
            "E": _make_node("E"),
        }
        edges = [
            Edge(source="A", target="B"),
            Edge(source="A", target="C"),
            Edge(source="B", target="D"),
            Edge(source="C", target="D"),
            Edge(source="C", target="E"),
        ]

        result = level_sets(nodes, edges)

        assert result == [["A"], ["B", "C"], ["D", "E"]]

    def test_cycle_raises(self):
        """A cycle raises FlowConfigError."""
        nodes = {
            "A": _make_node("A"),
            "B": _make_node("B"),
        }
        edges = [Edge(source="A", target="B"), Edge(source="B", target="A")]

        with pytest.raises(FlowConfigError, match="cycle"):
            level_sets(nodes, edges)

    def test_single_node(self):
        """Single node produces one level."""
        nodes = {"solo": _make_node("solo")}
        result = level_sets(nodes, [])
        assert result == [["solo"]]


# ---------------------------------------------------------------------------
# ExecutionEngine protocol tests
# ---------------------------------------------------------------------------


class TestExecutionEngineProtocol:
    """Tests that the ExecutionEngine protocol is runtime_checkable."""

    def test_protocol_is_runtime_checkable(self):
        """ExecutionEngine can be checked with isinstance at runtime."""

        class MyEngine:
            async def run(self, flow, input, state, context):
                pass

        engine = MyEngine()
        assert isinstance(engine, ExecutionEngine)

    def test_non_conforming_object_fails_check(self):
        """An object without the required method does not satisfy the protocol."""

        class NotAnEngine:
            pass

        obj = NotAnEngine()
        assert not isinstance(obj, ExecutionEngine)
