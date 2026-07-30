"""Tests for Node, Edge, and FlowConfigError (Task 5.1).

Validates Req 6.3 (Edge with condition) and Req 6.5 (conditional traversal).
"""

from __future__ import annotations

import pytest

from loomable.flow import Edge, FlowConfigError, Node
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.state import SharedState
from loomable.kernel.errors import LoomableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_fn(input):
    return f"echo: {input}"


def _make_runnable() -> FunctionRunnable:
    return FunctionRunnable(_dummy_fn)


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------


class TestEdge:
    def test_basic_construction(self):
        edge = Edge(source="a", target="b")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.condition is None

    def test_with_condition(self):
        pred = lambda state: state.get("ready") is True
        edge = Edge(source="a", target="b", condition=pred)
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.condition is pred

    def test_condition_evaluates_against_shared_state(self):
        """Req 6.5: condition predicate evaluated against SharedState."""
        state = SharedState()
        state.write("flag", True)

        edge_true = Edge(source="a", target="b", condition=lambda s: s.get("flag"))
        edge_false = Edge(source="a", target="b", condition=lambda s: s.get("missing"))

        assert edge_true.condition(state) is True
        assert not edge_false.condition(state)

    def test_edge_is_dataclass(self):
        """Edge should be a dataclass with equality."""
        e1 = Edge(source="x", target="y")
        e2 = Edge(source="x", target="y")
        assert e1 == e2

    def test_edge_with_different_targets_not_equal(self):
        e1 = Edge(source="x", target="y")
        e2 = Edge(source="x", target="z")
        assert e1 != e2


# ---------------------------------------------------------------------------
# Node tests
# ---------------------------------------------------------------------------


class TestNode:
    def test_basic_construction(self):
        runnable = _make_runnable()
        node = Node("my_node", runnable)
        assert node.node_id == "my_node"
        assert node.runnable is runnable
        assert node.require_confirmation is False
        assert node.manager is False

    def test_with_require_confirmation(self):
        runnable = _make_runnable()
        node = Node("hitl_node", runnable, require_confirmation=True)
        assert node.require_confirmation is True
        assert node.manager is False

    def test_with_manager_flag(self):
        runnable = _make_runnable()
        node = Node("mgr", runnable, manager=True)
        assert node.manager is True
        assert node.require_confirmation is False

    def test_with_both_flags(self):
        runnable = _make_runnable()
        node = Node("both", runnable, require_confirmation=True, manager=True)
        assert node.require_confirmation is True
        assert node.manager is True

    def test_repr_plain(self):
        node = Node("plain", _make_runnable())
        assert repr(node) == "Node('plain')"

    def test_repr_with_flags(self):
        node = Node("flagged", _make_runnable(), require_confirmation=True, manager=True)
        assert "hitl" in repr(node)
        assert "manager" in repr(node)

    def test_node_wraps_runnable_protocol(self):
        """Node.runnable satisfies the Runnable protocol."""
        runnable = _make_runnable()
        node = Node("n", runnable)
        assert isinstance(node.runnable, Runnable)


# ---------------------------------------------------------------------------
# FlowConfigError tests
# ---------------------------------------------------------------------------


class TestFlowConfigError:
    def test_inherits_from_loomable_error(self):
        err = FlowConfigError("duplicate node_id 'x'")
        assert isinstance(err, LoomableError)
        assert isinstance(err, Exception)

    def test_message_preserved(self):
        msg = "Node 'foo' referenced by edge but not found in the flow"
        err = FlowConfigError(msg)
        assert str(err) == msg

    def test_can_be_raised_and_caught(self):
        with pytest.raises(FlowConfigError, match="duplicate"):
            raise FlowConfigError("duplicate node_id 'alpha'")

    def test_catchable_as_loomable_error(self):
        with pytest.raises(LoomableError):
            raise FlowConfigError("cycle detected: a -> b -> a")
