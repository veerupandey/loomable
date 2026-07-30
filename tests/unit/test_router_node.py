"""Tests for RouterNode (Task 9.2).

Validates Requirements 11.4, 11.5, 11.6:
- RouterNode with predicate chooser selects the correct branch
- Unselected branches are skipped (selection written to state for edge gating)
- handoff=True marks the selected node's output as the final output
- RouterNode satisfies Runnable protocol
- Chooser must select from declared choices (invalid selection handled)
"""

from __future__ import annotations

import pytest

from loomable.agent.context import RunContext
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow import RouterNode
from loomable.flow.nodes import FlowConfigError
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.state import SharedState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(text: str) -> AgentOutput:
    """Create a simple text AgentOutput."""
    return AgentOutput(
        parts=[
            MediaPart(
                modality=Modality.TEXT,
                media_type="text/plain",
                data=text.encode("utf-8"),
            )
        ]
    )


def _chooser_always(node_id: str):
    """Return a callable that always selects the given node_id."""
    def chooser(input):
        return node_id
    return chooser


def _chooser_by_input():
    """Return a callable that uses the input directly as the selection."""
    def chooser(input):
        return input
    return chooser


class RunnableChooser:
    """A Runnable that selects based on input."""

    async def arun(self, input, *, context=None) -> RunResult:
        # The input is the node_id to select
        return RunResult(
            output=_make_output(str(input)),
            session_id="",
        )


class MetadataChooser:
    """A Runnable that returns selection via metadata."""

    def __init__(self, selection):
        self._selection = selection

    async def arun(self, input, *, context=None) -> RunResult:
        return RunResult(
            output=_make_output(""),
            session_id="",
            metadata={"selection": self._selection},
        )


# ---------------------------------------------------------------------------
# RouterNode satisfies Runnable protocol
# ---------------------------------------------------------------------------


class TestRouterNodeProtocol:
    """Req 11.4: RouterNode is a Runnable."""

    def test_satisfies_runnable_protocol(self):
        router = RouterNode(_chooser_always("branch_a"), choices=["branch_a", "branch_b"])
        assert isinstance(router, Runnable)

    def test_has_arun_method(self):
        router = RouterNode(_chooser_always("branch_a"), choices=["branch_a", "branch_b"])
        assert hasattr(router, "arun")
        assert callable(router.arun)


# ---------------------------------------------------------------------------
# Predicate chooser selects the correct branch
# ---------------------------------------------------------------------------


class TestRouterNodePredicateChooser:
    """Req 11.4: RouterNode with predicate chooser selects the correct branch."""

    @pytest.mark.asyncio
    async def test_callable_chooser_selects_branch(self):
        """A callable chooser picks the correct downstream node."""
        router = RouterNode(
            _chooser_always("branch_a"),
            choices=["branch_a", "branch_b"],
        )
        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await router.arun("some input", context=ctx)

        assert result.metadata["router_selected"] == "branch_a"
        assert state.get("_router_selection") == "branch_a"

    @pytest.mark.asyncio
    async def test_callable_chooser_dynamic_selection(self):
        """Chooser can dynamically pick based on input."""
        router = RouterNode(
            _chooser_by_input(),
            choices=["fast", "careful"],
        )
        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await router.arun("careful", context=ctx)

        assert result.metadata["router_selected"] == "careful"
        assert state.get("_router_selection") == "careful"

    @pytest.mark.asyncio
    async def test_runnable_chooser_selects_branch(self):
        """Req 11.5: A Runnable chooser selects among declared candidates."""
        chooser = RunnableChooser()
        router = RouterNode(
            chooser,
            choices=["node_x", "node_y"],
        )
        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await router.arun("node_x", context=ctx)

        assert result.metadata["router_selected"] == "node_x"
        assert state.get("_router_selection") == "node_x"

    @pytest.mark.asyncio
    async def test_metadata_based_selection(self):
        """Chooser can return selection via metadata."""
        chooser = MetadataChooser("branch_b")
        router = RouterNode(
            chooser,
            choices=["branch_a", "branch_b"],
        )
        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await router.arun("anything", context=ctx)

        assert result.metadata["router_selected"] == "branch_b"
        assert state.get("_router_selection") == "branch_b"

    @pytest.mark.asyncio
    async def test_multi_selection(self):
        """Chooser can select multiple branches."""
        chooser = MetadataChooser(["branch_a", "branch_b"])
        router = RouterNode(
            chooser,
            choices=["branch_a", "branch_b", "branch_c"],
        )
        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await router.arun("anything", context=ctx)

        assert result.metadata["router_selected"] == ["branch_a", "branch_b"]
        assert state.get("_router_selection") == ["branch_a", "branch_b"]


# ---------------------------------------------------------------------------
# Unselected branches are skipped (selection in state for edge gating)
# ---------------------------------------------------------------------------


class TestRouterNodeBranchSkipping:
    """Req 11.4: Unselected branches are skipped via state-based gating."""

    @pytest.mark.asyncio
    async def test_selection_written_to_state(self):
        """The router writes selection to _router_selection for edge conditions."""
        router = RouterNode(
            _chooser_always("selected_branch"),
            choices=["selected_branch", "other_branch"],
        )
        state = SharedState()
        ctx = RunContext(shared_state=state)

        await router.arun("input", context=ctx)

        # Downstream edge conditions can now gate on this value
        selection = state.get("_router_selection")
        assert selection == "selected_branch"

        # Simulating edge condition: only traverse if selected
        edge_condition_selected = lambda s: s.get("_router_selection") == "selected_branch"
        edge_condition_other = lambda s: s.get("_router_selection") == "other_branch"

        assert edge_condition_selected(state) is True
        assert edge_condition_other(state) is False

    @pytest.mark.asyncio
    async def test_no_state_write_without_context(self):
        """If no context/state is available, router still works but doesn't write state."""
        router = RouterNode(
            _chooser_always("branch_a"),
            choices=["branch_a", "branch_b"],
        )

        # No context at all
        result = await router.arun("input")
        assert result.metadata["router_selected"] == "branch_a"

    @pytest.mark.asyncio
    async def test_no_state_write_with_empty_context(self):
        """If context has no shared_state, router still works."""
        router = RouterNode(
            _chooser_always("branch_a"),
            choices=["branch_a", "branch_b"],
        )
        ctx = RunContext()  # shared_state is None by default

        result = await router.arun("input", context=ctx)
        assert result.metadata["router_selected"] == "branch_a"


# ---------------------------------------------------------------------------
# handoff=True marks the selected node's output as the final output
# ---------------------------------------------------------------------------


class TestRouterNodeHandoff:
    """Req 11.6: handoff=True means the chosen node owns the final output."""

    @pytest.mark.asyncio
    async def test_handoff_true_in_metadata(self):
        """When handoff=True, metadata includes router_handoff flag."""
        router = RouterNode(
            _chooser_always("delegate"),
            choices=["delegate", "other"],
            handoff=True,
        )
        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await router.arun("input", context=ctx)

        assert result.metadata["router_handoff"] is True
        assert result.metadata["router_selected"] == "delegate"

    @pytest.mark.asyncio
    async def test_handoff_false_no_flag(self):
        """When handoff=False (default), no router_handoff in metadata."""
        router = RouterNode(
            _chooser_always("delegate"),
            choices=["delegate", "other"],
            handoff=False,
        )
        state = SharedState()
        ctx = RunContext(shared_state=state)

        result = await router.arun("input", context=ctx)

        assert "router_handoff" not in result.metadata
        assert result.metadata["router_selected"] == "delegate"

    @pytest.mark.asyncio
    async def test_handoff_default_is_false(self):
        """The default for handoff is False."""
        router = RouterNode(
            _chooser_always("delegate"),
            choices=["delegate", "other"],
        )
        assert router.handoff is False

        state = SharedState()
        ctx = RunContext(shared_state=state)
        result = await router.arun("input", context=ctx)
        assert "router_handoff" not in result.metadata


# ---------------------------------------------------------------------------
# Invalid selection handling
# ---------------------------------------------------------------------------


class TestRouterNodeValidation:
    """Chooser must select from declared choices; invalid selection is an error."""

    @pytest.mark.asyncio
    async def test_invalid_single_selection_raises(self):
        """Selecting a node_id not in choices raises FlowConfigError."""
        router = RouterNode(
            _chooser_always("unknown_node"),
            choices=["branch_a", "branch_b"],
        )
        state = SharedState()
        ctx = RunContext(shared_state=state)

        with pytest.raises(FlowConfigError, match="unknown_node"):
            await router.arun("input", context=ctx)

    @pytest.mark.asyncio
    async def test_invalid_multi_selection_raises(self):
        """Selecting any node_id not in choices raises FlowConfigError."""
        chooser = MetadataChooser(["branch_a", "invalid_branch"])
        router = RouterNode(
            chooser,
            choices=["branch_a", "branch_b"],
        )
        state = SharedState()
        ctx = RunContext(shared_state=state)

        with pytest.raises(FlowConfigError, match="invalid_branch"):
            await router.arun("input", context=ctx)

    @pytest.mark.asyncio
    async def test_empty_selection_raises(self):
        """An empty string selection is invalid."""
        # Chooser returns empty string via output text
        router = RouterNode(
            lambda x: "",
            choices=["branch_a", "branch_b"],
        )
        state = SharedState()
        ctx = RunContext(shared_state=state)

        with pytest.raises(FlowConfigError):
            await router.arun("input", context=ctx)

    def test_non_callable_chooser_raises_type_error(self):
        """Passing something that's not Runnable or Callable raises TypeError."""
        with pytest.raises(TypeError, match="Runnable or Callable"):
            RouterNode(42, choices=["a", "b"])  # type: ignore


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRouterNodeRepr:
    def test_repr_without_handoff(self):
        router = RouterNode(_chooser_always("a"), choices=["a", "b"])
        assert repr(router) == "RouterNode(choices=['a', 'b'])"

    def test_repr_with_handoff(self):
        router = RouterNode(_chooser_always("a"), choices=["a", "b"], handoff=True)
        assert repr(router) == "RouterNode(choices=['a', 'b'], handoff=True)"
