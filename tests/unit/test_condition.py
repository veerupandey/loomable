"""Unit tests for the Condition class."""

import pytest
import asyncio

from loomable.flow.condition import Condition, _is_valid_composable
from loomable.flow.step import Step
from loomable.flow.loop import Loop
from loomable.flow.runnable import FunctionRunnable, Runnable
from loomable.flow.state import SharedState
from loomable.agent.context import RunContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_step(name: str, output_text: str = "done") -> Step:
    """Create a simple Step that returns a fixed text."""
    return Step(name=name, agent=lambda x: output_text)


def make_condition_true(state: SharedState) -> bool:
    return True


def make_condition_false(state: SharedState) -> bool:
    return False


# ---------------------------------------------------------------------------
# Construction validation tests
# ---------------------------------------------------------------------------


class TestConditionConstruction:
    """Test Condition construction validation."""

    def test_valid_construction_with_then_steps(self):
        """Condition accepts valid then_steps."""
        step = make_step("s1")
        cond = Condition(
            condition=make_condition_true,
            then_steps=[step],
        )
        assert cond.then_steps == [step]
        assert cond.else_steps is None

    def test_valid_construction_with_both_branches(self):
        """Condition accepts valid then_steps and else_steps."""
        s1 = make_step("s1")
        s2 = make_step("s2")
        cond = Condition(
            condition=make_condition_true,
            then_steps=[s1],
            else_steps=[s2],
        )
        assert cond.then_steps == [s1]
        assert cond.else_steps == [s2]

    def test_empty_then_steps_raises_value_error(self):
        """Empty then_steps raises ValueError."""
        with pytest.raises(ValueError, match="At least one then_step is required"):
            Condition(
                condition=make_condition_true,
                then_steps=[],
            )

    def test_invalid_type_in_then_steps_raises_type_error(self):
        """Invalid element type in then_steps raises TypeError."""
        with pytest.raises(TypeError, match="Invalid element type: int"):
            Condition(
                condition=make_condition_true,
                then_steps=[42],
            )

    def test_invalid_type_in_else_steps_raises_type_error(self):
        """Invalid element type in else_steps raises TypeError."""
        step = make_step("s1")
        with pytest.raises(TypeError, match="Invalid element type: str"):
            Condition(
                condition=make_condition_true,
                then_steps=[step],
                else_steps=["not_a_step"],
            )

    def test_invalid_type_dict_raises_type_error(self):
        """Dict in steps raises TypeError with correct message."""
        with pytest.raises(
            TypeError,
            match="Invalid element type: dict.*Expected Step, Condition, Parallel_Group, Loop, or Workflow",
        ):
            Condition(
                condition=make_condition_true,
                then_steps=[{}],
            )

    def test_nested_condition_in_then_steps(self):
        """Condition accepts nested Conditions in then_steps."""
        inner = Condition(
            condition=make_condition_true,
            then_steps=[make_step("inner")],
        )
        outer = Condition(
            condition=make_condition_false,
            then_steps=[inner],
        )
        assert outer.then_steps == [inner]

    def test_loop_in_then_steps(self):
        """Condition accepts Loop elements in then_steps."""
        loop = Loop(body=FunctionRunnable(lambda x: "loop_done"))
        cond = Condition(
            condition=make_condition_true,
            then_steps=[loop],
        )
        assert cond.then_steps == [loop]

    def test_multiple_steps_in_then(self):
        """Condition accepts multiple steps."""
        s1 = make_step("s1", "first")
        s2 = make_step("s2", "second")
        s3 = make_step("s3", "third")
        cond = Condition(
            condition=make_condition_true,
            then_steps=[s1, s2, s3],
        )
        assert len(cond.then_steps) == 3


# ---------------------------------------------------------------------------
# arun execution tests
# ---------------------------------------------------------------------------


class TestConditionExecution:
    """Test Condition arun behavior."""

    def test_true_condition_runs_then_steps(self):
        """When condition is True, then_steps execute."""
        step = Step(name="then_step", agent=lambda x: f"then:{x}")
        cond = Condition(
            condition=make_condition_true,
            then_steps=[step],
        )
        result = asyncio.run(cond.arun("hello"))
        assert result.output.text() == "then:hello"

    def test_false_condition_runs_else_steps(self):
        """When condition is False and else_steps exist, else_steps execute."""
        then_step = Step(name="then_step", agent=lambda x: f"then:{x}")
        else_step = Step(name="else_step", agent=lambda x: f"else:{x}")
        cond = Condition(
            condition=make_condition_false,
            then_steps=[then_step],
            else_steps=[else_step],
        )
        result = asyncio.run(cond.arun("hello"))
        assert result.output.text() == "else:hello"

    def test_false_condition_no_else_passes_through(self):
        """When condition is False and no else_steps, input passes through."""
        then_step = Step(name="then_step", agent=lambda x: f"then:{x}")
        cond = Condition(
            condition=make_condition_false,
            then_steps=[then_step],
        )
        result = asyncio.run(cond.arun("passthrough"))
        assert result.output.text() == "passthrough"

    def test_sequential_execution_in_then_branch(self):
        """Multiple then_steps execute sequentially, piping output."""
        s1 = Step(name="s1", agent=lambda x: f"[1:{x}]")
        s2 = Step(name="s2", agent=lambda x: f"[2:{x}]")
        cond = Condition(
            condition=make_condition_true,
            then_steps=[s1, s2],
        )
        result = asyncio.run(cond.arun("input"))
        # s1 produces "[1:input]", s2 receives that and produces "[2:[1:input]]"
        assert result.output.text() == "[2:[1:input]]"

    def test_condition_uses_shared_state(self):
        """The condition predicate receives SharedState from context."""

        def check_state(state: SharedState) -> bool:
            return state.get("flag") is True

        step = Step(name="flagged", agent=lambda x: "flag_was_set")
        cond = Condition(
            condition=check_state,
            then_steps=[step],
        )

        # With flag=True in state
        state = SharedState()
        state.write("flag", True)
        ctx = RunContext(shared_state=state)
        result = asyncio.run(cond.arun("test", context=ctx))
        assert result.output.text() == "flag_was_set"

    def test_condition_false_with_state(self):
        """The condition predicate returns False based on state."""

        def check_state(state: SharedState) -> bool:
            return state.get("flag") is True

        step = Step(name="flagged", agent=lambda x: "flag_was_set")
        cond = Condition(
            condition=check_state,
            then_steps=[step],
        )

        # With flag=False in state
        state = SharedState()
        state.write("flag", False)
        ctx = RunContext(shared_state=state)
        result = asyncio.run(cond.arun("test", context=ctx))
        # No else_steps, so input passes through
        assert result.output.text() == "test"

    def test_arun_creates_default_state_if_none(self):
        """arun creates a default SharedState if context has none."""

        def always_false(state: SharedState) -> bool:
            # The state should exist (not raise)
            return state.get("nonexistent") is not None

        step = Step(name="s", agent=lambda x: "ran")
        cond = Condition(
            condition=always_false,
            then_steps=[step],
        )
        # No context provided — should not crash
        result = asyncio.run(cond.arun("input"))
        # always_false returns False (get returns None), so passthrough
        assert result.output.text() == "input"


# ---------------------------------------------------------------------------
# Repr and properties tests
# ---------------------------------------------------------------------------


class TestConditionRepr:
    """Test Condition repr and properties."""

    def test_repr_no_else(self):
        """Repr shows step counts."""
        cond = Condition(
            condition=make_condition_true,
            then_steps=[make_step("s1"), make_step("s2")],
        )
        assert repr(cond) == "Condition(then_steps=2, else_steps=0)"

    def test_repr_with_else(self):
        """Repr shows step counts including else."""
        cond = Condition(
            condition=make_condition_true,
            then_steps=[make_step("s1")],
            else_steps=[make_step("e1"), make_step("e2"), make_step("e3")],
        )
        assert repr(cond) == "Condition(then_steps=1, else_steps=3)"

    def test_condition_property(self):
        """Condition exposes the predicate callable."""
        cond = Condition(
            condition=make_condition_true,
            then_steps=[make_step("s1")],
        )
        assert cond.condition is make_condition_true
