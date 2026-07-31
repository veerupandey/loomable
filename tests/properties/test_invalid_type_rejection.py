# Feature: workflow-ergonomics, Property 10: Invalid composable types are rejected at construction
"""Property 10: Invalid composable types are rejected at construction.

For any element that is not a valid composable type (Step, Condition,
Parallel_Group, Loop, Workflow), passing it in a Condition's `then_steps`
or `else_steps` SHALL raise a `TypeError` at construction time.

**Validates: Requirements 10.2**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.flow.condition import Condition
from loomable.flow.step import Step


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: invalid element types that should be rejected by Condition
# These are common Python types that are NOT valid composable elements.
invalid_elements = st.one_of(
    st.integers(),
    st.text(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
    st.binary(max_size=20),
    st.dictionaries(keys=st.text(max_size=5), values=st.integers(), max_size=3),
    st.lists(st.integers(), max_size=5),
    st.tuples(st.integers(), st.text()),
    st.frozensets(st.integers(), max_size=3),
)


def _dummy_predicate(state: Any) -> bool:
    """A trivial predicate for Condition construction."""
    return True


def _make_valid_step(name: str = "valid_step") -> Step:
    """Create a valid Step for use as a companion in step lists."""
    return Step(name=name, agent=lambda x: x)


# ---------------------------------------------------------------------------
# Property tests: Invalid types in then_steps
# ---------------------------------------------------------------------------


class TestInvalidTypeInThenSteps:
    """Invalid composable types in then_steps raise TypeError at construction."""

    @settings(max_examples=100, deadline=None)
    @given(invalid=invalid_elements)
    def test_single_invalid_element_in_then_steps_raises_type_error(
        self,
        invalid: Any,
    ) -> None:
        """Passing a single invalid element in then_steps raises TypeError."""
        with pytest.raises(TypeError, match="Invalid element type"):
            Condition(
                condition=_dummy_predicate,
                then_steps=[invalid],
            )

    @settings(max_examples=100, deadline=None)
    @given(invalid=invalid_elements)
    def test_invalid_element_mixed_with_valid_in_then_steps_raises_type_error(
        self,
        invalid: Any,
    ) -> None:
        """A valid step followed by an invalid element in then_steps raises TypeError."""
        valid_step = _make_valid_step()
        with pytest.raises(TypeError, match="Invalid element type"):
            Condition(
                condition=_dummy_predicate,
                then_steps=[valid_step, invalid],
            )

    @settings(max_examples=100, deadline=None)
    @given(invalid=invalid_elements)
    def test_error_message_contains_invalid_type_name(
        self,
        invalid: Any,
    ) -> None:
        """The TypeError message includes the name of the invalid type."""
        expected_type_name = type(invalid).__name__
        with pytest.raises(TypeError, match=expected_type_name):
            Condition(
                condition=_dummy_predicate,
                then_steps=[invalid],
            )


# ---------------------------------------------------------------------------
# Property tests: Invalid types in else_steps
# ---------------------------------------------------------------------------


class TestInvalidTypeInElseSteps:
    """Invalid composable types in else_steps raise TypeError at construction."""

    @settings(max_examples=100, deadline=None)
    @given(invalid=invalid_elements)
    def test_single_invalid_element_in_else_steps_raises_type_error(
        self,
        invalid: Any,
    ) -> None:
        """Passing a single invalid element in else_steps raises TypeError."""
        valid_step = _make_valid_step()
        with pytest.raises(TypeError, match="Invalid element type"):
            Condition(
                condition=_dummy_predicate,
                then_steps=[valid_step],
                else_steps=[invalid],
            )

    @settings(max_examples=100, deadline=None)
    @given(invalid=invalid_elements)
    def test_invalid_element_mixed_with_valid_in_else_steps_raises_type_error(
        self,
        invalid: Any,
    ) -> None:
        """A valid step followed by an invalid element in else_steps raises TypeError."""
        valid_step = _make_valid_step("then_step")
        else_valid = _make_valid_step("else_valid")
        with pytest.raises(TypeError, match="Invalid element type"):
            Condition(
                condition=_dummy_predicate,
                then_steps=[valid_step],
                else_steps=[else_valid, invalid],
            )

    @settings(max_examples=100, deadline=None)
    @given(invalid=invalid_elements)
    def test_else_steps_error_message_contains_expected_types(
        self,
        invalid: Any,
    ) -> None:
        """The TypeError message mentions the expected valid types."""
        valid_step = _make_valid_step()
        with pytest.raises(
            TypeError,
            match="Expected Step, Condition, Parallel_Group, Loop, or Workflow",
        ):
            Condition(
                condition=_dummy_predicate,
                then_steps=[valid_step],
                else_steps=[invalid],
            )
