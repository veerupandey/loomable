# Feature: workflow-ergonomics, Property 9: Duplicate step names are rejected at construction
"""Property 9: Duplicate step names are rejected at construction.

For any list of Steps where two or more Steps share the same `name`,
constructing a Workflow with that list SHALL raise a `FlowConfigError`
whose message contains the duplicate name.

**Validates: Requirements 10.1**
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.flow.nodes import FlowConfigError
from loomable.flow.step import Step
from loomable.flow.workflow import Workflow


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: non-empty name strings for steps — printable characters, no control
# chars, varied lengths.
step_name_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=30,
).filter(lambda s: s.strip() != "" and len(s) > 0)


# Strategy: list of 2-5 distinct step names (the base list before duplication)
distinct_names_st = st.lists(
    step_name_st,
    min_size=2,
    max_size=5,
    unique=True,
)


# Strategy: an insertion index for the duplicate step
# (parameterized by the list length at test time)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop(x: Any) -> str:
    """A minimal callable agent for testing."""
    return f"processed: {x}"


def _build_steps_with_duplicate(
    distinct_names: list[str], dup_index: int, insert_index: int
) -> tuple[list[Step], str]:
    """Build a step list with one duplicated name.

    Parameters
    ----------
    distinct_names:
        A list of unique names to create initial steps from.
    dup_index:
        Index into distinct_names selecting which name to duplicate.
    insert_index:
        Position in the step list where the duplicate step is inserted.

    Returns
    -------
    tuple of (steps list with duplicate, the duplicated name)
    """
    steps = [Step(name=n, agent=_noop) for n in distinct_names]
    dup_name = distinct_names[dup_index]
    # Insert a duplicate step at the chosen position
    steps.insert(insert_index, Step(name=dup_name, agent=_noop))
    return steps, dup_name


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestDuplicateNameRejection:
    """Property 9: Duplicate step names are rejected at construction."""

    @settings(max_examples=100, deadline=None)
    @given(
        distinct_names=distinct_names_st,
        data=st.data(),
    )
    def test_workflow_rejects_duplicate_step_names(
        self, distinct_names: list[str], data: st.DataObject
    ) -> None:
        """Constructing a Workflow with duplicate step names SHALL raise
        FlowConfigError whose message contains the duplicate name."""
        # Pick which name to duplicate
        dup_index = data.draw(
            st.integers(min_value=0, max_value=len(distinct_names) - 1)
        )
        # Pick where to insert the duplicate (anywhere in the list)
        insert_index = data.draw(
            st.integers(min_value=0, max_value=len(distinct_names))
        )

        steps, dup_name = _build_steps_with_duplicate(
            distinct_names, dup_index, insert_index
        )

        with pytest.raises(FlowConfigError) as exc_info:
            Workflow(name="test-workflow", steps=steps)

        # Verify the error message contains the duplicate name
        assert dup_name in str(exc_info.value)

    @settings(max_examples=100, deadline=None)
    @given(
        distinct_names=distinct_names_st,
        data=st.data(),
    )
    def test_error_message_format(
        self, distinct_names: list[str], data: st.DataObject
    ) -> None:
        """The FlowConfigError message SHALL follow the format
        "Duplicate step name: '{name}'"."""
        dup_index = data.draw(
            st.integers(min_value=0, max_value=len(distinct_names) - 1)
        )
        insert_index = data.draw(
            st.integers(min_value=0, max_value=len(distinct_names))
        )

        steps, dup_name = _build_steps_with_duplicate(
            distinct_names, dup_index, insert_index
        )

        with pytest.raises(FlowConfigError) as exc_info:
            Workflow(name="test-workflow", steps=steps)

        expected_msg = f"Duplicate step name: '{dup_name}'"
        assert expected_msg in str(exc_info.value)

    @settings(max_examples=100, deadline=None)
    @given(distinct_names=distinct_names_st)
    def test_no_duplicates_does_not_raise(self, distinct_names: list[str]) -> None:
        """A Workflow with all-unique step names SHALL NOT raise
        FlowConfigError — only duplicates trigger the error."""
        steps = [Step(name=n, agent=_noop) for n in distinct_names]

        # Should not raise — all names are unique
        try:
            workflow = Workflow(name="test-workflow", steps=steps)
        except FlowConfigError:
            pytest.fail(
                "FlowConfigError raised for unique step names — "
                "only duplicates should trigger this error"
            )
