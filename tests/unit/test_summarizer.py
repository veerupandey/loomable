"""Unit tests for loomable.kernel.summarizer.

Tests the Summarizer class: checkpoint triggering, summary production,
objective/decision preservation, and context item replacement.
"""

import pytest

from loomable.kernel.models import ContextItem, StructuredSummary, Turn
from loomable.kernel.summarizer import Summarizer


# ---------------------------------------------------------------------------
# should_summarize tests
# ---------------------------------------------------------------------------


class TestShouldSummarize:
    """Tests for Summarizer.should_summarize()."""

    def test_triggers_at_positive_multiple(self) -> None:
        """Summarization triggers at positive multiples of checkpoint_interval."""
        s = Summarizer(checkpoint_interval=5)
        assert s.should_summarize(5) is True
        assert s.should_summarize(10) is True
        assert s.should_summarize(15) is True

    def test_does_not_trigger_at_zero(self) -> None:
        """Step 0 is not a positive multiple — no summarization."""
        s = Summarizer(checkpoint_interval=5)
        assert s.should_summarize(0) is False

    def test_does_not_trigger_at_non_multiples(self) -> None:
        """Non-multiples do not trigger summarization."""
        s = Summarizer(checkpoint_interval=5)
        assert s.should_summarize(1) is False
        assert s.should_summarize(3) is False
        assert s.should_summarize(7) is False
        assert s.should_summarize(11) is False

    def test_interval_of_one_triggers_every_positive_step(self) -> None:
        """With interval=1, every positive step triggers."""
        s = Summarizer(checkpoint_interval=1)
        for step in range(1, 10):
            assert s.should_summarize(step) is True
        assert s.should_summarize(0) is False

    def test_negative_step_does_not_trigger(self) -> None:
        """Negative steps do not trigger summarization."""
        s = Summarizer(checkpoint_interval=5)
        assert s.should_summarize(-5) is False
        assert s.should_summarize(-10) is False

    def test_invalid_checkpoint_interval_raises(self) -> None:
        """Checkpoint interval must be positive."""
        with pytest.raises(ValueError, match="positive"):
            Summarizer(checkpoint_interval=0)
        with pytest.raises(ValueError, match="positive"):
            Summarizer(checkpoint_interval=-3)


# ---------------------------------------------------------------------------
# summarize tests
# ---------------------------------------------------------------------------


class TestSummarize:
    """Tests for Summarizer.summarize()."""

    def _make_turns(self, count: int, start_step: int = 1) -> list[Turn]:
        """Helper to create test turns."""
        return [
            Turn(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Turn {i} content at step {start_step + i}",
                tokens=50,
                step=start_step + i,
            )
            for i in range(count)
        ]

    def test_produces_structured_summary(self) -> None:
        """summarize() returns a StructuredSummary."""
        s = Summarizer(checkpoint_interval=5)
        turns = self._make_turns(5)
        result = s.summarize(turns)
        assert isinstance(result, StructuredSummary)

    def test_covers_steps_range(self) -> None:
        """Summary covers the step range of input turns."""
        s = Summarizer(checkpoint_interval=5)
        turns = self._make_turns(5, start_step=3)
        result = s.summarize(turns)
        # Steps are 3, 4, 5, 6, 7
        assert result.covers_steps == range(3, 8)

    def test_summary_has_fewer_tokens_than_original(self) -> None:
        """Summary token count should be less than total original tokens."""
        s = Summarizer(checkpoint_interval=5)
        turns = self._make_turns(5)
        original_tokens = sum(t.tokens for t in turns)
        result = s.summarize(turns)
        assert result.tokens < original_tokens

    def test_summary_tokens_at_least_one(self) -> None:
        """Summary token count is at least 1."""
        s = Summarizer(checkpoint_interval=5)
        turns = [Turn(role="user", content="hi", tokens=1, step=1)]
        result = s.summarize(turns)
        assert result.tokens >= 1

    def test_empty_turns_raises(self) -> None:
        """Cannot summarize empty turns."""
        s = Summarizer(checkpoint_interval=5)
        with pytest.raises(ValueError, match="empty"):
            s.summarize([])

    def test_preserves_objectives(self) -> None:
        """Summary preserves extracted objectives (Req 10.4)."""
        s = Summarizer(checkpoint_interval=5)
        turns = [
            Turn(
                role="user",
                content="Objective: Build a REST API for user management",
                tokens=20,
                step=1,
            ),
            Turn(
                role="assistant",
                content="I need to set up the database schema first",
                tokens=20,
                step=2,
            ),
            Turn(
                role="user",
                content="Goal: ensure all endpoints have auth",
                tokens=20,
                step=3,
            ),
        ]
        result = s.summarize(turns)
        assert len(result.objectives) >= 2
        assert any("REST API" in obj for obj in result.objectives)
        assert any("auth" in obj for obj in result.objectives)

    def test_preserves_decisions(self) -> None:
        """Summary preserves extracted decisions (Req 10.4)."""
        s = Summarizer(checkpoint_interval=5)
        turns = [
            Turn(
                role="user",
                content="Decision: use PostgreSQL for the database",
                tokens=20,
                step=1,
            ),
            Turn(
                role="assistant",
                content="We decided to use JWT for authentication",
                tokens=20,
                step=2,
            ),
        ]
        result = s.summarize(turns)
        assert len(result.decisions) >= 2
        assert any("PostgreSQL" in dec for dec in result.decisions)
        assert any("JWT" in dec for dec in result.decisions)

    def test_summary_text_not_empty(self) -> None:
        """Summary text is non-empty."""
        s = Summarizer(checkpoint_interval=5)
        turns = self._make_turns(3)
        result = s.summarize(turns)
        assert len(result.text) > 0


# ---------------------------------------------------------------------------
# apply_summarization tests
# ---------------------------------------------------------------------------


class TestApplySummarization:
    """Tests for Summarizer.apply_summarization()."""

    def test_replaces_turn_items_with_summary(self) -> None:
        """Covered turn items are replaced by a summary item in context."""
        s = Summarizer(checkpoint_interval=3)
        turns = [
            Turn(role="user", content="hello", tokens=10, step=1),
            Turn(role="assistant", content="hi there", tokens=15, step=2),
            Turn(role="user", content="how are you", tokens=12, step=3),
        ]
        context_items = [
            ContextItem(kind="system", tokens=100, priority=100, pinned=True),
            ContextItem(kind="turn", tokens=10, priority=30),
            ContextItem(kind="turn", tokens=15, priority=30),
            ContextItem(kind="turn", tokens=12, priority=30),
        ]

        summary, new_items = s.apply_summarization(turns, context_items)

        # System item preserved
        assert new_items[0].kind == "system"
        # Turn items removed, summary added
        turn_items = [it for it in new_items if it.kind == "turn"]
        summary_items = [it for it in new_items if it.kind == "summary"]
        assert len(turn_items) == 0
        assert len(summary_items) == 1

    def test_summary_item_tokens_match_summary(self) -> None:
        """The summary context item has the same token count as the summary."""
        s = Summarizer(checkpoint_interval=3)
        turns = [
            Turn(role="user", content="hello", tokens=10, step=1),
            Turn(role="assistant", content="world", tokens=10, step=2),
            Turn(role="user", content="test", tokens=10, step=3),
        ]
        context_items = [
            ContextItem(kind="turn", tokens=10, priority=30),
            ContextItem(kind="turn", tokens=10, priority=30),
            ContextItem(kind="turn", tokens=10, priority=30),
        ]

        summary, new_items = s.apply_summarization(turns, context_items)
        summary_item = [it for it in new_items if it.kind == "summary"][0]
        assert summary_item.tokens == summary.tokens

    def test_total_tokens_reduced_after_summarization(self) -> None:
        """Token count after summarization is less than or equal to before."""
        s = Summarizer(checkpoint_interval=5)
        turns = [
            Turn(role="user", content=f"content {i}", tokens=50, step=i)
            for i in range(1, 6)
        ]
        context_items = [
            ContextItem(kind="system", tokens=100, priority=100, pinned=True),
        ] + [
            ContextItem(kind="turn", tokens=50, priority=30)
            for _ in range(5)
        ]

        original_tokens = sum(it.tokens for it in context_items)
        _summary, new_items = s.apply_summarization(turns, context_items)
        new_tokens = sum(it.tokens for it in new_items)

        assert new_tokens <= original_tokens

    def test_pinned_items_preserved(self) -> None:
        """Pinned items are never removed during summarization."""
        s = Summarizer(checkpoint_interval=2)
        turns = [
            Turn(role="user", content="a", tokens=10, step=1),
            Turn(role="assistant", content="b", tokens=10, step=2),
        ]
        context_items = [
            ContextItem(kind="system", tokens=100, priority=100, pinned=True),
            ContextItem(kind="schema", tokens=50, priority=90, pinned=True),
            ContextItem(kind="turn", tokens=10, priority=30),
            ContextItem(kind="turn", tokens=10, priority=30),
        ]

        _summary, new_items = s.apply_summarization(turns, context_items)

        pinned = [it for it in new_items if it.pinned]
        assert len(pinned) == 2
        assert pinned[0].kind == "system"
        assert pinned[1].kind == "schema"

    def test_returns_valid_structured_summary(self) -> None:
        """The returned summary is a valid StructuredSummary."""
        s = Summarizer(checkpoint_interval=2)
        turns = [
            Turn(role="user", content="Objective: finish the project", tokens=20, step=1),
            Turn(role="assistant", content="Decision: use Python", tokens=20, step=2),
        ]
        context_items = [
            ContextItem(kind="turn", tokens=20, priority=30),
            ContextItem(kind="turn", tokens=20, priority=30),
        ]

        summary, _new_items = s.apply_summarization(turns, context_items)

        assert isinstance(summary, StructuredSummary)
        assert summary.covers_steps == range(1, 3)
        assert len(summary.objectives) >= 1
        assert len(summary.decisions) >= 1
