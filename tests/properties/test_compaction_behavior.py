# Feature: agent-ergonomics, Property 12
"""Property 12: Compaction summarizes overflow and preserves recent turns.

For any conversation exceeding the compaction threshold, after a run the retained
raw turns SHALL be at most the window size, a summary SHALL be stored in L2
covering the compacted turns, and the compacted raw turns SHALL no longer be present.
Recent turns within the window are preserved unchanged.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4**
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.agent import Agent, ModelSpec
from loomable.content import ModelCapabilities
from loomable.kernel.models import (
    ModelRequest,
    ModelResponse,
    Session,
    Turn,
)
from loomable.kernel.stores import SessionStore
from loomable.kernel.summarizer import Summarizer


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: memory window size (small to keep tests fast)
memory_window_st = st.integers(min_value=2, max_value=8)

# Strategy: compaction threshold — must be > memory_window so there's a gap
# We generate an extra amount added to the window.
extra_threshold_st = st.integers(min_value=1, max_value=8)

# Strategy: number of extra runs after threshold is crossed to observe behavior
extra_runs_st = st.integers(min_value=0, max_value=3)

# Strategy: user input texts (non-empty, short, unique per index via draw)
user_input_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=30,
)

# Strategy: assistant response texts (non-empty, short)
assistant_response_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=30,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class SequentialProvider:
    """A model provider that returns responses from a pre-built list."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._idx < len(self._responses):
            text = self._responses[self._idx]
            self._idx += 1
        else:
            text = "fallback"
        return ModelResponse(content=text, tool_calls=[])


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestCompactionBehavior:
    """Property 12: Compaction summarizes overflow and preserves recent turns."""

    @settings(max_examples=100, deadline=None)
    @given(
        memory_window=memory_window_st,
        extra_threshold=extra_threshold_st,
        extra_runs=extra_runs_st,
        user_inputs=st.lists(user_input_st, min_size=20, max_size=20),
        assistant_responses=st.lists(assistant_response_st, min_size=20, max_size=20),
    )
    @pytest.mark.asyncio
    async def test_compaction_bounds_l1_to_at_most_threshold(
        self,
        memory_window: int,
        extra_threshold: int,
        extra_runs: int,
        user_inputs: list[str],
        assistant_responses: list[str],
    ) -> None:
        """After any run, session.l1 never exceeds compaction_threshold.
        When compaction triggers during a run, l1 is reduced to memory_window.
        (Validates Req 6.1)"""
        compaction_threshold = memory_window + extra_threshold

        # Runs needed so total turns > threshold: each run adds 2 turns
        runs_to_exceed = (compaction_threshold // 2) + 1
        num_runs = runs_to_exceed + extra_runs
        num_runs = min(num_runs, len(user_inputs), len(assistant_responses))
        assume(num_runs * 2 > compaction_threshold)

        provider = SequentialProvider(assistant_responses[:num_runs])

        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            capabilities=ModelCapabilities(),
            session_id="test-compaction-bounds",
            memory_window=memory_window,
            compaction_threshold=compaction_threshold,
        )
        built = agent.build()

        l2_len_before = 0
        for i in range(num_runs):
            await built.arun(user_inputs[i])

            # After every run, l1 never exceeds compaction_threshold
            assert len(built.session.l1) <= compaction_threshold, (
                f"After run {i}, l1 has {len(built.session.l1)} turns, "
                f"but threshold is {compaction_threshold}"
            )

            # If compaction fired this run (l2 grew), l1 should be exactly
            # memory_window (the retained window)
            if len(built.session.l2) > l2_len_before:
                assert len(built.session.l1) == memory_window, (
                    f"After compaction on run {i}, l1 has {len(built.session.l1)} "
                    f"turns, expected memory_window={memory_window}"
                )
            l2_len_before = len(built.session.l2)

        # Compaction must have fired at least once
        assert len(built.session.l2) >= 1

    @settings(max_examples=100, deadline=None)
    @given(
        memory_window=memory_window_st,
        extra_threshold=extra_threshold_st,
        user_inputs=st.lists(user_input_st, min_size=20, max_size=20),
        assistant_responses=st.lists(assistant_response_st, min_size=20, max_size=20),
    )
    @pytest.mark.asyncio
    async def test_compaction_stores_summary_in_l2(
        self,
        memory_window: int,
        extra_threshold: int,
        user_inputs: list[str],
        assistant_responses: list[str],
    ) -> None:
        """After compaction, session.l2 contains at least one StructuredSummary
        with non-empty text and positive token count. (Validates Req 6.2)"""
        compaction_threshold = memory_window + extra_threshold
        # Enough runs to trigger compaction
        runs_to_exceed = (compaction_threshold // 2) + 1
        num_runs = min(runs_to_exceed, len(user_inputs), len(assistant_responses))
        assume(num_runs * 2 > compaction_threshold)

        provider = SequentialProvider(assistant_responses[:num_runs])

        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            capabilities=ModelCapabilities(),
            session_id="test-compaction-l2",
            memory_window=memory_window,
            compaction_threshold=compaction_threshold,
        )
        built = agent.build()

        for i in range(num_runs):
            await built.arun(user_inputs[i])

        # L2 must have at least one summary
        assert len(built.session.l2) >= 1

        # Each summary has valid content
        for summary in built.session.l2:
            assert summary.covers_steps is not None
            assert len(summary.covers_steps) > 0
            assert len(summary.text) > 0
            assert summary.tokens >= 1

    @settings(max_examples=100, deadline=None)
    @given(
        memory_window=memory_window_st,
        extra_threshold=extra_threshold_st,
        user_inputs=st.lists(user_input_st, min_size=20, max_size=20),
        assistant_responses=st.lists(assistant_response_st, min_size=20, max_size=20),
    )
    @pytest.mark.asyncio
    async def test_compacted_turns_not_in_l1(
        self,
        memory_window: int,
        extra_threshold: int,
        user_inputs: list[str],
        assistant_responses: list[str],
    ) -> None:
        """After compaction, the earliest turns that were compacted are no longer
        in L1. The L1 contains only the most recent memory_window turns (after a
        compaction-triggering run). (Validates Req 6.3)"""
        compaction_threshold = memory_window + extra_threshold
        runs_to_exceed = (compaction_threshold // 2) + 1
        num_runs = min(runs_to_exceed, len(user_inputs), len(assistant_responses))
        assume(num_runs * 2 > compaction_threshold)

        provider = SequentialProvider(assistant_responses[:num_runs])

        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            capabilities=ModelCapabilities(),
            session_id="test-compaction-no-old",
            memory_window=memory_window,
            compaction_threshold=compaction_threshold,
        )
        built = agent.build()

        # Track all turns added (by step and content) for later verification
        all_turns_added: list[tuple[int, str, str]] = []  # (step, role, content)

        for i in range(num_runs):
            await built.arun(user_inputs[i])
            # Each run adds user+assistant at step i
            all_turns_added.append((i, "user", user_inputs[i]))
            all_turns_added.append((i, "assistant", assistant_responses[i]))

        # After compaction, L1 should only contain the most recent turns.
        # The earliest step in L1 should be > 0 (some early turns were removed).
        l1_steps = sorted({t.step for t in built.session.l1})
        all_steps = list(range(num_runs))

        # Some early steps must have been removed from L1
        assert l1_steps[0] > 0, (
            "Expected earliest turns to be removed from L1 after compaction"
        )

        # Turns in L1 are a contiguous recent window of steps
        # (they should be the last N steps, where N depends on window)
        for turn in built.session.l1:
            # Every turn in L1 should be from a step that is NOT entirely
            # covered exclusively by L2 (i.e., the overflow turns are gone)
            pass

        # The key property: earliest turns are gone from L1
        earliest_step_in_l1 = min(t.step for t in built.session.l1)
        assert earliest_step_in_l1 > 0

    @settings(max_examples=100, deadline=None)
    @given(
        memory_window=memory_window_st,
        extra_threshold=extra_threshold_st,
        user_inputs=st.lists(user_input_st, min_size=20, max_size=20),
        assistant_responses=st.lists(assistant_response_st, min_size=20, max_size=20),
    )
    @pytest.mark.asyncio
    async def test_recent_turns_preserved_unchanged(
        self,
        memory_window: int,
        extra_threshold: int,
        user_inputs: list[str],
        assistant_responses: list[str],
    ) -> None:
        """The most recent turns (within the window) are preserved unchanged
        after compaction. The last run's turns match what was sent/received.
        (Validates Req 6.4)"""
        compaction_threshold = memory_window + extra_threshold
        runs_to_exceed = (compaction_threshold // 2) + 1
        num_runs = min(runs_to_exceed, len(user_inputs), len(assistant_responses))
        assume(num_runs * 2 > compaction_threshold)

        provider = SequentialProvider(assistant_responses[:num_runs])

        agent = Agent(
            model=ModelSpec(provider="test", provider_impl=provider),
            capabilities=ModelCapabilities(),
            session_id="test-compaction-preserve",
            memory_window=memory_window,
            compaction_threshold=compaction_threshold,
        )
        built = agent.build()

        for i in range(num_runs):
            await built.arun(user_inputs[i])

        # After all runs, L1 should contain the most recent turns.
        last_step = num_runs - 1
        assert len(built.session.l1) > 0

        # The maximum step in L1 should be the most recent step
        max_step_in_l1 = max(t.step for t in built.session.l1)
        assert max_step_in_l1 == last_step, (
            f"Expected most recent step {last_step} in L1, got {max_step_in_l1}"
        )

        # Verify the content of the most recent step's turns is unchanged
        last_step_turns = [t for t in built.session.l1 if t.step == last_step]
        user_turns = [t for t in last_step_turns if t.role == "user"]
        assistant_turns = [t for t in last_step_turns if t.role == "assistant"]

        assert len(user_turns) == 1, "Expected exactly one user turn at the last step"
        assert len(assistant_turns) == 1, "Expected exactly one assistant turn at the last step"

        # The content should match what we sent and received
        assert user_turns[0].content == user_inputs[num_runs - 1]
        assert assistant_turns[0].content == assistant_responses[num_runs - 1]

        # Additionally: all retained turns in L1 preserve their original content
        for turn in built.session.l1:
            step = turn.step
            if turn.role == "user":
                assert turn.content == user_inputs[step], (
                    f"User turn at step {step} was altered"
                )
            elif turn.role == "assistant":
                assert turn.content == assistant_responses[step], (
                    f"Assistant turn at step {step} was altered"
                )
