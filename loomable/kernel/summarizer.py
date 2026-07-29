"""loomable.kernel.summarizer - Checkpoint summarization for the agent framework.

Implements the Summarizer that compresses accumulated conversation history into
structured summaries at configured checkpoint intervals. Summaries preserve task
objectives and decisions, are stored as L2 content, and replace the covered raw
turns in the context window.

Requirements covered: 10.1, 10.2, 10.3, 10.4, 11.2
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from loomable.kernel.models import ContextItem, StructuredSummary, Turn


# ---------------------------------------------------------------------------
# Objective/Decision extraction patterns
# ---------------------------------------------------------------------------

_OBJECTIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:objective|goal|aim|target|task):\s*(.+)", re.IGNORECASE),
    re.compile(r"(?:we need to|I need to|must|should)\s+(.+)", re.IGNORECASE),
]

_DECISION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:decision|decided|chosen|selected|agreed):\s*(.+)", re.IGNORECASE),
    re.compile(r"(?:we decided|I decided|let's go with|chose to)\s+(.+)", re.IGNORECASE),
]


def _extract_objectives(content: str) -> list[str]:
    """Extract objectives from turn content using pattern matching."""
    objectives: list[str] = []
    for pattern in _OBJECTIVE_PATTERNS:
        for match in pattern.finditer(content):
            text = match.group(1).strip().rstrip(".")
            if text and text not in objectives:
                objectives.append(text)
    return objectives


def _extract_decisions(content: str) -> list[str]:
    """Extract decisions from turn content using pattern matching."""
    decisions: list[str] = []
    for pattern in _DECISION_PATTERNS:
        for match in pattern.finditer(content):
            text = match.group(1).strip().rstrip(".")
            if text and text not in decisions:
                decisions.append(text)
    return decisions


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


@dataclass
class Summarizer:
    """Produces checkpoint summaries at configured intervals.

    The Summarizer compresses accumulated conversation turns into a
    StructuredSummary that preserves task objectives and decisions.
    Summarization triggers exactly when the step count is a positive
    multiple of checkpoint_interval.

    Attributes:
        checkpoint_interval: Number of steps between summarization
            checkpoints. Must be a positive integer.
    """

    checkpoint_interval: int

    def __post_init__(self) -> None:
        if self.checkpoint_interval <= 0:
            raise ValueError(
                f"checkpoint_interval must be positive, got {self.checkpoint_interval}"
            )

    def should_summarize(self, step: int) -> bool:
        """Determine whether summarization should trigger at this step.

        Returns True exactly when step is a positive multiple of
        checkpoint_interval.

        Args:
            step: The current step number (must be positive for trigger).

        Returns:
            True if step > 0 and step % checkpoint_interval == 0.
        """
        return step > 0 and step % self.checkpoint_interval == 0

    def summarize(self, turns: list[Turn]) -> StructuredSummary:
        """Produce a StructuredSummary from the given turns.

        Extracts objectives and decisions from turn content using pattern
        matching, then produces a compressed text summary. The summary
        preserves identified objectives and decisions (Req 10.4).

        Args:
            turns: The list of turns to summarize. Must not be empty.

        Returns:
            A StructuredSummary covering the step range of the input turns.
        """
        if not turns:
            raise ValueError("Cannot summarize an empty list of turns")

        # Determine the step range covered
        steps = [t.step for t in turns]
        min_step = min(steps)
        max_step = max(steps)
        covers_steps = range(min_step, max_step + 1)

        # Extract objectives and decisions from all turn content
        all_objectives: list[str] = []
        all_decisions: list[str] = []

        for turn in turns:
            for obj in _extract_objectives(turn.content):
                if obj not in all_objectives:
                    all_objectives.append(obj)
            for dec in _extract_decisions(turn.content):
                if dec not in all_decisions:
                    all_decisions.append(dec)

        # Produce a compressed text summary
        text_parts: list[str] = []
        text_parts.append(
            f"Summary of steps {min_step}-{max_step} "
            f"({len(turns)} turns):"
        )

        if all_objectives:
            text_parts.append("Objectives: " + "; ".join(all_objectives))
        if all_decisions:
            text_parts.append("Decisions: " + "; ".join(all_decisions))

        # Include a brief content digest
        for turn in turns:
            # Truncate each turn's content for the summary text
            snippet = turn.content[:80].replace("\n", " ")
            if len(turn.content) > 80:
                snippet += "..."
            text_parts.append(f"  [{turn.role}@step{turn.step}]: {snippet}")

        text = "\n".join(text_parts)

        # Token count for the summary: approximate as significantly less
        # than the sum of covered turns (compression)
        original_tokens = sum(t.tokens for t in turns)
        # Use a rough estimate: summary tokens are ~30% of original,
        # with a minimum of 1 token
        summary_tokens = max(1, original_tokens * 3 // 10)

        return StructuredSummary(
            covers_steps=covers_steps,
            objectives=all_objectives,
            decisions=all_decisions,
            text=text,
            tokens=summary_tokens,
        )

    def apply_summarization(
        self,
        turns: list[Turn],
        context_items: list[ContextItem],
    ) -> tuple[StructuredSummary, list[ContextItem]]:
        """Summarize turns and replace covered raw turn items in the context.

        This performs the full checkpoint summarization workflow:
        1. Produces a StructuredSummary from the turns.
        2. Removes context items of kind "turn" that correspond to the
           covered step range.
        3. Adds a new context item of kind "summary" representing the
           structured summary.

        Args:
            turns: The turns to summarize (L1 content).
            context_items: The current context item list.

        Returns:
            A tuple of (summary, updated_context_items) where the covered
            turn items have been replaced by a single summary item.
        """
        summary = self.summarize(turns)

        # Remove turn items that are within the covered step range
        # We identify them by kind == "turn" and filter by token matching
        # Since ContextItems don't carry step info, we remove turn items
        # whose token count matches one of the covered turns
        covered_turn_tokens = {t.tokens for t in turns}
        covered_turn_count = len(turns)

        # Strategy: remove turn-kind items from the context that correspond
        # to the summarized turns. We remove the first N turn items where
        # N = len(turns), since turns are added in order.
        new_items: list[ContextItem] = []
        turns_removed = 0

        for item in context_items:
            if item.kind == "turn" and turns_removed < covered_turn_count:
                turns_removed += 1
                # Skip this item (it's being replaced by the summary)
            else:
                new_items.append(item)

        # Add the summary item
        summary_item = ContextItem(
            kind="summary",
            tokens=summary.tokens,
            priority=50,  # Medium-high priority for summaries
            pinned=False,
        )
        new_items.append(summary_item)

        return summary, new_items
