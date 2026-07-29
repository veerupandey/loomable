"""loomable.agent.summarize - Model-based summarizer for the agent harness.

Implements the kernel Summarizer.summarize(turns) -> StructuredSummary contract
via a model call, with a synchronous bridge and a kernel-style regex fallback
when the model call fails.

Requirements covered: 5.1, 5.2, 5.3
"""

from __future__ import annotations

__all__ = ["LLMSummarizer"]

import asyncio
import re
import threading
from typing import TYPE_CHECKING

from loomable.kernel.models import ModelRequest, ModelResponse, StructuredSummary, Turn

if TYPE_CHECKING:
    from loomable.kernel.contracts import ModelProvider


# ---------------------------------------------------------------------------
# Summarization prompt template
# ---------------------------------------------------------------------------

_SUMMARIZE_SYSTEM = (
    "You are a precise summarizer. Given a conversation excerpt, produce a "
    "structured summary with the following sections on separate lines:\n"
    "OBJECTIVES: <semicolon-separated list of objectives/goals>\n"
    "DECISIONS: <semicolon-separated list of decisions made>\n"
    "SUMMARY: <concise narrative summary of what happened>\n"
    "If there are no objectives or decisions, write NONE for that field."
)


def _render_turns_prompt(turns: list[Turn]) -> str:
    """Render a list of turns into a user prompt for the summarizer model."""
    lines: list[str] = []
    for turn in turns:
        lines.append(f"[{turn.role} @ step {turn.step}]: {turn.content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_OBJ_PATTERN = re.compile(r"^OBJECTIVES:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_DEC_PATTERN = re.compile(r"^DECISIONS:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
_SUM_PATTERN = re.compile(r"^SUMMARY:\s*(.+)$", re.MULTILINE | re.IGNORECASE)


def _parse_model_response(text: str) -> tuple[list[str], list[str], str]:
    """Parse the model's structured response into (objectives, decisions, summary_text).

    Returns empty lists / the raw text as fallback if sections are missing.
    """
    objectives: list[str] = []
    decisions: list[str] = []
    summary_text = text.strip()

    obj_match = _OBJ_PATTERN.search(text)
    if obj_match:
        raw = obj_match.group(1).strip()
        if raw.upper() != "NONE":
            objectives = [o.strip() for o in raw.split(";") if o.strip()]

    dec_match = _DEC_PATTERN.search(text)
    if dec_match:
        raw = dec_match.group(1).strip()
        if raw.upper() != "NONE":
            decisions = [d.strip() for d in raw.split(";") if d.strip()]

    sum_match = _SUM_PATTERN.search(text)
    if sum_match:
        summary_text = sum_match.group(1).strip()

    return objectives, decisions, summary_text


# ---------------------------------------------------------------------------
# Kernel-style regex fallback (mirrors loomable.kernel.summarizer logic)
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


def _fallback_summary(turns: list[Turn]) -> StructuredSummary:
    """Produce a kernel-style regex-based summary as a fallback.

    This mirrors the kernel Summarizer logic so compaction never breaks.
    """
    steps = [t.step for t in turns]
    min_step = min(steps)
    max_step = max(steps)
    covers_steps = range(min_step, max_step + 1)

    all_objectives: list[str] = []
    all_decisions: list[str] = []

    for turn in turns:
        for obj in _extract_objectives(turn.content):
            if obj not in all_objectives:
                all_objectives.append(obj)
        for dec in _extract_decisions(turn.content):
            if dec not in all_decisions:
                all_decisions.append(dec)

    text_parts: list[str] = [
        f"Summary of steps {min_step}-{max_step} ({len(turns)} turns):"
    ]
    if all_objectives:
        text_parts.append("Objectives: " + "; ".join(all_objectives))
    if all_decisions:
        text_parts.append("Decisions: " + "; ".join(all_decisions))

    for turn in turns:
        snippet = turn.content[:80].replace("\n", " ")
        if len(turn.content) > 80:
            snippet += "..."
        text_parts.append(f"  [{turn.role}@step{turn.step}]: {snippet}")

    text = "\n".join(text_parts)

    # Token estimate: ~30% of original, minimum 1
    original_tokens = sum(t.tokens for t in turns)
    summary_tokens = max(1, original_tokens * 3 // 10)

    return StructuredSummary(
        covers_steps=covers_steps,
        objectives=all_objectives,
        decisions=all_decisions,
        text=text,
        tokens=summary_tokens,
    )


# ---------------------------------------------------------------------------
# Synchronous bridge for async model calls
# ---------------------------------------------------------------------------


def _run_async_sync(coro):
    """Run an async coroutine synchronously.

    Uses asyncio.run when no event loop is running. When an event loop is
    already running (e.g., inside an async context), uses a dedicated worker
    thread with its own event loop via run_coroutine_threadsafe.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop running — safe to use asyncio.run
        return asyncio.run(coro)

    # A loop is already running — run in a worker thread
    result_future = asyncio.run_coroutine_threadsafe(coro, loop)
    try:
        return result_future.result(timeout=120)
    except Exception:
        # If run_coroutine_threadsafe doesn't work (e.g., the loop isn't in
        # the right state), fall back to a new thread with its own loop.
        result = None
        exception = None

        def _worker():
            nonlocal result, exception
            try:
                result = asyncio.run(coro)
            except Exception as exc:
                exception = exc

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join(timeout=120)

        if exception is not None:
            raise exception
        return result


# ---------------------------------------------------------------------------
# LLMSummarizer
# ---------------------------------------------------------------------------


class LLMSummarizer:
    """Model-based summarizer with the SAME .summarize(turns) -> StructuredSummary
    contract as the kernel Summarizer, so it drops into the existing _persist_session
    compaction path with no other change (duck-typed; builder accepts either).

    On model call failure, falls back to a kernel-style regex summary so compaction
    never breaks the run.
    """

    def __init__(self, model: "ModelProvider", *, max_tokens: int = 512) -> None:
        self._model = model
        self._max_tokens = max_tokens

    def summarize(self, turns: list[Turn]) -> StructuredSummary:
        """Render turns -> a summarization prompt, call the model, and parse the
        response into StructuredSummary(covers_steps, objectives, decisions, text,
        tokens). Falls back to a kernel-style regex summary if the model call fails,
        so compaction never breaks the run.

        Args:
            turns: The list of turns to summarize. Must not be empty.

        Returns:
            A StructuredSummary covering the step range of the input turns.
        """
        if not turns:
            raise ValueError("Cannot summarize an empty list of turns")

        # Try model-based summarization
        try:
            response = self._call_model(turns)
            return self._parse_response(response, turns)
        except Exception:
            # Fallback to kernel-style regex summary on any failure
            return _fallback_summary(turns)

    def _call_model(self, turns: list[Turn]) -> ModelResponse:
        """Call the model synchronously via the async bridge."""
        request = ModelRequest(
            messages=[
                {"role": "system", "content": _SUMMARIZE_SYSTEM},
                {"role": "user", "content": _render_turns_prompt(turns)},
            ],
            max_tokens=self._max_tokens,
            temperature=0.3,
        )

        async def _do_complete():
            return await self._model.complete(request)

        return _run_async_sync(_do_complete())

    def _parse_response(self, response: ModelResponse, turns: list[Turn]) -> StructuredSummary:
        """Parse a model response into a StructuredSummary."""
        steps = [t.step for t in turns]
        min_step = min(steps)
        max_step = max(steps)
        covers_steps = range(min_step, max_step + 1)

        text = response.content.strip() if response.content else ""

        # Parse structured sections from the model response
        objectives, decisions, summary_text = _parse_model_response(text)

        # Use the full response text if no SUMMARY: section was found separately
        if not summary_text:
            summary_text = text if text else f"Summary of steps {min_step}-{max_step}"

        # Estimate token count from the response usage or the summary text length
        tokens = response.usage.get("output_tokens", 0) or response.usage.get(
            "completion_tokens", 0
        )
        if tokens <= 0:
            # Cheap fallback estimator: ~4 chars per token
            tokens = max(1, len(summary_text) // 4)

        return StructuredSummary(
            covers_steps=covers_steps,
            objectives=objectives,
            decisions=decisions,
            text=summary_text,
            tokens=tokens,
        )
