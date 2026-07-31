"""loomable.agent.routing - Complexity router for run strategy selection.

Provides a pre-flight classifier that selects the appropriate run strategy
(single-shot, tool-loop, or plan) based on input complexity signals. The
default implementation uses only the standard library (no ML dependencies).

An optional model-based classifier can be injected to override the heuristic.

Depends only on the standard library and ``loomable.content``.
"""

from __future__ import annotations

__all__ = ["RunStrategy", "ComplexityRouter"]

import re
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from loomable.content.message import AgentInput


class RunStrategy(Enum):
    """The run strategy selected by the complexity router."""

    SINGLE = "single"  # no tools needed; single-shot completion
    TOOL_LOOP = "tool_loop"  # default ReAct loop with tool use
    PLAN = "plan"  # escalate to plan_and_execute Flow (plan → map → synthesize)


class ModelClassifier(Protocol):
    """Protocol for an optional model-based classifier injected into the router."""

    def classify(self, agent_input: "AgentInput", *, has_tools: bool) -> RunStrategy: ...


# Cues that suggest multi-step or complex tasks requiring a plan.
_STEP_CUE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\band\s+then\b", re.IGNORECASE),
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bfor\s+each\b", re.IGNORECASE),
    re.compile(r"\bstep\s+by\s+step\b", re.IGNORECASE),
    re.compile(r"\bfirst\b.*\bthen\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\banalyze\s+and\b", re.IGNORECASE),
    re.compile(r"\bbreak\s+down\b", re.IGNORECASE),
    re.compile(r"\bdecompose\b", re.IGNORECASE),
    re.compile(r"\bmultiple\s+steps\b", re.IGNORECASE),
]

# Explicit numbered section lists (1. ... 2. ... 3. ...) often mean multi-part work.
_SECTION_LIST_PATTERN = re.compile(
    r"(?:(?:^|\n)\s*(?:\d+[\.\)]\s+\S+)){3,}",
    re.IGNORECASE,
)

# Threshold constants for the heuristic.
_TOKEN_LENGTH_PLAN_THRESHOLD = 500  # rough token count (chars / 4) above which we consider PLAN
_TOKEN_LENGTH_TOOL_THRESHOLD = 100  # below this, likely simple enough for SINGLE
_QUESTION_COUNT_PLAN_THRESHOLD = 3  # 3+ questions suggest multi-step reasoning
_STEP_CUE_PLAN_THRESHOLD = 2  # 2+ step cues suggest PLAN


class ComplexityRouter:
    """Cheap pre-flight classifier selecting single-shot vs tool-loop vs plan.

    Default is heuristic (stdlib only): token length, question-count, presence of
    conjunction/step cues ('and then', 'compare', 'for each'), and whether tools exist.
    An optional model-based classifier can be injected to override the heuristic.

    Opt-in via ``Agent(complexity_router=...)``.
    """

    def __init__(self, model_classifier: ModelClassifier | None = None) -> None:
        self._model_classifier = model_classifier

    def classify(self, agent_input: "AgentInput", *, has_tools: bool) -> RunStrategy:
        """Classify the input and return the appropriate run strategy.

        If a model-based classifier was injected, it is used instead of the
        heuristic. Otherwise, the heuristic analyzes:
        - Approximate token length of the input text
        - Number of questions (sentences ending with '?')
        - Presence of conjunction/step cues
        - Whether tools are available (``has_tools``)

        Returns
        -------
        RunStrategy
            SINGLE if input is simple and no tools are needed,
            TOOL_LOOP if tools are available and complexity is moderate,
            PLAN if complexity signals suggest multi-step reasoning.
        """
        # Delegate to model-based classifier if injected.
        if self._model_classifier is not None:
            return self._model_classifier.classify(agent_input, has_tools=has_tools)

        # Extract all text from the input messages.
        text = self._extract_text(agent_input)

        # Compute complexity signals.
        token_estimate = len(text) // 4  # rough char-to-token ratio
        question_count = text.count("?")
        step_cue_count = sum(1 for pat in _STEP_CUE_PATTERNS if pat.search(text))
        if _SECTION_LIST_PATTERN.search(text):
            step_cue_count += 1

        # Decision logic:
        # 1. If complexity signals are strong, escalate to PLAN (regardless of tools).
        if self._should_plan(token_estimate, question_count, step_cue_count):
            return RunStrategy.PLAN

        # 2. If tools are available, default to TOOL_LOOP.
        if has_tools:
            return RunStrategy.TOOL_LOOP

        # 3. No tools and not complex enough for PLAN → SINGLE.
        return RunStrategy.SINGLE

    def _should_plan(
        self, token_estimate: int, question_count: int, step_cue_count: int
    ) -> bool:
        """Determine whether the input warrants escalation to PLAN.

        Uses a scoring approach: each signal contributes a score, and if the
        total exceeds a threshold, PLAN is selected.
        """
        score = 0

        # Long inputs are more likely to need planning.
        if token_estimate >= _TOKEN_LENGTH_PLAN_THRESHOLD:
            score += 2
        elif token_estimate >= _TOKEN_LENGTH_TOOL_THRESHOLD:
            score += 1

        # Multiple questions suggest multi-part tasks.
        if question_count >= _QUESTION_COUNT_PLAN_THRESHOLD:
            score += 2
        elif question_count >= 2:
            score += 1

        # Step cues directly indicate multi-step reasoning.
        # Multiple cues are a very strong signal.
        if step_cue_count >= 3:
            score += 3
        elif step_cue_count >= _STEP_CUE_PLAN_THRESHOLD:
            score += 2
        elif step_cue_count >= 1:
            score += 1

        # Threshold: score >= 3 triggers PLAN.
        return score >= 3

    @staticmethod
    def _extract_text(agent_input: "AgentInput") -> str:
        """Extract concatenated text from all messages in the input."""
        from loomable.content.parts import Modality

        pieces: list[str] = []
        for message in agent_input.messages:
            for part in message.parts:
                if part.modality is Modality.TEXT and part.data is not None:
                    pieces.append(part.data.decode("utf-8"))
        return " ".join(pieces)
