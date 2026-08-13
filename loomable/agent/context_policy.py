"""ContextPolicy — unified auto-compaction and context bounding.

Used by Agent (and optionally Workflow runtimes) so long-running jobs stay
within the model window without losing pinned facts or the active plan.

L1 = live turns / working context
L2 = compacted summaries
L3 = durable notes / facts (out of band; not compacted here)
"""

from __future__ import annotations

__all__ = ["ContextPolicy", "CompactionResult"]

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class CompactionResult:
    """Outcome of one compaction pass."""

    compacted: bool
    turns_before: int
    turns_after: int
    summaries_added: int = 0
    reason: str = ""


@dataclass
class ContextPolicy:
    """Policy for keeping long runs inside the context window.

    Parameters
    ----------
    memory_window:
        Recent L1 turns to keep uncompacted.
    compaction_threshold:
        Compact when ``len(l1) > compaction_threshold``.
    soft_limit_ratio:
        Fraction of ``token_budget`` that triggers proactive compaction
        before a model call (default 0.75).
    hard_limit_ratio:
        Fraction of budget that forces aggressive spill of bulky tool text.
    """

    memory_window: int = 8
    compaction_threshold: int = 16
    soft_limit_ratio: float = 0.75
    hard_limit_ratio: float = 0.92
    token_budget: int = 8192

    def should_compact_turns(self, l1_len: int) -> bool:
        return l1_len > self.compaction_threshold

    def estimate_tokens(self, messages: Sequence[Any]) -> int:
        """Cheap token estimate (~4 chars / token) for OpenAI-style messages."""
        import json

        try:
            return max(1, len(json.dumps(list(messages), default=str)) // 4)
        except (TypeError, ValueError):
            return max(1, sum(len(str(m)) for m in messages) // 4)

    def needs_soft_compaction(self, messages: Sequence[Any]) -> bool:
        return self.estimate_tokens(messages) >= int(self.token_budget * self.soft_limit_ratio)

    def needs_hard_spill(self, messages: Sequence[Any]) -> bool:
        return self.estimate_tokens(messages) >= int(self.token_budget * self.hard_limit_ratio)

    def compact_turns(
        self,
        l1: list[Any],
        *,
        pinned_steps: set[int] | frozenset[int] | None = None,
        summarizer: Any | None = None,
    ) -> tuple[list[Any], list[Any], CompactionResult]:
        """Compact overflow L1 turns into L2 summaries.

        Returns ``(new_l1, new_summaries, result)``.
        """
        pinned = pinned_steps or set()
        before = len(l1)
        if not self.should_compact_turns(before) or summarizer is None:
            return l1, [], CompactionResult(
                compacted=False,
                turns_before=before,
                turns_after=before,
                reason="below_threshold_or_no_summarizer",
            )

        window = self.memory_window if self.memory_window else before
        overflow_count = before - window
        if overflow_count <= 0:
            return l1, [], CompactionResult(
                compacted=False,
                turns_before=before,
                turns_after=before,
                reason="nothing_to_overflow",
            )

        overflow_slice = l1[:overflow_count]
        pinned_turns = [t for t in overflow_slice if getattr(t, "step", -1) in pinned]
        non_pinned = [t for t in overflow_slice if getattr(t, "step", -1) not in pinned]
        summaries: list[Any] = []
        if non_pinned:
            summaries.append(summarizer.summarize(non_pinned))
        new_l1 = pinned_turns + l1[overflow_count:]
        return new_l1, summaries, CompactionResult(
            compacted=bool(summaries),
            turns_before=before,
            turns_after=len(new_l1),
            summaries_added=len(summaries),
            reason="overflow_summarized",
        )

    def spill_bulky_tool_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tool_chars: int = 2000,
        force: bool = False,
    ) -> list[dict[str, Any]]:
        """Trim oversized tool payloads.

        When ``force`` is False (default), only runs if the hard token limit
        is exceeded. Oversized tool bodies are replaced with a truncated stub
        so long runs can continue.
        """
        if not force and not self.needs_hard_spill(messages):
            return messages

        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                out.append(msg)
                continue
            if role == "tool" and isinstance(content, str) and len(content) > max_tool_chars:
                stub = content[: max_tool_chars // 2] + (
                    f"\n...[truncated {len(content) - max_tool_chars} chars by ContextPolicy]..."
                )
                new_msg = dict(msg)
                new_msg["content"] = stub
                out.append(new_msg)
                continue
            if role == "tool" and isinstance(content, list):
                new_parts = []
                for part in content:
                    if (
                        isinstance(part, dict)
                        and part.get("type") == "text"
                        and isinstance(part.get("text"), str)
                        and len(part["text"]) > max_tool_chars
                    ):
                        text = part["text"]
                        new_parts.append(
                            {
                                "type": "text",
                                "text": text[: max_tool_chars // 2]
                                + f"\n...[truncated {len(text) - max_tool_chars} chars]...",
                            }
                        )
                    else:
                        new_parts.append(part)
                new_msg = dict(msg)
                new_msg["content"] = new_parts
                out.append(new_msg)
                continue
            out.append(msg)
        return out
