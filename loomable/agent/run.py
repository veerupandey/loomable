"""loomable.agent.run - Run result and streaming chunk models.

These are the return shapes of the high-level run flow:

- :class:`RunResult` is what :meth:`BuiltAgent.arun` returns: the produced
  :class:`~loomable.content.AgentOutput` plus run metadata (session id, token
  usage, and tool activity). ``sub_results`` and ``structured`` are placeholders
  for later tasks (orchestration / structured output) and default to ``None``.
- :class:`RunChunk` is what :meth:`BuiltAgent.astream` yields: an incremental
  output part plus a ``done`` flag marking the final chunk (Req 1.5).

Depends only on the standard library and ``loomable.content``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loomable.agent.events import Event
from loomable.content import AgentOutput, MediaPart
from loomable.content.parts import Modality
from loomable.media import Audio, File, Image, Video
from loomable.media.types import _MediaBase

if TYPE_CHECKING:
    from loomable.flow.loop import VerdictResult


def _tool_name(outcome: Any) -> str:
    if getattr(outcome, "result", None) is not None:
        return str((outcome.result.metadata or {}).get("tool_name", ""))
    if getattr(outcome, "error", None) is not None:
        details = getattr(outcome.error, "details", None) or {}
        return str(details.get("tool_name", ""))
    return ""


def _tool_content(outcome: Any) -> str:
    if getattr(outcome, "result", None) is not None:
        content = outcome.result.content
        return "" if content is None else str(content)
    return ""


def extract_thoughts(tool_activity: list[Any] | None) -> list[str]:
    """Return text contents from ``think`` tool calls."""
    thoughts: list[str] = []
    for outcome in tool_activity or []:
        if _tool_name(outcome) == "think":
            text = _tool_content(outcome).strip()
            if text:
                thoughts.append(text)
    return thoughts


def extract_plan_steps(tool_activity: list[Any] | None) -> list[str] | None:
    """Best-effort plan steps from ``plan`` tool metadata or JSON content."""
    for outcome in tool_activity or []:
        if _tool_name(outcome) != "plan":
            continue
        if getattr(outcome, "result", None) is None:
            continue
        meta = outcome.result.metadata or {}
        steps = meta.get("plan_steps")
        if isinstance(steps, list) and steps:
            return [str(s) for s in steps]
        content = _tool_content(outcome).strip()
        if content.startswith("{"):
            try:
                data = json.loads(content)
                raw = data.get("plan_steps") or data.get("steps")
                if isinstance(raw, list) and raw:
                    return [str(s) for s in raw]
            except (json.JSONDecodeError, TypeError):
                pass
    return None


@dataclass
class RunResult:
    """The result of a single agent run (Req 1.4).

    Attributes
    ----------
    output:
        The produced :class:`~loomable.content.AgentOutput`.
    session_id:
        The session the run belongs to.
    usage:
        Token usage reported by the provider (input/output token counts).
    tool_activity:
        Tool outcomes observed during the run (empty for a plain single turn).
    sub_results:
        Per-sub-agent results for multi-agent orchestration (task 8); ``None`` here.
    structured:
        The parsed/validated structured object for structured output (task 9.1);
        ``None`` here.
    metadata:
        Optional run metadata dict. When tiered routing is active and a fallback
        is used, contains a ``"tier_substitution"`` key with the
        :class:`~loomable.kernel.model_router.TierSubstitution` record (Req 7.1–7.3).
    thoughts:
        Contents of ``think`` tool calls made during the run.
    plan:
        Plan steps from a ``plan`` tool call when available.
    reasoning:
        Native provider reasoning segments when exposed; otherwise empty unless
        populated from think-tool fallback by the run path.
    """

    output: AgentOutput
    session_id: str
    usage: dict[str, int] = field(default_factory=dict)
    tool_activity: list[Any] = field(default_factory=list)
    sub_results: dict[str, Any] | None = None
    structured: object | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: list[Event] = field(default_factory=list)
    verification: "VerdictResult | None" = None
    thoughts: list[str] = field(default_factory=list)
    plan: list[str] | None = None
    reasoning: list[str] = field(default_factory=list)

    # --- Convenience properties for multimodal output access (Req 6) ---

    @property
    def text(self) -> str:
        """Return the concatenated text from model output (Req 6.1)."""
        return self.output.text()

    @property
    def images(self) -> list[Image]:
        """Return all images: model-generated first, then tool-generated (Req 6.2, 6.6)."""
        return self._collect_media(Modality.IMAGE, Image)

    @property
    def audio(self) -> list[Audio]:
        """Return all audio: model-generated first, then tool-generated (Req 6.3, 6.6)."""
        return self._collect_media(Modality.AUDIO, Audio)

    @property
    def videos(self) -> list[Video]:
        """Return all videos: model-generated first, then tool-generated (Req 6.4, 6.6)."""
        return self._collect_media(Modality.VIDEO, Video)

    @property
    def files(self) -> list[File]:
        """Return all files from tool metadata only (Req 6.7).

        Models do not generate arbitrary files, so only tool metadata is checked.
        """
        result: list[File] = []
        for outcome in self.tool_activity:
            if outcome.result is not None:
                media_items = outcome.result.metadata.get("media", [])
                for item in media_items:
                    if isinstance(item, File):
                        result.append(item)
        return result

    def _collect_media(
        self, modality: Modality, media_cls: type[_MediaBase]
    ) -> list[Any]:
        """Collect media of a given modality from model output and tool metadata.

        Ordering: model-generated media first, then tool-generated media in
        invocation order (Req 6.6). Returns empty list when none present (Req 6.5).
        """
        result: list[Any] = []

        # 1. Model media first: iterate output parts filtered by modality
        for part in self.output.parts:
            if part.modality is modality:
                result.append(media_cls.from_media_part(part))

        # 2. Tool media second: iterate tool_activity in invocation order
        for outcome in self.tool_activity:
            if outcome.result is not None:
                media_items = outcome.result.metadata.get("media", [])
                for item in media_items:
                    if isinstance(item, media_cls):
                        result.append(item)

        return result


@dataclass
class RunChunk:
    """A single incremental output part yielded while streaming (Req 1.5).

    ``done`` marks the final chunk of a stream.
    """

    delta: MediaPart
    done: bool = False
