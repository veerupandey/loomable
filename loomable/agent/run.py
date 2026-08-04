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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loomable.agent.events import Event
from loomable.content import AgentOutput, MediaPart
from loomable.content.parts import Modality
from loomable.media import Audio, File, Image, Video
from loomable.media.types import _MediaBase

if TYPE_CHECKING:
    from loomable.flow.loop import VerdictResult


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
