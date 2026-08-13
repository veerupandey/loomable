"""loomable.content.capabilities - Model capabilities and kernel bridging.

This module holds:

- ``ModelCapabilities``: the declared set of input/output :class:`Modality` values a
  configured model supports. Input defaults to text+image+video; output defaults to
  text (Req 6.1/6.2 updated for multimodal-by-default).
- ``to_model_request``: map an :class:`AgentInput` of multimodal parts to the
  provider-agnostic ``ModelRequest`` whose ``messages`` carry an OpenAI-style,
  ordered content array (Req 4.3, 4.5).
- ``from_model_response``: rebuild an :class:`AgentOutput` from a kernel
  ``ModelResponse`` (text from ``content``, media from ``metadata["media"]``)
  (Req 5.2, 5.3, 5.5).

Depends only on the standard library and ``loomable.kernel.models`` (for the
provider-agnostic ``ModelRequest`` / ``ModelResponse`` shapes). It must not depend
on ``loomable.agent`` or ``loomable.serve``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

from loomable.kernel.models import ModelRequest, ModelResponse

from .message import AgentInput, AgentOutput
from .parts import Image, MediaPart, Modality, Text, Video


def _default_input_modalities() -> frozenset[Modality]:
    return frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO})


@dataclass(frozen=True)
class ModelCapabilities:
    """The modalities a configured model supports for input and output.

    Input defaults to text + image + video. Output defaults to text-only.
    Pass an explicit ``ModelCapabilities`` to lock an agent to text-only.
    Audio remains opt-in.
    """

    input: frozenset[Modality] = field(default_factory=_default_input_modalities)
    output: frozenset[Modality] = field(default_factory=lambda: frozenset({Modality.TEXT}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _data_uri(media_type: str, data: bytes) -> str:
    """Encode inline ``data`` as a base64 ``data:`` URI using ``media_type``."""
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _media_url(part: MediaPart) -> str:
    """Return the URL for an image/video part: its ``uri`` or a base64 data URI."""
    if part.uri is not None:
        return part.uri
    # data is guaranteed present when uri is absent (MediaPart invariant).
    return _data_uri(part.media_type, part.data or b"")


def _part_to_content(part: MediaPart) -> dict[str, Any]:
    """Map a single :class:`MediaPart` to a provider-agnostic content-array entry."""
    if part.modality is Modality.TEXT:
        text = part.data.decode("utf-8") if part.data is not None else ""
        return {"type": "text", "text": text}
    if part.modality is Modality.IMAGE:
        return {"type": "image_url", "image_url": {"url": _media_url(part)}}
    # Modality.VIDEO
    return {"type": "video_url", "video_url": {"url": _media_url(part)}}


def _coerce_modality(raw: Any, media_type: str | None) -> Modality:
    """Resolve a media entry's modality from an explicit value or its media type."""
    if isinstance(raw, Modality):
        return raw
    if isinstance(raw, str):
        try:
            return Modality(raw)
        except ValueError:
            pass
    if media_type:
        prefix = media_type.split("/", 1)[0]
        for modality in Modality:
            if modality.value == prefix:
                return modality
    # Media entries carry non-text media; default to image when unresolved.
    return Modality.IMAGE


def _media_from_entry(entry: dict[str, Any]) -> MediaPart:
    """Rebuild a :class:`MediaPart` from a ``metadata["media"]`` entry.

    Entries are dicts with a ``modality`` and/or ``media_type`` plus exactly one of
    ``data`` (base64-encoded bytes) or ``uri``.
    """
    media_type = entry.get("media_type")
    modality = _coerce_modality(entry.get("modality"), media_type)

    uri = entry.get("uri")
    data_b64 = entry.get("data")
    data = base64.b64decode(data_b64) if data_b64 is not None else None

    if modality is Modality.IMAGE:
        return Image(data=data, uri=uri, media_type=media_type or "image/png")
    if modality is Modality.VIDEO:
        return Video(data=data, uri=uri, media_type=media_type or "video/mp4")
    # Modality.TEXT
    if data is not None:
        return Text(data.decode("utf-8"))
    return MediaPart(
        modality=Modality.TEXT,
        media_type=media_type or "text/plain",
        uri=uri,
    )


# ---------------------------------------------------------------------------
# Kernel bridging
# ---------------------------------------------------------------------------


def to_model_request(
    agent_input: AgentInput,
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 1.0,
    max_tokens: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelRequest:
    """Map an :class:`AgentInput` to a provider-agnostic ``ModelRequest``.

    Each :class:`Message` becomes ``{"role": role, "content": [...]}`` where the
    content is an ordered array of provider-agnostic parts:

    - text → ``{"type": "text", "text": <decoded utf-8>}``
    - image → ``{"type": "image_url", "image_url": {"url": <uri or data URI>}}``
    - video → ``{"type": "video_url", "video_url": {"url": <uri or data URI>}}``

    Message and part ordering is preserved (Req 4.3, 4.5). Inline ``data`` is encoded
    as a base64 ``data:`` URI using the part's ``media_type``.
    """
    messages: list[dict[str, Any]] = [
        {
            "role": message.role,
            "content": [_part_to_content(part) for part in message.parts],
        }
        for message in agent_input.messages
    ]
    return ModelRequest(
        messages=messages,
        tools=list(tools) if tools is not None else [],
        temperature=temperature,
        max_tokens=max_tokens,
        metadata=dict(metadata) if metadata is not None else {},
    )


def from_model_response(response: ModelResponse) -> AgentOutput:
    """Rebuild an :class:`AgentOutput` from a kernel ``ModelResponse``.

    A leading text part is produced from ``response.content`` when non-empty, then a
    media part for each entry in ``response.metadata["media"]`` (Req 5.2, 5.5). A
    text-only response yields exactly one text part (Req 5.3). If there is neither
    content nor media, a single empty text part keeps the ``AgentOutput`` non-empty.
    """
    parts: list[MediaPart] = []
    if response.content:
        parts.append(Text(response.content))

    media_entries = response.metadata.get("media", []) if response.metadata else []
    for entry in media_entries:
        parts.append(_media_from_entry(entry))

    if not parts:
        parts.append(Text(""))

    return AgentOutput(parts=parts)
