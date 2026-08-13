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
    Prefer high-level helpers (:func:`capabilities_for`, ``modalities=`` on
    :class:`~loomable.agent.Agent`) over constructing this with frozensets.
    Audio remains opt-in.
    """

    input: frozenset[Modality] = field(default_factory=_default_input_modalities)
    output: frozenset[Modality] = field(default_factory=lambda: frozenset({Modality.TEXT}))


_MODALITY_ALIASES: dict[str, Modality] = {
    "text": Modality.TEXT,
    "txt": Modality.TEXT,
    "image": Modality.IMAGE,
    "images": Modality.IMAGE,
    "img": Modality.IMAGE,
    "video": Modality.VIDEO,
    "videos": Modality.VIDEO,
    "audio": Modality.AUDIO,
    "sound": Modality.AUDIO,
}


def _parse_modality_token(token: str) -> Modality:
    key = token.strip().lower()
    if key not in _MODALITY_ALIASES:
        raise ValueError(
            f"Unknown modality {token!r}. "
            "Use: text, image, video, audio (or combinations like 'text+image')."
        )
    return _MODALITY_ALIASES[key]


def _parse_modality_set(spec: str | list[str] | set[str] | frozenset[str] | frozenset[Modality] | set[Modality]) -> frozenset[Modality]:
    """Parse a user-friendly modality spec into a frozenset of Modality."""
    if isinstance(spec, (set, frozenset)):
        items = list(spec)
        if items and all(isinstance(i, Modality) for i in items):
            return frozenset(items)  # type: ignore[arg-type]
        return frozenset(_parse_modality_token(str(i)) for i in items)
    if isinstance(spec, list):
        return frozenset(_parse_modality_token(str(i)) for i in spec)
    if isinstance(spec, str):
        raw = spec.strip().lower()
        if raw in {"text", "text-only", "text_only"}:
            return frozenset({Modality.TEXT})
        if raw in {"default", "multimodal", "media", "all"}:
            return _default_input_modalities()
        if raw in {"full", "any"}:
            return frozenset({Modality.TEXT, Modality.IMAGE, Modality.VIDEO, Modality.AUDIO})
        parts = [p for p in raw.replace(",", "+").replace("|", "+").split("+") if p.strip()]
        if not parts:
            raise ValueError(f"Empty modalities spec: {spec!r}")
        return frozenset(_parse_modality_token(p) for p in parts)
    raise TypeError(f"Unsupported modalities spec type: {type(spec).__name__}")


def capabilities_for(
    modalities: str | list[str] | set[str] | ModelCapabilities | None = None,
    *,
    input: str | list[str] | set[str] | None = None,  # noqa: A002
    output: str | list[str] | set[str] | None = None,
) -> ModelCapabilities:
    """Build :class:`ModelCapabilities` from plain strings — no frozensets required.

    Examples::

        capabilities_for("text")                  # text in/out
        capabilities_for("text+image+video")      # default-like input, text out
        capabilities_for(input="text+audio", output="text")
        capabilities_for(["text", "image"])
    """
    if isinstance(modalities, ModelCapabilities):
        return modalities
    if modalities is not None and (input is not None or output is not None):
        raise ValueError("Pass either modalities= or input=/output=, not both")

    if input is not None or output is not None:
        in_set = _parse_modality_set(input) if input is not None else _default_input_modalities()
        out_set = _parse_modality_set(output) if output is not None else frozenset({Modality.TEXT})
        return ModelCapabilities(input=in_set, output=out_set)

    if modalities is None:
        return ModelCapabilities()

    in_set = _parse_modality_set(modalities)
    return ModelCapabilities(input=in_set, output=frozenset({Modality.TEXT}))


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
