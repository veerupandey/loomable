"""loomable.content.parts - Typed multimodal content parts.

Defines the ``Modality`` enum and the frozen ``MediaPart`` dataclass along with
the ``Text`` / ``Image`` / ``Video`` convenience constructors. A ``MediaPart``
represents exactly one modality (text, image, or video), carries a media type
indicator, and references its payload as either inline ``data`` bytes or a
``uri`` (never both, never neither).

Depends only on the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import MediaPartError


class Modality(Enum):
    """A single multimodal modality."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


# Maps a modality to the required media_type prefix (e.g. "image/" for IMAGE).
_MODALITY_PREFIX: dict[Modality, str] = {
    Modality.TEXT: "text/",
    Modality.IMAGE: "image/",
    Modality.VIDEO: "video/",
    Modality.AUDIO: "audio/",
}


@dataclass(frozen=True)
class MediaPart:
    """A single unit of multimodal content.

    Invariants enforced in ``__post_init__``:
    - Exactly one of ``data`` / ``uri`` is set (Req 3.5).
    - ``media_type`` prefix matches ``modality`` (Req 3.6).
    """

    modality: Modality
    media_type: str
    data: bytes | None = None
    uri: str | None = None

    def __post_init__(self) -> None:
        # Req 3.5: exactly one of data/uri must be provided.
        has_data = self.data is not None
        has_uri = self.uri is not None
        if has_data and has_uri:
            raise MediaPartError(
                "MediaPart must have exactly one of 'data' or 'uri', not both."
            )
        if not has_data and not has_uri:
            raise MediaPartError(
                "MediaPart must have exactly one of 'data' or 'uri'; neither was provided."
            )

        # Req 3.6: media_type prefix must be consistent with the modality.
        expected_prefix = _MODALITY_PREFIX[self.modality]
        if not self.media_type.startswith(expected_prefix):
            raise MediaPartError(
                f"media_type '{self.media_type}' is inconsistent with modality "
                f"'{self.modality.value}' (expected prefix '{expected_prefix}')."
            )


def Text(text: str) -> MediaPart:
    """Construct a ``text/plain`` MediaPart from a string.

    The text is stored as UTF-8 encoded bytes in ``data``.
    """
    return MediaPart(
        modality=Modality.TEXT,
        media_type="text/plain",
        data=text.encode("utf-8"),
    )


def Image(
    *,
    data: bytes | None = None,
    uri: str | None = None,
    media_type: str = "image/png",
) -> MediaPart:
    """Construct an image MediaPart from inline bytes or a URI."""
    return MediaPart(
        modality=Modality.IMAGE,
        media_type=media_type,
        data=data,
        uri=uri,
    )


def Video(
    *,
    data: bytes | None = None,
    uri: str | None = None,
    media_type: str = "video/mp4",
) -> MediaPart:
    """Construct a video MediaPart from inline bytes or a URI."""
    return MediaPart(
        modality=Modality.VIDEO,
        media_type=media_type,
        data=data,
        uri=uri,
    )
