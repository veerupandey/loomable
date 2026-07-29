"""loomable.agent.media - High-level multimodal input helpers.

Convenience wrappers over the low-level ``loomable.content`` ``Image`` / ``Video``
constructors that build a ``MediaPart`` from one of three mutually exclusive
sources (Req 4.2):

- ``path`` - read the file bytes, inferring the ``media_type`` from the file
  extension when not supplied.
- ``data`` - use the provided raw bytes directly.
- ``uri`` - reference an external resource by URI.

Exactly one of ``path`` / ``data`` / ``uri`` must be provided; supplying zero or
more than one raises ``ValueError``.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from loomable.content import Image, MediaPart, Video

# Fallback media types used when extension-based inference fails.
_IMAGE_FALLBACK = "image/png"
_VIDEO_FALLBACK = "video/mp4"


def _read_path(path: str | Path, media_type: str | None, fallback: str) -> tuple[bytes, str]:
    """Read the bytes at ``path`` and resolve a media type.

    If ``media_type`` is None, infer it from the file extension via
    ``mimetypes.guess_type``. The guess is only accepted when it is consistent
    with the expected modality (its prefix matches ``fallback``'s prefix);
    otherwise ``fallback`` is used (e.g. an unknown extension guessed as
    ``application/octet-stream`` for an image falls back to ``image/png``).
    """
    file_path = Path(path)
    data = file_path.read_bytes()
    if media_type is None:
        expected_prefix = fallback.split("/", 1)[0] + "/"
        guessed, _ = mimetypes.guess_type(file_path.name)
        media_type = guessed if guessed and guessed.startswith(expected_prefix) else fallback
    return data, media_type


def _validate_single_source(
    path: str | Path | None, data: bytes | None, uri: str | None
) -> None:
    """Ensure exactly one of ``path`` / ``data`` / ``uri`` is provided."""
    provided = [name for name, value in (
        ("path", path),
        ("data", data),
        ("uri", uri),
    ) if value is not None]
    if len(provided) != 1:
        raise ValueError(
            "Exactly one of 'path', 'data', or 'uri' must be provided; "
            f"got {len(provided)} ({', '.join(provided) or 'none'})."
        )


def image(
    path: str | Path | None = None,
    *,
    data: bytes | None = None,
    uri: str | None = None,
    media_type: str | None = None,
) -> MediaPart:
    """Construct an image input ``MediaPart`` from a path, bytes, or URI (Req 4.2).

    Exactly one of ``path`` / ``data`` / ``uri`` must be provided. When ``path`` is
    given, the file bytes are read and ``media_type`` is inferred from the extension
    when not supplied (falling back to ``image/png``). For ``data`` / ``uri`` a None
    ``media_type`` defaults to ``image/png``.
    """
    _validate_single_source(path, data, uri)
    if path is not None:
        file_data, resolved = _read_path(path, media_type, _IMAGE_FALLBACK)
        return Image(data=file_data, media_type=resolved)
    if data is not None:
        return Image(data=data, media_type=media_type or _IMAGE_FALLBACK)
    return Image(uri=uri, media_type=media_type or _IMAGE_FALLBACK)


def video(
    path: str | Path | None = None,
    *,
    data: bytes | None = None,
    uri: str | None = None,
    media_type: str | None = None,
) -> MediaPart:
    """Construct a video input ``MediaPart`` from a path, bytes, or URI (Req 4.2).

    Exactly one of ``path`` / ``data`` / ``uri`` must be provided. When ``path`` is
    given, the file bytes are read and ``media_type`` is inferred from the extension
    when not supplied (falling back to ``video/mp4``). For ``data`` / ``uri`` a None
    ``media_type`` defaults to ``video/mp4``.
    """
    _validate_single_source(path, data, uri)
    if path is not None:
        file_data, resolved = _read_path(path, media_type, _VIDEO_FALLBACK)
        return Video(data=file_data, media_type=resolved)
    if data is not None:
        return Video(data=data, media_type=media_type or _VIDEO_FALLBACK)
    return Video(uri=uri, media_type=media_type or _VIDEO_FALLBACK)
