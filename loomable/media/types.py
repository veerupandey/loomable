"""loomable.media.types - High-level media dataclasses.

Provides ergonomic ``Image``, ``Audio``, ``Video``, and ``File`` classes that
unify URL, filepath, raw bytes, and base64 source representations behind a
single interface with lazy resolution.

Depends only on ``loomable.content`` (for ``MediaPart``, ``Modality``) and stdlib.
"""

from __future__ import annotations

import base64
import mimetypes
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


from loomable.content.parts import MediaPart, Modality


#: Deterministic extension -> MIME map for common media types.
#: ``mimetypes.guess_type`` is platform/registry dependent (e.g. ``.webp`` is
#: unknown on many Windows setups), so we resolve well-known media extensions
#: explicitly first for plug-and-play consistency across platforms.
_EXT_MIME: dict[str, str] = {
    # images
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "svg": "image/svg+xml",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "ico": "image/x-icon",
    "heic": "image/heic",
    "heif": "image/heif",
    "avif": "image/avif",
    # audio
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
    "m4a": "audio/mp4",
    "weba": "audio/webm",
    # video
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "mkv": "video/x-matroska",
    "avi": "video/x-msvideo",
    "mpeg": "video/mpeg",
    "mpg": "video/mpeg",
}


def _guess_mime(ext: str) -> str | None:
    """Resolve a MIME type from a bare extension.

    Prefer the platform ``mimetypes`` database (so behavior matches system
    expectations where available), then fall back to an explicit map for
    common media types the platform may not register (e.g. ``.webp`` on
    many Windows setups). This keeps inference plug-and-play consistent
    without overriding a platform's own known mappings.
    """
    key = ext.lower().lstrip(".")
    guessed, _ = mimetypes.guess_type(f"file.{key}")
    if guessed:
        return guessed
    return _EXT_MIME.get(key)


class MediaResolveError(Exception):
    """Raised when media content cannot be resolved (file not found, URL unreachable)."""

    pass


@dataclass
class _MediaBase:
    """Base class for all high-level media types.

    Exactly one of ``url``, ``filepath``, or ``content`` must be provided.
    """

    url: str | None = None
    filepath: str | Path | None = None
    content: bytes | str | None = None
    format: str | None = None
    mime_type: str | None = None

    # Internal resolved state (lazy, not part of __init__)
    _resolved_bytes: bytes | None = field(init=False, repr=False, default=None)
    _resolved_path: Path | None = field(init=False, repr=False, default=None)

    # Subclasses set this as a ClassVar
    _modality: ClassVar[Modality | None] = None

    def __post_init__(self) -> None:
        # Validate exactly one source is provided
        sources = []
        if self.url is not None:
            sources.append("url")
        if self.filepath is not None:
            sources.append("filepath")
        if self.content is not None:
            sources.append("content")

        if len(sources) == 0:
            raise ValueError(
                "Exactly one of 'url', 'filepath', or 'content' must be provided; "
                "none were given."
            )
        if len(sources) > 1:
            raise ValueError(
                f"Exactly one of 'url', 'filepath', or 'content' must be provided; "
                f"got conflicting parameters: {', '.join(sources)}."
            )

        # Handle base64 string content
        if isinstance(self.content, str):
            try:
                self._resolved_bytes = base64.b64decode(self.content)
            except Exception as exc:
                raise ValueError(
                    f"'content' string could not be decoded as base64: {exc}"
                ) from exc
        elif isinstance(self.content, bytes):
            self._resolved_bytes = self.content

        # Handle filepath: resolve to absolute Path, defer I/O
        if self.filepath is not None:
            self._resolved_path = Path(self.filepath).resolve()

        # Auto-infer format and mime_type from file extension
        self._infer_format_and_mime()

    def _infer_format_and_mime(self) -> None:
        """Infer format and mime_type from file extension when not explicitly set."""
        # Determine file extension source
        ext: str | None = None

        if self._resolved_path is not None:
            ext = self._resolved_path.suffix.lstrip(".")
        elif self.url is not None:
            # Try to extract extension from URL path
            from urllib.parse import urlparse

            path = urlparse(self.url).path
            if "." in path.split("/")[-1]:
                ext = path.rsplit(".", 1)[-1].lower()

        # Infer format from extension
        if self.format is None and ext:
            self.format = ext

        # Infer mime_type from format or extension
        if self.mime_type is None:
            if self.format:
                guessed = _guess_mime(self.format)
                if guessed:
                    self.mime_type = guessed
            elif ext:
                guessed = _guess_mime(ext)
                if guessed:
                    self.mime_type = guessed

        # If we have mime_type but not format, infer format from mime_type
        if self.format is None and self.mime_type is not None:
            # e.g. "image/png" -> "png"
            parts = self.mime_type.split("/")
            if len(parts) == 2:
                self.format = parts[1]

    def _resolve_bytes(self) -> bytes:
        """Resolve content bytes lazily, caching the result.

        For filepath sources, reads the file. For URL sources, fetches via HTTP.
        For content sources, returns already-cached bytes.

        Raises:
            MediaResolveError: If the file is not found or the URL is unreachable.
        """
        if self._resolved_bytes is not None:
            return self._resolved_bytes

        if self._resolved_path is not None:
            try:
                self._resolved_bytes = self._resolved_path.read_bytes()
            except (FileNotFoundError, OSError) as exc:
                raise MediaResolveError(
                    f"Cannot resolve media from filepath '{self._resolved_path}': {exc}"
                ) from exc
        elif self.url is not None:
            try:
                with urllib.request.urlopen(self.url) as response:  # noqa: S310
                    self._resolved_bytes = response.read()
            except Exception as exc:
                raise MediaResolveError(
                    f"Cannot resolve media from URL '{self.url}': {exc}"
                ) from exc
        else:
            raise MediaResolveError(
                "Cannot resolve media bytes: no source available."
            )

        return self._resolved_bytes

    def save(self, path: str | Path) -> None:
        """Write the resolved content bytes to disk.

        Fetches URL or reads file if needed before writing.

        Args:
            path: Destination file path.

        Raises:
            MediaResolveError: If content cannot be resolved.
        """
        resolved = self._resolve_bytes()
        Path(path).write_bytes(resolved)

    def to_base64(self) -> str:
        """Return the content as a base64-encoded string.

        Raises:
            MediaResolveError: If content cannot be resolved.
        """
        resolved = self._resolve_bytes()
        return base64.b64encode(resolved).decode()

    def to_data_uri(self) -> str:
        """Return a complete data URI string.

        Format: ``data:{mime_type};base64,{base64_data}``

        Raises:
            MediaResolveError: If content cannot be resolved.
        """
        mime = self.mime_type or "application/octet-stream"
        return f"data:{mime};base64,{self.to_base64()}"

    def to_media_part(self) -> MediaPart:
        """Convert to a low-level ``MediaPart`` for kernel-layer interop.

        - URL source (no content/filepath): produces ``MediaPart(uri=url)``
        - Filepath/content source: resolves bytes, produces ``MediaPart(data=bytes)``

        Raises:
            MediaResolveError: If resolution is needed and fails.
        """
        modality = self._modality or Modality.TEXT
        mime = self.mime_type or "application/octet-stream"

        if self.url is not None and self._resolved_bytes is None:
            # URL source without already-resolved bytes: use uri
            return MediaPart(
                modality=modality,
                media_type=mime,
                uri=self.url,
            )
        else:
            # Filepath or content source: resolve to bytes
            resolved = self._resolve_bytes()
            return MediaPart(
                modality=modality,
                media_type=mime,
                data=resolved,
            )

    @classmethod
    def from_media_part(cls, part: MediaPart) -> "_MediaBase":
        """Wrap a ``MediaPart`` back into the appropriate high-level media class.

        - If ``part.uri`` is set: constructs with ``url=part.uri``
        - If ``part.data`` is set: constructs with ``content=part.data``

        When called on ``_MediaBase`` directly, uses the part's modality to
        determine which subclass to return. When called on a specific subclass
        (e.g. ``Image.from_media_part(...)``), returns that subclass.
        """
        # Determine the target class
        if cls is _MediaBase:
            target_cls = _MODALITY_TO_CLASS.get(part.modality, File)
        else:
            target_cls = cls

        if part.uri is not None:
            return target_cls(url=part.uri, mime_type=part.media_type)
        else:
            return target_cls(content=part.data, mime_type=part.media_type)


@dataclass
class Image(_MediaBase):
    """High-level image media class.

    Accepts a ``detail`` parameter for OpenAI-style detail level control
    ("high", "low", or "auto").
    """

    detail: str | None = None
    _modality: ClassVar[Modality] = Modality.IMAGE


@dataclass
class Audio(_MediaBase):
    """High-level audio media class.

    Accepts an optional ``duration`` (seconds) for informational purposes.
    """

    duration: float | None = None
    _modality: ClassVar[Modality] = Modality.AUDIO


@dataclass
class Video(_MediaBase):
    """High-level video media class.

    Accepts an optional ``duration`` (seconds) for informational purposes.
    """

    duration: float | None = None
    _modality: ClassVar[Modality] = Modality.VIDEO


@dataclass
class File(_MediaBase):
    """High-level generic file media class.

    Accepts an optional ``filename`` hint for the original filename.
    """

    filename: str | None = None
    _modality: ClassVar[Modality | None] = None


# Mapping from Modality to the appropriate subclass, used by
# _MediaBase.from_media_part() when called on the base class.
_MODALITY_TO_CLASS: dict[Modality, type[_MediaBase]] = {
    Modality.IMAGE: Image,
    Modality.AUDIO: Audio,
    Modality.VIDEO: Video,
}
