"""loomable.media - High-level multimodal media classes.

Provides ergonomic ``Image``, ``Audio``, ``Video``, and ``File`` classes that
accept URLs, file paths, raw bytes, or base64 strings as sources. Includes
convenience methods for saving, encoding, and converting to low-level
``MediaPart`` instances.

This module depends only on ``loomable.content`` (for ``MediaPart``, ``Modality``)
and the standard library. It does NOT depend on ``loomable.kernel``.
"""

from .types import Audio, File, Image, MediaResolveError, Video

__all__ = ["Image", "Audio", "Video", "File", "MediaResolveError"]
