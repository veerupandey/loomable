# Feature: multimodal-io, Property 3: Format and MIME Type Inference
"""Property 3: Format and MIME Type Inference.

For any filepath with a recognized file extension, the Media_Class SHALL infer
both a `format` attribute matching the extension (e.g., "png") and a `mime_type`
attribute matching the standard MIME mapping (e.g., "image/png"), and these two
SHALL be consistent with each other.

**Validates: Requirements 1.7, 1.8**
"""

from __future__ import annotations

import mimetypes

from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.media.types import Audio, File, Image, Video


# ---------------------------------------------------------------------------
# Known extension/MIME pairs that Python's mimetypes module reliably recognizes
# across platforms. We use only extensions with unambiguous MIME mappings.
# ---------------------------------------------------------------------------

KNOWN_EXTENSIONS: list[tuple[str, str]] = [
    ("png", "image/png"),
    ("jpg", "image/jpeg"),
    ("jpeg", "image/jpeg"),
    ("gif", "image/gif"),
    ("webp", "image/webp"),
    ("wav", "audio/x-wav"),
    ("mp3", "audio/mpeg"),
    ("mp4", "video/mp4"),
    ("avi", "video/x-msvideo"),
    ("pdf", "application/pdf"),
    ("txt", "text/plain"),
    ("html", "text/html"),
    ("css", "text/css"),
    ("json", "application/json"),
    ("xml", "application/xml"),
]

# Filter to only include extensions that Python's mimetypes resolves on this platform
_VALID_PAIRS: list[tuple[str, str]] = []
for ext, expected_mime in KNOWN_EXTENSIONS:
    guessed, _ = mimetypes.guess_type(f"file.{ext}")
    if guessed is not None:
        _VALID_PAIRS.append((ext, guessed))

# Strategy: pick a known extension/mime pair
known_ext_mime = st.sampled_from(_VALID_PAIRS)

# Strategy: generate directory path prefixes
dir_prefixes = st.sampled_from([
    "/tmp/",
    "/home/user/docs/",
    "/var/data/media/",
    "C:/Users/dev/files/",
    "./relative/path/",
])

# Strategy: generate a filename stem (no dots, just alphanumeric)
filename_stems = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)

# Strategy: pick a media class to construct
media_classes = st.sampled_from([Image, Audio, Video, File])


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestFormatAndMimeTypeInference:
    """Property 3: Format and MIME type are correctly inferred from file extension."""

    @settings(max_examples=100)
    @given(
        ext_mime=known_ext_mime,
        prefix=dir_prefixes,
        stem=filename_stems,
        cls=media_classes,
    )
    def test_format_inferred_from_extension(
        self, ext_mime: tuple[str, str], prefix: str, stem: str, cls: type
    ) -> None:
        """When filepath has a known extension, format equals that extension."""
        ext, _expected_mime = ext_mime
        filepath = f"{prefix}{stem}.{ext}"

        instance = cls(filepath=filepath)

        assert instance.format == ext

    @settings(max_examples=100)
    @given(
        ext_mime=known_ext_mime,
        prefix=dir_prefixes,
        stem=filename_stems,
        cls=media_classes,
    )
    def test_mime_type_matches_standard_mapping(
        self, ext_mime: tuple[str, str], prefix: str, stem: str, cls: type
    ) -> None:
        """When filepath has a known extension, mime_type matches mimetypes.guess_type."""
        ext, expected_mime = ext_mime
        filepath = f"{prefix}{stem}.{ext}"

        instance = cls(filepath=filepath)

        assert instance.mime_type == expected_mime

    @settings(max_examples=100)
    @given(
        ext_mime=known_ext_mime,
        prefix=dir_prefixes,
        stem=filename_stems,
        cls=media_classes,
    )
    def test_format_and_mime_type_are_consistent(
        self, ext_mime: tuple[str, str], prefix: str, stem: str, cls: type
    ) -> None:
        """format and mime_type are consistent: guessing MIME from format yields mime_type."""
        ext, _expected_mime = ext_mime
        filepath = f"{prefix}{stem}.{ext}"

        instance = cls(filepath=filepath)

        # Consistency check: mime_type derived from format should match the stored mime_type
        guessed_from_format, _ = mimetypes.guess_type(f"file.{instance.format}")
        assert guessed_from_format == instance.mime_type
