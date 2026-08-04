# Feature: multimodal-io, Property 7: Auto-Coercion Dispatch
"""Property 7: Auto-Coercion Dispatch.

For any input value passed to ``_coerce_media_item``: if it is a ``str`` starting
with ``http://`` or ``https://`` the result SHALL have ``uri`` set to that string;
if it is a non-URL ``str`` the result SHALL encode data read from that filepath;
if it is ``bytes`` the result SHALL have ``data`` equal to those bytes; if it is
a ``Media_Class`` instance it SHALL convert via ``to_media_part()``.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.agent.builder import _coerce_media_item
from loomable.content.parts import MediaPart, Modality
from loomable.media import Image, Audio, Video


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Extension maps per modality — ensures MediaPart's media_type prefix check passes
_MODALITY_EXTENSIONS: dict[str, list[str]] = {
    "image": ["png", "jpg", "gif", "webp"],
    "video": ["mp4", "avi", "webm"],
    "audio": ["wav", "mp3", "ogg"],
}


@st.composite
def url_for_modality(draw, modality: str | None = None):
    """Generate a URL with an extension matching the target modality."""
    mod = modality or draw(st.sampled_from(["image", "video", "audio"]))
    scheme = draw(st.sampled_from(["http", "https"]))
    host = draw(st.from_regex(r"[a-z]{3,8}", fullmatch=True))
    tld = draw(st.sampled_from(["com", "org", "net", "io"]))
    path_segment = draw(st.from_regex(r"[a-z]{2,8}", fullmatch=True))
    ext = draw(st.sampled_from(_MODALITY_EXTENSIONS[mod]))
    return f"{scheme}://{host}.{tld}/{path_segment}.{ext}"


# Random bytes (non-empty to be a meaningful media payload)
random_bytes = st.binary(min_size=1, max_size=100)

# Target modalities supported by _coerce_media_item
target_modalities = st.sampled_from(["image", "video", "audio"])


@st.composite
def media_class_from_url_st(draw):
    """Generate a Media_Class instance constructed from a URL."""
    modality = draw(target_modalities)
    url = draw(url_for_modality(modality))
    cls_map = {"image": Image, "video": Video, "audio": Audio}
    return cls_map[modality](url=url)


@st.composite
def media_class_from_bytes_st(draw):
    """Generate a Media_Class instance constructed from bytes content."""
    modality = draw(target_modalities)
    content = draw(random_bytes)
    mime_map = {"image": "image/png", "video": "video/mp4", "audio": "audio/wav"}
    cls_map = {"image": Image, "video": Video, "audio": Audio}
    return cls_map[modality](content=content, mime_type=mime_map[modality])


# Combined Media_Class strategy
media_class_instances = st.one_of(media_class_from_url_st(), media_class_from_bytes_st())


# ---------------------------------------------------------------------------
# Property tests: URL strings → uri set
# ---------------------------------------------------------------------------


class TestURLStringCoercion:
    """URL strings (http:// or https://) coerce to MediaPart with uri set."""

    @settings(max_examples=100)
    @given(modality=target_modalities, data=st.data())
    def test_url_string_produces_uri_part(self, modality: str, data) -> None:
        """A URL string SHALL produce a MediaPart with uri set to that string."""
        url = data.draw(url_for_modality(modality))
        result = _coerce_media_item(url, modality)

        assert isinstance(result, MediaPart)
        assert result.uri == url
        assert result.data is None


# ---------------------------------------------------------------------------
# Property tests: bytes → data equals input
# ---------------------------------------------------------------------------


class TestBytesCoercion:
    """Raw bytes coerce to MediaPart with data equal to the input bytes."""

    @settings(max_examples=100)
    @given(data=random_bytes, modality=target_modalities)
    def test_bytes_produces_data_part(self, data: bytes, modality: str) -> None:
        """Bytes input SHALL produce a MediaPart with data equal to those bytes."""
        result = _coerce_media_item(data, modality)

        assert isinstance(result, MediaPart)
        assert result.data == data
        assert result.uri is None


# ---------------------------------------------------------------------------
# Property tests: Media_Class → converts via to_media_part()
# ---------------------------------------------------------------------------


class TestMediaClassCoercion:
    """Media_Class instances coerce via their to_media_part() method."""

    @settings(max_examples=100)
    @given(instance=media_class_instances, modality=target_modalities)
    def test_media_class_produces_same_as_to_media_part(
        self, instance, modality: str
    ) -> None:
        """A Media_Class instance SHALL convert via to_media_part()."""
        result = _coerce_media_item(instance, modality)
        expected = instance.to_media_part()

        assert isinstance(result, MediaPart)
        assert result == expected


# ---------------------------------------------------------------------------
# Property tests: non-URL filepath strings → data from file
# ---------------------------------------------------------------------------


class TestFilepathStringCoercion:
    """Non-URL strings coerce as filepaths, reading data from that file."""

    @settings(max_examples=100)
    @given(data=random_bytes, modality=target_modalities)
    def test_filepath_string_reads_file_content(
        self, data: bytes, modality: str
    ) -> None:
        """A non-URL string SHALL be treated as a filepath and data read from it.

        We create a real temp file to validate the round-trip: the coerced
        MediaPart's data SHALL equal the file contents.
        """
        # Determine extension based on modality
        ext_map = {"image": ".png", "video": ".mp4", "audio": ".wav"}
        ext = ext_map[modality]

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(data)
            filepath = f.name

        try:
            result = _coerce_media_item(filepath, modality)

            assert isinstance(result, MediaPart)
            assert result.data == data
            assert result.uri is None
        finally:
            Path(filepath).unlink(missing_ok=True)
