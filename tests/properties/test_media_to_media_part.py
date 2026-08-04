# Feature: multimodal-io, Property 6: to_media_part Conversion
"""Property 6: to_media_part Conversion.

For any valid Media_Class instance, ``to_media_part()`` SHALL produce a
``MediaPart`` where: the modality matches the class's modality, the media_type
matches the class's mime_type, and either ``data`` equals the resolved bytes
(for filepath/content sources) or ``uri`` equals the url (for URL sources).

**Validates: Requirements 2.5**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.content.parts import MediaPart, Modality
from loomable.media import Audio, File, Image, Video


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# MIME types valid for each media class (must match modality prefix)
image_mime_types = st.sampled_from(["image/png", "image/jpeg", "image/gif", "image/webp"])
audio_mime_types = st.sampled_from(["audio/wav", "audio/mpeg", "audio/ogg", "audio/flac"])
video_mime_types = st.sampled_from(["video/mp4", "video/webm", "video/avi", "video/mpeg"])
text_mime_types = st.sampled_from(["text/plain", "text/csv", "text/html", "text/xml"])

random_bytes = st.binary(min_size=1, max_size=500)

# URL strategy: generate valid-looking http(s) URLs
url_schemes = st.sampled_from(["http://", "https://"])
url_domains = st.sampled_from(["example.com", "cdn.test.io", "media.host.org"])
url_paths = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="/-_"),
    min_size=1,
    max_size=30,
)


@st.composite
def url_strategy(draw: st.DrawFn) -> str:
    """Generate a valid-looking URL string."""
    scheme = draw(url_schemes)
    domain = draw(url_domains)
    path = draw(url_paths)
    return f"{scheme}{domain}/{path}"


@st.composite
def image_with_url(draw: st.DrawFn) -> Image:
    """Generate an Image with url source."""
    url = draw(url_strategy())
    mime = draw(image_mime_types)
    return Image(url=url, mime_type=mime)


@st.composite
def image_with_content(draw: st.DrawFn) -> Image:
    """Generate an Image with content (bytes) source."""
    data = draw(random_bytes)
    mime = draw(image_mime_types)
    return Image(content=data, mime_type=mime)


@st.composite
def audio_with_url(draw: st.DrawFn) -> Audio:
    """Generate an Audio with url source."""
    url = draw(url_strategy())
    mime = draw(audio_mime_types)
    return Audio(url=url, mime_type=mime)


@st.composite
def audio_with_content(draw: st.DrawFn) -> Audio:
    """Generate an Audio with content (bytes) source."""
    data = draw(random_bytes)
    mime = draw(audio_mime_types)
    return Audio(content=data, mime_type=mime)


@st.composite
def video_with_url(draw: st.DrawFn) -> Video:
    """Generate a Video with url source."""
    url = draw(url_strategy())
    mime = draw(video_mime_types)
    return Video(url=url, mime_type=mime)


@st.composite
def video_with_content(draw: st.DrawFn) -> Video:
    """Generate a Video with content (bytes) source."""
    data = draw(random_bytes)
    mime = draw(video_mime_types)
    return Video(content=data, mime_type=mime)


@st.composite
def file_with_url(draw: st.DrawFn) -> File:
    """Generate a File with url source and text/ mime_type.

    File has _modality=None, so to_media_part() uses Modality.TEXT as fallback.
    The mime_type must have a 'text/' prefix to satisfy MediaPart validation.
    """
    url = draw(url_strategy())
    mime = draw(text_mime_types)
    return File(url=url, mime_type=mime)


@st.composite
def file_with_content(draw: st.DrawFn) -> File:
    """Generate a File with content (bytes) source and text/ mime_type.

    File has _modality=None, so to_media_part() uses Modality.TEXT as fallback.
    The mime_type must have a 'text/' prefix to satisfy MediaPart validation.
    """
    data = draw(random_bytes)
    mime = draw(text_mime_types)
    return File(content=data, mime_type=mime)


# Combined strategies for URL-source and content-source instances
url_source_instances = st.one_of(
    image_with_url(), audio_with_url(), video_with_url(), file_with_url()
)

content_source_instances = st.one_of(
    image_with_content(), audio_with_content(), video_with_content(), file_with_content()
)

all_instances = st.one_of(url_source_instances, content_source_instances)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestToMediaPartConversion:
    """to_media_part() produces a MediaPart with correct modality, media_type, and data/uri."""

    @settings(max_examples=100)
    @given(instance=url_source_instances)
    def test_url_source_produces_uri(self, instance) -> None:
        """For URL-source instances, result.uri == instance.url and result.data is None."""
        result = instance.to_media_part()

        assert isinstance(result, MediaPart)
        assert result.uri == instance.url
        assert result.data is None

    @settings(max_examples=100)
    @given(instance=content_source_instances)
    def test_content_source_produces_data(self, instance) -> None:
        """For content-source instances, result.data == instance._resolved_bytes and result.uri is None."""
        result = instance.to_media_part()

        assert isinstance(result, MediaPart)
        assert result.data == instance._resolved_bytes
        assert result.uri is None

    @settings(max_examples=100)
    @given(instance=all_instances)
    def test_modality_matches_class(self, instance) -> None:
        """For all instances, result.modality == expected modality for the class."""
        result = instance.to_media_part()

        # File._modality is None, to_media_part falls back to Modality.TEXT
        expected_modality = instance._modality or Modality.TEXT
        assert result.modality == expected_modality

    @settings(max_examples=100)
    @given(instance=all_instances)
    def test_media_type_matches_mime_type(self, instance) -> None:
        """For all instances, result.media_type == instance.mime_type."""
        result = instance.to_media_part()

        expected_mime = instance.mime_type or "application/octet-stream"
        assert result.media_type == expected_mime

    @settings(max_examples=100)
    @given(instance=all_instances)
    def test_file_class_uses_text_modality_fallback(self, instance) -> None:
        """File class with _modality=None uses Modality.TEXT as fallback in to_media_part()."""
        result = instance.to_media_part()

        if isinstance(instance, File):
            assert result.modality == Modality.TEXT
        elif isinstance(instance, Image):
            assert result.modality == Modality.IMAGE
        elif isinstance(instance, Audio):
            assert result.modality == Modality.AUDIO
        elif isinstance(instance, Video):
            assert result.modality == Modality.VIDEO
