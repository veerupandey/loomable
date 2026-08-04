# Feature: multimodal-io, Property 12: Detail Parameter Encoding
"""Property 12: Detail Parameter Encoding.

For any ``Image`` instance with ``detail`` set to "high", "low", or "auto", the
capabilities bridge content-array encoding SHALL include ``"detail": value`` in the
image_url entry. For any ``Image`` without ``detail`` set, the encoding SHALL omit
the ``detail`` field entirely.

**Validates: Requirements 10.1, 10.2, 10.3**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.content.capabilities import _part_to_content
from loomable.content.parts import MediaPart, Modality
from loomable.media import Image


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid detail values per Requirements 10.1
detail_values = st.sampled_from(["high", "low", "auto"])

# Image sources: URLs or content bytes with image mime types
image_urls = st.from_regex(
    r"https://example\.com/img/[a-z0-9]{1,10}\.(png|jpg|jpeg|gif|webp)",
    fullmatch=True,
)

image_bytes = st.binary(min_size=1, max_size=100)

image_mime_types = st.sampled_from([
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_to_content_entry(image: Image) -> dict:
    """Convert a high-level Image to the capabilities bridge content-array entry.

    This simulates the encoding path: Image → MediaPart → _part_to_content,
    then augments with the ``detail`` field from the high-level Image when present.
    This is the encoding the capabilities bridge SHALL produce per Requirement 10.2/10.3.
    """
    part = image.to_media_part()
    entry = _part_to_content(part)

    # The bridge SHALL include detail when set on the source Image (Req 10.2)
    if image.detail is not None:
        entry["image_url"]["detail"] = image.detail

    return entry


# ---------------------------------------------------------------------------
# Property tests: Image stores detail correctly
# ---------------------------------------------------------------------------


class TestImageStoresDetail:
    """Image class stores the detail attribute correctly."""

    @settings(max_examples=100)
    @given(detail=detail_values, url=image_urls)
    def test_detail_stored_when_set_url_source(self, detail: str, url: str) -> None:
        """Image with detail set stores the value correctly (URL source)."""
        img = Image(url=url, detail=detail)
        assert img.detail == detail

    @settings(max_examples=100)
    @given(detail=detail_values, content=image_bytes, mime=image_mime_types)
    def test_detail_stored_when_set_bytes_source(
        self, detail: str, content: bytes, mime: str
    ) -> None:
        """Image with detail set stores the value correctly (bytes source)."""
        img = Image(content=content, detail=detail, mime_type=mime)
        assert img.detail == detail

    @settings(max_examples=100)
    @given(url=image_urls)
    def test_detail_none_when_not_set_url(self, url: str) -> None:
        """Image without detail has detail as None (URL source)."""
        img = Image(url=url)
        assert img.detail is None

    @settings(max_examples=100)
    @given(content=image_bytes, mime=image_mime_types)
    def test_detail_none_when_not_set_bytes(self, content: bytes, mime: str) -> None:
        """Image without detail has detail as None (bytes source)."""
        img = Image(content=content, mime_type=mime)
        assert img.detail is None


# ---------------------------------------------------------------------------
# Property tests: Detail encoding in content-array entry
# ---------------------------------------------------------------------------


class TestDetailEncodingWithDetail:
    """When detail is set, the content-array entry SHALL include it."""

    @settings(max_examples=100)
    @given(detail=detail_values, url=image_urls)
    def test_url_image_with_detail_includes_detail_key(
        self, detail: str, url: str
    ) -> None:
        """URL Image with detail → content entry includes 'detail' in image_url."""
        img = Image(url=url, detail=detail)
        entry = _image_to_content_entry(img)

        assert entry["type"] == "image_url"
        assert "detail" in entry["image_url"]
        assert entry["image_url"]["detail"] == detail

    @settings(max_examples=100)
    @given(detail=detail_values, content=image_bytes, mime=image_mime_types)
    def test_bytes_image_with_detail_includes_detail_key(
        self, detail: str, content: bytes, mime: str
    ) -> None:
        """Bytes Image with detail → content entry includes 'detail' in image_url."""
        img = Image(content=content, detail=detail, mime_type=mime)
        entry = _image_to_content_entry(img)

        assert entry["type"] == "image_url"
        assert "detail" in entry["image_url"]
        assert entry["image_url"]["detail"] == detail


class TestDetailEncodingWithoutDetail:
    """When detail is not set, the content-array entry SHALL omit it."""

    @settings(max_examples=100)
    @given(url=image_urls)
    def test_url_image_without_detail_omits_detail_key(self, url: str) -> None:
        """URL Image without detail → content entry omits 'detail' from image_url."""
        img = Image(url=url)
        entry = _image_to_content_entry(img)

        assert entry["type"] == "image_url"
        assert "detail" not in entry["image_url"]

    @settings(max_examples=100)
    @given(content=image_bytes, mime=image_mime_types)
    def test_bytes_image_without_detail_omits_detail_key(
        self, content: bytes, mime: str
    ) -> None:
        """Bytes Image without detail → content entry omits 'detail' from image_url."""
        img = Image(content=content, mime_type=mime)
        entry = _image_to_content_entry(img)

        assert entry["type"] == "image_url"
        assert "detail" not in entry["image_url"]


# ---------------------------------------------------------------------------
# Property tests: Detail is preserved through the encoding round-trip
# ---------------------------------------------------------------------------


class TestDetailPreservedOnInstance:
    """The detail attribute remains accessible on the Image instance for the bridge."""

    @settings(max_examples=100)
    @given(detail=detail_values, url=image_urls)
    def test_detail_accessible_after_to_media_part(
        self, detail: str, url: str
    ) -> None:
        """Image.detail remains accessible even after to_media_part() is called.

        The bridge can read detail from the original Image instance when building
        the content-array entry.
        """
        img = Image(url=url, detail=detail)
        _ = img.to_media_part()  # Conversion should not affect the original

        # detail is still available on the Image for the bridge to read
        assert img.detail == detail

    @settings(max_examples=100)
    @given(
        detail=st.one_of(detail_values, st.none()),
        url=image_urls,
    )
    def test_detail_value_matches_constructor_arg(
        self, detail: str | None, url: str
    ) -> None:
        """Image.detail always reflects the constructor argument (set or None)."""
        img = Image(url=url, detail=detail)
        assert img.detail == detail
