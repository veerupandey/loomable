# Feature: multimodal-io, Property 5: Base64 and Data URI Encoding
"""Property 5: Base64 and Data URI Encoding.

For any Media_Class instance with resolvable content bytes and a known
mime_type, ``to_base64()`` SHALL equal ``base64.b64encode(content_bytes).decode()``,
and ``to_data_uri()`` SHALL equal ``f"data:{mime_type};base64,{to_base64()}"``.

**Validates: Requirements 2.2, 2.3**
"""

from __future__ import annotations

import base64

from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.media import Image, Audio, Video, File


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

random_bytes = st.binary(min_size=1, max_size=500)

known_mime_types = st.sampled_from([
    "image/png",
    "audio/wav",
    "video/mp4",
    "application/octet-stream",
])

media_classes = st.sampled_from([Image, Audio, Video, File])


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestBase64AndDataURIEncoding:
    """to_base64() and to_data_uri() produce correct encodings."""

    @settings(max_examples=100)
    @given(
        raw_bytes=random_bytes,
        mime=known_mime_types,
        cls=media_classes,
    )
    def test_to_base64_equals_standard_encoding(
        self, raw_bytes: bytes, mime: str, cls: type
    ) -> None:
        """to_base64() SHALL equal base64.b64encode(content_bytes).decode()."""
        media = cls(content=raw_bytes, mime_type=mime)
        expected = base64.b64encode(raw_bytes).decode()
        assert media.to_base64() == expected

    @settings(max_examples=100)
    @given(
        raw_bytes=random_bytes,
        mime=known_mime_types,
        cls=media_classes,
    )
    def test_to_data_uri_format(
        self, raw_bytes: bytes, mime: str, cls: type
    ) -> None:
        """to_data_uri() SHALL equal f"data:{mime_type};base64,{to_base64()}"."""
        media = cls(content=raw_bytes, mime_type=mime)
        b64 = media.to_base64()
        expected = f"data:{mime};base64,{b64}"
        assert media.to_data_uri() == expected
