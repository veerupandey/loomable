# Feature: multimodal-io, Property 2: Content Storage Round-Trip
"""Property 2: Content Storage Round-Trip.

For any sequence of random bytes, constructing a Media_Class with
``content=bytes`` SHALL store those exact bytes, and constructing with
``content=base64.b64encode(bytes).decode()`` SHALL also resolve to those
exact bytes when ``._resolved_bytes`` is accessed.

**Validates: Requirements 1.4, 1.5**
"""

from __future__ import annotations

import base64

from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.media import Image, Audio, Video, File


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

random_bytes = st.binary(min_size=0, max_size=1000)

media_classes = st.sampled_from([Image, Audio, Video, File])


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestContentStorageRoundTrip:
    """Content stored via bytes or base64 string round-trips correctly."""

    @settings(max_examples=100)
    @given(data=random_bytes, cls=media_classes)
    def test_bytes_content_preserved(self, data: bytes, cls: type) -> None:
        """Constructing with content=bytes stores those exact bytes in _resolved_bytes."""
        media = cls(content=data)
        assert media._resolved_bytes == data

    @settings(max_examples=100)
    @given(data=random_bytes, cls=media_classes)
    def test_base64_content_decoded_correctly(self, data: bytes, cls: type) -> None:
        """Constructing with content=base64_string decodes to the original bytes."""
        b64_str = base64.b64encode(data).decode()
        media = cls(content=b64_str)
        assert media._resolved_bytes == data
