# Feature: multimodal-io, Property 4: Save Round-Trip
"""Property-based test verifying that saving media to disk and reading back
yields the original bytes unchanged.

**Validates: Requirements 2.1**
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.media import Image, Audio, Video, File


# Strategy: pick one of the media classes and construct with random bytes
media_class_st = st.sampled_from([Image, Audio, Video, File])


@settings(max_examples=100)
@given(
    raw_bytes=st.binary(min_size=1, max_size=1000),
    cls=media_class_st,
)
def test_save_roundtrip_preserves_bytes(raw_bytes: bytes, cls: type) -> None:
    """For any Media_Class instance constructed from content=bytes,
    calling save(path) and then reading back the bytes from path
    SHALL yield the original bytes unchanged.
    """
    media = cls(content=raw_bytes)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp_path = Path(tmp.name)

    try:
        media.save(tmp_path)
        read_back = tmp_path.read_bytes()
        assert read_back == raw_bytes, (
            f"Round-trip failed: wrote {len(raw_bytes)} bytes, "
            f"read back {len(read_back)} bytes"
        )
    finally:
        # Clean up
        tmp_path.unlink(missing_ok=True)
