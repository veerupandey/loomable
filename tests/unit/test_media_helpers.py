"""Unit tests for loomable.agent.media - High-level image/video helpers.

Covers construction from raw bytes, a URI, and a file path (with media-type
inference from the extension), plus the exactly-one-source validation error.
"""

import pytest

from loomable.agent import image, video
from loomable.content import Modality


class TestImageHelper:
    """image() convenience wrapper over the content Image constructor."""

    def test_from_bytes(self):
        part = image(data=b"\x89PNG\r\n")
        assert part.modality is Modality.IMAGE
        assert part.data == b"\x89PNG\r\n"
        assert part.uri is None
        assert part.media_type == "image/png"

    def test_from_bytes_explicit_media_type(self):
        part = image(data=b"\xff\xd8\xff", media_type="image/jpeg")
        assert part.media_type == "image/jpeg"
        assert part.data == b"\xff\xd8\xff"

    def test_from_uri(self):
        part = image(uri="https://example.com/cat.png")
        assert part.modality is Modality.IMAGE
        assert part.uri == "https://example.com/cat.png"
        assert part.data is None
        assert part.media_type == "image/png"

    def test_from_path_infers_media_type(self, tmp_path):
        file_path = tmp_path / "photo.jpg"
        file_path.write_bytes(b"jpeg-bytes")
        part = image(file_path)
        assert part.modality is Modality.IMAGE
        assert part.data == b"jpeg-bytes"
        assert part.uri is None
        assert part.media_type == "image/jpeg"

    def test_from_path_unknown_extension_falls_back(self, tmp_path):
        file_path = tmp_path / "blob.bin"
        file_path.write_bytes(b"raw")
        part = image(file_path)
        assert part.media_type == "image/png"
        assert part.data == b"raw"

    def test_from_path_string(self, tmp_path):
        file_path = tmp_path / "pic.png"
        file_path.write_bytes(b"png-bytes")
        part = image(str(file_path))
        assert part.data == b"png-bytes"
        assert part.media_type == "image/png"

    def test_no_source_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            image()

    def test_multiple_sources_raise(self):
        with pytest.raises(ValueError, match="Exactly one"):
            image(data=b"x", uri="https://example.com/x.png")


class TestVideoHelper:
    """video() convenience wrapper over the content Video constructor."""

    def test_from_bytes(self):
        part = video(data=b"\x00\x00\x00\x18ftyp")
        assert part.modality is Modality.VIDEO
        assert part.data == b"\x00\x00\x00\x18ftyp"
        assert part.uri is None
        assert part.media_type == "video/mp4"

    def test_from_uri(self):
        part = video(uri="https://example.com/clip.mp4")
        assert part.modality is Modality.VIDEO
        assert part.uri == "https://example.com/clip.mp4"
        assert part.data is None
        assert part.media_type == "video/mp4"

    def test_from_path_infers_media_type(self, tmp_path):
        file_path = tmp_path / "clip.webm"
        file_path.write_bytes(b"webm-bytes")
        part = video(file_path)
        assert part.modality is Modality.VIDEO
        assert part.data == b"webm-bytes"
        assert part.media_type == "video/webm"

    def test_from_path_unknown_extension_falls_back(self, tmp_path):
        file_path = tmp_path / "movie.xyz"
        file_path.write_bytes(b"raw")
        part = video(file_path)
        assert part.media_type == "video/mp4"

    def test_no_source_raises(self):
        with pytest.raises(ValueError, match="Exactly one"):
            video()

    def test_multiple_sources_raise(self):
        with pytest.raises(ValueError, match="Exactly one"):
            video(path="a.mp4", data=b"x")
