"""Sanity unit tests for the low-level multimodal content parts (task 2.1)."""

from __future__ import annotations

import pytest

from loomable.content import Image, MediaPart, MediaPartError, Modality, Text, Video


class TestText:
    def test_text_stores_utf8_bytes(self) -> None:
        part = Text("hello")
        assert part.modality is Modality.TEXT
        assert part.media_type == "text/plain"
        assert part.data == b"hello"
        assert part.uri is None

    def test_text_unicode(self) -> None:
        part = Text("héllo \u2603")
        assert part.data == "héllo \u2603".encode("utf-8")


class TestImage:
    def test_image_from_data(self) -> None:
        part = Image(data=b"\x89PNG")
        assert part.modality is Modality.IMAGE
        assert part.media_type == "image/png"
        assert part.data == b"\x89PNG"

    def test_image_from_uri(self) -> None:
        part = Image(uri="https://example.com/x.jpg", media_type="image/jpeg")
        assert part.uri == "https://example.com/x.jpg"
        assert part.media_type == "image/jpeg"


class TestVideo:
    def test_video_from_data(self) -> None:
        part = Video(data=b"\x00\x00")
        assert part.modality is Modality.VIDEO
        assert part.media_type == "video/mp4"

    def test_video_from_uri(self) -> None:
        part = Video(uri="https://example.com/v.webm", media_type="video/webm")
        assert part.uri == "https://example.com/v.webm"


class TestMediaPartExclusivity:
    def test_neither_data_nor_uri_raises(self) -> None:
        with pytest.raises(MediaPartError):
            MediaPart(modality=Modality.IMAGE, media_type="image/png")

    def test_both_data_and_uri_raises(self) -> None:
        with pytest.raises(MediaPartError):
            MediaPart(
                modality=Modality.IMAGE,
                media_type="image/png",
                data=b"x",
                uri="https://example.com/x.png",
            )


class TestModalityConsistency:
    def test_mismatched_modality_and_media_type_raises(self) -> None:
        with pytest.raises(MediaPartError):
            MediaPart(modality=Modality.IMAGE, media_type="video/mp4", data=b"x")

    def test_text_prefix_required_for_text(self) -> None:
        with pytest.raises(MediaPartError):
            MediaPart(modality=Modality.TEXT, media_type="image/png", data=b"x")

    def test_consistent_prefix_ok(self) -> None:
        part = MediaPart(modality=Modality.VIDEO, media_type="video/mp4", data=b"x")
        assert part.modality is Modality.VIDEO


def test_media_part_error_is_value_error() -> None:
    assert issubclass(MediaPartError, ValueError)


def test_media_part_is_frozen() -> None:
    part = Text("x")
    with pytest.raises(Exception):
        part.data = b"y"  # type: ignore[misc]
