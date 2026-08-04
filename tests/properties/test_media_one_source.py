# Feature: multimodal-io, Property 1: Exactly-One-Source Validation
"""Property 1: Exactly-One-Source Validation.

For any combination of `url`, `filepath`, and `content` parameters passed to a
Media_Class constructor, the constructor SHALL succeed if and only if exactly one
parameter is non-None; otherwise it SHALL raise `ValueError`.

**Validates: Requirements 1.1, 1.2, 1.3**
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.media.types import (
    Audio,
    File,
    Image,
    Video,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: optional URL (non-None or None)
optional_url = st.one_of(
    st.none(),
    st.from_regex(r"https://example\.com/[a-z0-9]{1,10}\.(png|jpg|wav|mp4)", fullmatch=True),
)

# Strategy: optional filepath (non-None or None)
optional_filepath = st.one_of(
    st.none(),
    st.from_regex(r"/tmp/[a-z0-9]{1,10}\.(png|jpg|wav|mp4|txt)", fullmatch=True),
)

# Strategy: optional content (non-None bytes or None)
optional_content = st.one_of(
    st.none(),
    st.binary(min_size=1, max_size=50),
)

# All four media classes to test
MEDIA_CLASSES = [Image, Audio, Video, File]

media_class_strategy = st.sampled_from(MEDIA_CLASSES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_non_none(*args) -> int:
    """Count how many of the provided args are not None."""
    return sum(1 for a in args if a is not None)


# ---------------------------------------------------------------------------
# Property tests: Exactly one source → constructor succeeds
# ---------------------------------------------------------------------------


class TestExactlyOneSourceSucceeds:
    """Constructor succeeds when exactly one of url/filepath/content is non-None."""

    @settings(max_examples=100)
    @given(
        media_cls=media_class_strategy,
        url=st.from_regex(r"https://example\.com/[a-z0-9]{1,10}\.(png|jpg|wav|mp4)", fullmatch=True),
    )
    def test_url_only_succeeds(self, media_cls, url: str) -> None:
        """Providing only url (filepath=None, content=None) constructs successfully."""
        instance = media_cls(url=url)
        assert instance.url == url
        assert instance.filepath is None
        assert instance.content is None

    @settings(max_examples=100)
    @given(
        media_cls=media_class_strategy,
        filepath=st.from_regex(r"/tmp/[a-z0-9]{1,10}\.(png|jpg|wav|mp4|txt)", fullmatch=True),
    )
    def test_filepath_only_succeeds(self, media_cls, filepath: str) -> None:
        """Providing only filepath (url=None, content=None) constructs successfully."""
        instance = media_cls(filepath=filepath)
        assert instance.url is None
        assert instance.filepath == filepath
        assert instance.content is None

    @settings(max_examples=100)
    @given(
        media_cls=media_class_strategy,
        content=st.binary(min_size=1, max_size=50),
    )
    def test_content_only_succeeds(self, media_cls, content: bytes) -> None:
        """Providing only content bytes (url=None, filepath=None) constructs successfully."""
        instance = media_cls(content=content)
        assert instance.url is None
        assert instance.filepath is None
        assert instance.content == content


# ---------------------------------------------------------------------------
# Property tests: Zero sources → raises ValueError
# ---------------------------------------------------------------------------


class TestZeroSourcesRaisesValueError:
    """Constructor raises ValueError when no source parameter is provided."""

    @settings(max_examples=100)
    @given(media_cls=media_class_strategy)
    def test_no_sources_raises_value_error(self, media_cls) -> None:
        """Passing url=None, filepath=None, content=None raises ValueError."""
        with pytest.raises(ValueError, match="none were given"):
            media_cls(url=None, filepath=None, content=None)

    @settings(max_examples=100)
    @given(media_cls=media_class_strategy)
    def test_default_construction_raises_value_error(self, media_cls) -> None:
        """Calling constructor with no arguments raises ValueError."""
        with pytest.raises(ValueError, match="none were given"):
            media_cls()


# ---------------------------------------------------------------------------
# Property tests: Multiple sources → raises ValueError naming conflicts
# ---------------------------------------------------------------------------


class TestMultipleSourcesRaisesValueError:
    """Constructor raises ValueError when more than one source is provided."""

    @settings(max_examples=100)
    @given(
        media_cls=media_class_strategy,
        url=st.from_regex(r"https://example\.com/[a-z0-9]{1,10}\.(png|jpg)", fullmatch=True),
        filepath=st.from_regex(r"/tmp/[a-z0-9]{1,10}\.(png|jpg)", fullmatch=True),
    )
    def test_url_and_filepath_raises(self, media_cls, url: str, filepath: str) -> None:
        """Providing both url and filepath raises ValueError naming the conflicts."""
        with pytest.raises(ValueError, match="conflicting parameters"):
            media_cls(url=url, filepath=filepath)

    @settings(max_examples=100)
    @given(
        media_cls=media_class_strategy,
        url=st.from_regex(r"https://example\.com/[a-z0-9]{1,10}\.(png|jpg)", fullmatch=True),
        content=st.binary(min_size=1, max_size=50),
    )
    def test_url_and_content_raises(self, media_cls, url: str, content: bytes) -> None:
        """Providing both url and content raises ValueError naming the conflicts."""
        with pytest.raises(ValueError, match="conflicting parameters"):
            media_cls(url=url, content=content)

    @settings(max_examples=100)
    @given(
        media_cls=media_class_strategy,
        filepath=st.from_regex(r"/tmp/[a-z0-9]{1,10}\.(png|jpg)", fullmatch=True),
        content=st.binary(min_size=1, max_size=50),
    )
    def test_filepath_and_content_raises(self, media_cls, filepath: str, content: bytes) -> None:
        """Providing both filepath and content raises ValueError naming the conflicts."""
        with pytest.raises(ValueError, match="conflicting parameters"):
            media_cls(filepath=filepath, content=content)

    @settings(max_examples=100)
    @given(
        media_cls=media_class_strategy,
        url=st.from_regex(r"https://example\.com/[a-z0-9]{1,10}\.(png|jpg)", fullmatch=True),
        filepath=st.from_regex(r"/tmp/[a-z0-9]{1,10}\.(png|jpg)", fullmatch=True),
        content=st.binary(min_size=1, max_size=50),
    )
    def test_all_three_sources_raises(self, media_cls, url: str, filepath: str, content: bytes) -> None:
        """Providing all three sources raises ValueError naming the conflicts."""
        with pytest.raises(ValueError, match="conflicting parameters"):
            media_cls(url=url, filepath=filepath, content=content)


# ---------------------------------------------------------------------------
# Property tests: General enumeration of all 8 combinations
# ---------------------------------------------------------------------------


class TestAllCombinations:
    """Enumerate all 8 None/non-None combinations for the 3 params."""

    @settings(max_examples=100)
    @given(
        media_cls=media_class_strategy,
        url=optional_url,
        filepath=optional_filepath,
        content=optional_content,
    )
    def test_constructor_succeeds_iff_exactly_one_source(
        self, media_cls, url, filepath, content
    ) -> None:
        """Constructor succeeds iff exactly one of url/filepath/content is non-None."""
        non_none_count = count_non_none(url, filepath, content)

        if non_none_count == 1:
            # Should succeed without error
            instance = media_cls(url=url, filepath=filepath, content=content)
            assert instance is not None
        elif non_none_count == 0:
            # Should raise ValueError indicating no source
            with pytest.raises(ValueError):
                media_cls(url=url, filepath=filepath, content=content)
        else:
            # non_none_count >= 2: should raise ValueError for conflicts
            with pytest.raises(ValueError):
                media_cls(url=url, filepath=filepath, content=content)
