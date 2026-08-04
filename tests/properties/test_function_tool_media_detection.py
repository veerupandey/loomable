# Feature: multimodal-io, Property 8: FunctionTool Media Detection
"""Property 8: FunctionTool Media Detection.

For any return value from a tool function that is a `_MediaBase` instance or a
list containing `_MediaBase` instances, the resulting `ToolResult` SHALL have
`metadata["media"]` containing all those instances, AND `content` SHALL be a
non-empty string representation. For any non-media return value, `metadata`
SHALL NOT contain a `"media"` key.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.tools import FunctionTool, tool
from loomable.kernel.models import ToolResult
from loomable.media.types import (
    _MediaBase,
    Audio,
    File,
    Image,
    Video,
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: generate a random URL string
urls = st.from_regex(r"https://example\.com/[a-z0-9]{1,20}\.(png|jpg|wav|mp4)", fullmatch=True)

# Strategy: generate random content bytes (non-empty)
content_bytes = st.binary(min_size=1, max_size=100)

# Strategy: generate a single valid media instance (Image, Audio, Video, or File)
# Each media type is constructed with either a url or content (bytes) source.
media_instances = st.one_of(
    # Image with url
    urls.map(lambda u: Image(url=u)),
    # Image with content bytes
    content_bytes.map(lambda b: Image(content=b, format="png")),
    # Audio with url
    urls.map(lambda u: Audio(url=u)),
    # Audio with content bytes
    content_bytes.map(lambda b: Audio(content=b, format="wav")),
    # Video with url
    urls.map(lambda u: Video(url=u)),
    # Video with content bytes
    content_bytes.map(lambda b: Video(content=b, format="mp4")),
    # File with url
    urls.map(lambda u: File(url=u)),
    # File with content bytes
    content_bytes.map(lambda b: File(content=b, format="bin")),
)

# Strategy: generate a list of media instances (at least one)
media_lists = st.lists(media_instances, min_size=1, max_size=5)

# Strategy: generate non-media values (strings, ints, dicts, floats, lists of non-media)
non_media_values = st.one_of(
    st.text(min_size=0, max_size=50),
    st.integers(min_value=-(2**31), max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
    st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.integers(min_value=-100, max_value=100),
        max_size=5,
    ),
    st.lists(st.integers(min_value=-100, max_value=100), max_size=5),
    st.lists(st.text(min_size=1, max_size=10), max_size=5),
)


# ---------------------------------------------------------------------------
# Property tests: Single media return detection
# ---------------------------------------------------------------------------


class TestSingleMediaReturnDetection:
    """When a tool returns a single _MediaBase instance, ToolResult has metadata["media"]."""

    @settings(max_examples=100, deadline=None)
    @given(media=media_instances)
    @pytest.mark.asyncio
    async def test_single_media_detected_in_metadata(self, media: _MediaBase) -> None:
        """A single _MediaBase return value is stored in metadata["media"] as a list."""
        captured = {"val": media}

        @tool
        def media_tool() -> Any:
            """Return a media instance."""
            return captured["val"]

        result = await media_tool.invoke({})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert "media" in result.metadata
        assert result.metadata["media"] == [media]

    @settings(max_examples=100, deadline=None)
    @given(media=media_instances)
    @pytest.mark.asyncio
    async def test_single_media_content_is_nonempty_string(self, media: _MediaBase) -> None:
        """When media is detected, content is a non-empty string representation."""
        captured = {"val": media}

        @tool
        def media_tool() -> Any:
            """Return a media instance."""
            return captured["val"]

        result = await media_tool.invoke({})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert isinstance(result.content, str)
        assert len(result.content) > 0


# ---------------------------------------------------------------------------
# Property tests: List with media instances detection
# ---------------------------------------------------------------------------


class TestListMediaReturnDetection:
    """When a tool returns a list with _MediaBase instances, all are captured."""

    @settings(max_examples=100, deadline=None)
    @given(media_list=media_lists)
    @pytest.mark.asyncio
    async def test_list_media_all_captured_in_metadata(
        self, media_list: list[_MediaBase]
    ) -> None:
        """All _MediaBase instances in a returned list are stored in metadata["media"]."""
        captured = {"val": media_list}

        @tool
        def multi_media_tool() -> Any:
            """Return a list of media instances."""
            return captured["val"]

        result = await multi_media_tool.invoke({})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert "media" in result.metadata
        assert result.metadata["media"] == media_list

    @settings(max_examples=100, deadline=None)
    @given(media_list=media_lists)
    @pytest.mark.asyncio
    async def test_list_media_content_is_nonempty_string(
        self, media_list: list[_MediaBase]
    ) -> None:
        """When a list of media is returned, content is a non-empty string."""
        captured = {"val": media_list}

        @tool
        def multi_media_tool() -> Any:
            """Return a list of media instances."""
            return captured["val"]

        result = await multi_media_tool.invoke({})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert isinstance(result.content, str)
        assert len(result.content) > 0


# ---------------------------------------------------------------------------
# Property tests: Non-media return values are unchanged
# ---------------------------------------------------------------------------


class TestNonMediaReturnUnchanged:
    """When a tool returns a non-media value, metadata has no "media" key."""

    @settings(max_examples=100, deadline=None)
    @given(value=non_media_values)
    @pytest.mark.asyncio
    async def test_non_media_no_media_key(self, value: Any) -> None:
        """Non-media return values produce no 'media' key in metadata."""
        captured = {"val": value}

        @tool
        def plain_tool() -> Any:
            """Return a plain value."""
            return captured["val"]

        result = await plain_tool.invoke({})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert "media" not in result.metadata

    @settings(max_examples=100, deadline=None)
    @given(value=non_media_values)
    @pytest.mark.asyncio
    async def test_non_media_content_equals_return_value(self, value: Any) -> None:
        """Non-media return values are stored directly in ToolResult.content."""
        captured = {"val": value}

        @tool
        def plain_tool() -> Any:
            """Return a plain value."""
            return captured["val"]

        result = await plain_tool.invoke({})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert result.content == value
