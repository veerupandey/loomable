# Feature: built-in-toolkits, Property 9: URL text extraction strips HTML tags
# Feature: built-in-toolkits, Property 10: HTTP error status returns error message
"""Property tests for URLTools.

Property 9: For any HTML content fetched from a URL (mocked), extract_text SHALL
return text containing no HTML tags, and when max_length is configured, the output
length SHALL not exceed that limit.

Property 10: For any HTTP response with a non-2xx status code, URLTools SHALL
return a result containing an error message that includes the status code.

**Validates: Requirements 5.2, 5.3, 5.7**
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.toolkits.url_tools import URLTools


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# HTML tag names commonly used in web pages
_TAG_NAMES = ["p", "div", "span", "h1", "h2", "h3", "a", "li", "strong", "em"]

# Strategy for plain text content (no angle brackets)
plain_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "Z"),
        blacklist_characters="<>",
    ),
    min_size=1,
    max_size=100,
)


@st.composite
def html_content(draw: st.DrawFn) -> tuple[str, list[str]]:
    """Generate HTML content wrapping plain text in random tags.

    Returns (html_string, list_of_plain_text_segments).
    """
    num_segments = draw(st.integers(min_value=1, max_value=5))
    segments: list[str] = []
    html_parts: list[str] = []

    for _ in range(num_segments):
        tag = draw(st.sampled_from(_TAG_NAMES))
        text = draw(plain_text)
        segments.append(text)
        html_parts.append(f"<{tag}>{text}</{tag}>")

    html = "<html><body>" + "".join(html_parts) + "</body></html>"
    return html, segments


# Strategy for non-2xx status codes
non_2xx_status_codes = st.one_of(
    st.integers(min_value=100, max_value=199),
    st.integers(min_value=300, max_value=599),
)

# Strategy for max_length values
max_length_values = st.integers(min_value=1, max_value=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_httpx_client(status_code: int, text: str) -> MagicMock:
    """Create a mock httpx.AsyncClient that returns a response with the given
    status code and text."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = text
    mock_response.is_redirect = False
    mock_response.headers = {}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


# ---------------------------------------------------------------------------
# Property 9: URL text extraction strips HTML tags
# ---------------------------------------------------------------------------


class TestExtractTextStripsHtmlTags:
    """Property 9: URL text extraction strips HTML tags.

    For any HTML content fetched from a URL (mocked), extract_text SHALL return
    text containing no HTML tags, and when max_length is configured, the output
    length SHALL not exceed that limit.

    **Validates: Requirements 5.2, 5.7**
    """

    @settings(max_examples=20)
    @given(data=st.data())
    async def test_extract_text_contains_no_html_tags(
        self, data: st.DataObject
    ) -> None:
        """extract_text result contains no < or > characters (no HTML tags)."""
        html, _segments = data.draw(html_content(), label="html_content")

        tools = URLTools()
        mock_client = _mock_httpx_client(200, html)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tools._extract_text("http://example.com")

        # No HTML tags in the result
        assert "<" not in result, f"Found '<' in result: {result!r}"
        assert ">" not in result, f"Found '>' in result: {result!r}"

    @settings(max_examples=20)
    @given(data=st.data())
    async def test_extract_text_respects_max_length(
        self, data: st.DataObject
    ) -> None:
        """When max_length is configured, extract_text output length does not
        exceed that limit."""
        html, _segments = data.draw(html_content(), label="html_content")
        max_length = data.draw(max_length_values, label="max_length")

        tools = URLTools(max_length=max_length)
        mock_client = _mock_httpx_client(200, html)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tools._extract_text("http://example.com")

        assert len(result) <= max_length, (
            f"Result length {len(result)} exceeds max_length {max_length}"
        )


# ---------------------------------------------------------------------------
# Property 10: HTTP error status returns error message
# ---------------------------------------------------------------------------


class TestHttpErrorStatusReturnsErrorMessage:
    """Property 10: HTTP error status returns error message.

    For any HTTP response with a non-2xx status code, URLTools SHALL return a
    result containing an error message that includes the status code.

    **Validates: Requirements 5.3**
    """

    @settings(max_examples=20)
    @given(status_code=non_2xx_status_codes)
    async def test_fetch_url_returns_error_with_status_code(
        self, status_code: int
    ) -> None:
        """fetch_url with non-2xx status returns error containing the status code."""
        tools = URLTools()
        mock_client = _mock_httpx_client(status_code, "")

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tools._fetch_url("http://example.com")

        # Result should contain "Error" indicator
        assert "Error" in result, f"Expected 'Error' in result: {result!r}"
        # Result should contain the status code number
        assert str(status_code) in result, (
            f"Expected status code '{status_code}' in result: {result!r}"
        )

    @settings(max_examples=20)
    @given(status_code=non_2xx_status_codes)
    async def test_extract_text_returns_error_with_status_code(
        self, status_code: int
    ) -> None:
        """extract_text with non-2xx status returns error containing the status code."""
        tools = URLTools()
        mock_client = _mock_httpx_client(status_code, "")

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await tools._extract_text("http://example.com")

        # Result should contain "Error" indicator
        assert "Error" in result, f"Expected 'Error' in result: {result!r}"
        # Result should contain the status code number
        assert str(status_code) in result, (
            f"Expected status code '{status_code}' in result: {result!r}"
        )
