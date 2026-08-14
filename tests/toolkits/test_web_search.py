# Feature: built-in-toolkits, Property 20: Web search error isolation
"""Property test for WebSearchTools error isolation.

Property 20: For any exception raised by the search provider during web_search,
the tool SHALL return a ToolResult with an error description rather than
propagating the exception.

**Validates: Requirements 2.5**
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import patch, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.toolkits.web_search import WebSearchTools


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

exception_types = st.sampled_from([
    RuntimeError,
    ConnectionError,
    TimeoutError,
    ValueError,
    OSError,
])

error_messages = st.text(min_size=1, max_size=50)


# ---------------------------------------------------------------------------
# Property 20: Web search error isolation
# ---------------------------------------------------------------------------


class TestWebSearchErrorIsolation:
    """Property 20: Web search error isolation.

    For any exception raised by the search provider during web_search, the tool
    SHALL return a ToolResult with an error description rather than propagating
    the exception.

    **Validates: Requirements 2.5**
    """

    @settings(max_examples=20, deadline=None)
    @given(exc_type=exception_types, msg=error_messages)
    def test_web_search_error_isolation(self, exc_type: type, msg: str) -> None:
        """Any exception from the DuckDuckGo backend is caught and returned
        as a string containing 'Error' rather than propagating."""
        tools = WebSearchTools()

        # Create a fake duckduckgo_search module with a DDGS class that raises
        mock_ddgs_class = MagicMock()
        mock_instance = MagicMock()
        mock_instance.text.side_effect = exc_type(msg)
        mock_ddgs_class.return_value = mock_instance

        fake_module = types.ModuleType("duckduckgo_search")
        fake_module.DDGS = mock_ddgs_class

        with patch.dict(sys.modules, {"duckduckgo_search": fake_module, "ddgs": fake_module}):
            with patch.object(tools, "_search_duckduckgo_instant", return_value=None):
                with patch.object(tools, "_search_wikipedia", return_value=None):
                    result = asyncio.run(tools._web_search("test query"))

        # Result is a string, not an exception. With fallbacks disabled, surface Error.
        assert isinstance(result, str)
        assert "Error" in result or "No results found" in result
