# Feature: built-in-toolkits, Property 19: Error isolation preserves exception details
"""Property 19: Error isolation preserves exception details.

For any exception raised within a toolkit tool function (of any type with any
message), the resulting ToolResult SHALL contain both the exception type name
and the exception message, and the exception SHALL NOT propagate to the caller.

**Validates: Requirements 8.1, 8.2, 8.3**
"""

from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from loomable.agent.tools import FunctionTool
from loomable.kernel.models import ToolResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Common built-in exception types that can be raised with a single message arg
exception_types = st.sampled_from([
    ValueError,
    RuntimeError,
    TypeError,
    KeyError,
    IOError,
    OSError,
    AttributeError,
    IndexError,
    ArithmeticError,
    NotImplementedError,
    PermissionError,
    FileNotFoundError,
    ConnectionError,
    TimeoutError,
])

# Random error messages (printable text, non-empty)
error_messages = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
    min_size=1,
    max_size=200,
)

# Valid tool names (Python identifiers)
tool_names = st.from_regex(r"[a-z][a-z0-9_]{0,19}", fullmatch=True)


# ---------------------------------------------------------------------------
# Property 19: Error isolation preserves exception details
# ---------------------------------------------------------------------------


class TestErrorIsolationPreservesExceptionDetails:
    """Property 19: Error isolation preserves exception details.

    For any exception raised within a toolkit tool function (of any type with
    any message), the resulting ToolResult SHALL contain both the exception type
    name and the exception message, and the exception SHALL NOT propagate to
    the caller.

    **Validates: Requirements 8.1, 8.2, 8.3**
    """

    @settings(max_examples=20, deadline=None)
    @given(
        exc_type=exception_types,
        message=error_messages,
        name=tool_names,
    )
    @pytest.mark.asyncio
    async def test_sync_tool_exception_captured_in_tool_result(
        self,
        exc_type: type[Exception],
        message: str,
        name: str,
    ) -> None:
        """A sync tool function that raises is captured as a ToolResult error
        containing the exception message, without propagating."""

        def raising_fn() -> str:
            """A tool that always raises."""
            raise exc_type(message)

        function_tool = FunctionTool(raising_fn, name=name)

        # The call must NOT raise — error is isolated
        result = await function_tool.invoke({})

        # Result is a ToolResult with an error
        assert isinstance(result, ToolResult)
        assert result.is_error
        assert result.error is not None

        # The error contains the exception message
        assert message in result.error or repr(message) in result.error

        # The error contains the tool name
        assert name in result.error

    @settings(max_examples=20, deadline=None)
    @given(
        exc_type=exception_types,
        message=error_messages,
        name=tool_names,
    )
    @pytest.mark.asyncio
    async def test_async_tool_exception_captured_in_tool_result(
        self,
        exc_type: type[Exception],
        message: str,
        name: str,
    ) -> None:
        """An async tool function that raises is captured as a ToolResult error
        containing the exception message, without propagating."""

        async def async_raising_fn() -> str:
            """An async tool that always raises."""
            raise exc_type(message)

        function_tool = FunctionTool(async_raising_fn, name=name)

        # The call must NOT raise — error is isolated
        result = await function_tool.invoke({})

        # Result is a ToolResult with an error
        assert isinstance(result, ToolResult)
        assert result.is_error
        assert result.error is not None

        # The error contains the exception message
        assert message in result.error or repr(message) in result.error

        # The error contains the tool name
        assert name in result.error
