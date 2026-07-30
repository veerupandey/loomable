# Feature: agent-ergonomics, Property 5
"""Property 5: Function tool errors are isolated and named.

For any decorated function that raises an exception, invoking the tool SHALL
return a ToolResult with an error identifying the tool name rather than
propagating the exception.

**Validates: Requirements 2.6**
"""

from __future__ import annotations

import pytest

from loomable.agent.tools import tool
from loomable.kernel.models import ToolResult


# ---------------------------------------------------------------------------
# Sync functions — various exception types
# ---------------------------------------------------------------------------


class TestSyncErrorIsolation:
    """Sync function exceptions are caught and returned as named ToolResult errors."""

    @pytest.mark.asyncio
    async def test_value_error_isolated(self):
        @tool
        def validate(x: int) -> int:
            """Validate input."""
            raise ValueError("x must be positive")

        result = await validate.invoke({"x": -1})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert result.error is not None
        assert "validate" in result.error
        assert "x must be positive" in result.error

    @pytest.mark.asyncio
    async def test_runtime_error_isolated(self):
        @tool
        def compute(n: int) -> int:
            """Compute something."""
            raise RuntimeError("computation overflow")

        result = await compute.invoke({"n": 999})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "compute" in result.error
        assert "computation overflow" in result.error

    @pytest.mark.asyncio
    async def test_key_error_isolated(self):
        @tool
        def lookup(key: str) -> str:
            """Lookup a key."""
            raise KeyError("missing_key")

        result = await lookup.invoke({"key": "missing_key"})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "lookup" in result.error

    @pytest.mark.asyncio
    async def test_type_error_isolated(self):
        @tool
        def concat(a: str, b: str) -> str:
            """Concatenate strings."""
            raise TypeError("unsupported operand types")

        result = await concat.invoke({"a": "hello", "b": "world"})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "concat" in result.error
        assert "unsupported operand types" in result.error

    @pytest.mark.asyncio
    async def test_generic_exception_isolated(self):
        @tool
        def risky() -> str:
            """Risky operation."""
            raise Exception("something went wrong")

        result = await risky.invoke({})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "risky" in result.error
        assert "something went wrong" in result.error

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self):
        """Invoking a failing tool must NOT raise — error is captured in result."""

        @tool
        def explode() -> None:
            """Explode."""
            raise RuntimeError("boom")

        # This should NOT raise
        result = await explode.invoke({})
        assert result.is_error


# ---------------------------------------------------------------------------
# Async functions — various exception types
# ---------------------------------------------------------------------------


class TestAsyncErrorIsolation:
    """Async function exceptions are caught and returned as named ToolResult errors."""

    @pytest.mark.asyncio
    async def test_value_error_isolated(self):
        @tool
        async def async_validate(x: int) -> int:
            """Validate input async."""
            raise ValueError("negative not allowed")

        result = await async_validate.invoke({"x": -5})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "async_validate" in result.error
        assert "negative not allowed" in result.error

    @pytest.mark.asyncio
    async def test_runtime_error_isolated(self):
        @tool
        async def async_compute(n: int) -> int:
            """Async compute."""
            raise RuntimeError("async overflow")

        result = await async_compute.invoke({"n": 42})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "async_compute" in result.error
        assert "async overflow" in result.error

    @pytest.mark.asyncio
    async def test_key_error_isolated(self):
        @tool
        async def async_lookup(key: str) -> str:
            """Async lookup."""
            raise KeyError("not_found")

        result = await async_lookup.invoke({"key": "abc"})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "async_lookup" in result.error

    @pytest.mark.asyncio
    async def test_type_error_isolated(self):
        @tool
        async def async_concat(a: str, b: str) -> str:
            """Async concat."""
            raise TypeError("bad types")

        result = await async_concat.invoke({"a": "x", "b": "y"})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "async_concat" in result.error
        assert "bad types" in result.error

    @pytest.mark.asyncio
    async def test_generic_exception_isolated(self):
        @tool
        async def async_risky() -> str:
            """Async risky."""
            raise Exception("async failure")

        result = await async_risky.invoke({})

        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "async_risky" in result.error
        assert "async failure" in result.error

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self):
        """Invoking a failing async tool must NOT raise."""

        @tool
        async def async_explode() -> None:
            """Async explode."""
            raise RuntimeError("kaboom")

        # This should NOT raise
        result = await async_explode.invoke({})
        assert result.is_error


# ---------------------------------------------------------------------------
# Named tool override — error message uses the configured name
# ---------------------------------------------------------------------------


class TestErrorUsesConfiguredName:
    """When a tool name is overridden, the error message uses the override."""

    @pytest.mark.asyncio
    async def test_sync_override_name_in_error(self):
        @tool(name="my_custom_tool")
        def original_name() -> str:
            """Will fail."""
            raise ValueError("failure")

        result = await original_name.invoke({})

        assert result.is_error
        assert "my_custom_tool" in result.error
        # The original function name should NOT appear
        assert "original_name" not in result.error

    @pytest.mark.asyncio
    async def test_async_override_name_in_error(self):
        @tool(name="async_custom")
        async def original_async() -> str:
            """Will fail async."""
            raise RuntimeError("async break")

        result = await original_async.invoke({})

        assert result.is_error
        assert "async_custom" in result.error
        assert "original_async" not in result.error
