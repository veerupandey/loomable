# Feature: agent-ergonomics, Property 3
"""Property 3: Decorated function becomes an invocable tool.

For any decorated function, @tool SHALL produce a Tool whose name/description
reflect the function (or overrides) and whose invocation with valid args returns
a ToolResult derived from the function's return value, for both sync and async
functions.

**Validates: Requirements 2.1, 2.2, 2.4, 2.5**
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.agent.tools import FunctionTool, tool
from loomable.kernel.contracts import Tool
from loomable.kernel.models import ToolResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: valid Python identifier names for tool name overrides
# Python identifiers: start with letter/underscore, followed by alnum/underscore
valid_identifiers = st.from_regex(r"[a-z_][a-z0-9_]{0,29}", fullmatch=True)

# Strategy: non-empty description strings
descriptions = st.text(min_size=1, max_size=100)

# Strategy: simple return values the functions can produce
simple_return_values = st.one_of(
    st.text(max_size=50),
    st.integers(min_value=-(2**31), max_value=2**31),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
    st.lists(st.integers(min_value=-100, max_value=100), max_size=5),
    st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.integers(min_value=-100, max_value=100),
        max_size=5,
    ),
)

# Strategy: integer arguments for numeric tool functions
int_args = st.integers(min_value=-(2**31), max_value=2**31)

# Strategy: string arguments
str_args = st.text(min_size=0, max_size=50)


# ---------------------------------------------------------------------------
# Property tests: @tool produces a Tool with correct name/description
# ---------------------------------------------------------------------------


class TestToolNameDescriptionDefaults:
    """@tool produces a Tool whose name/description reflect the function."""

    @settings(max_examples=100)
    @given(func_name=valid_identifiers)
    def test_tool_name_defaults_to_function_name(self, func_name: str) -> None:
        """The tool name defaults to the function's __name__."""
        # Dynamically create a function with the given name
        exec_globals: dict[str, Any] = {}
        exec(
            f"def {func_name}(x: int) -> int:\n"
            f"    '''A docstring.'''\n"
            f"    return x\n",
            exec_globals,
        )
        fn = exec_globals[func_name]

        result = tool(fn)

        assert isinstance(result, Tool)
        assert isinstance(result, FunctionTool)
        assert result.name == func_name

    @settings(max_examples=100)
    @given(docstring=descriptions)
    def test_tool_description_defaults_to_docstring(self, docstring: str) -> None:
        """The tool description defaults to the function's docstring."""

        def sample_fn(x: int) -> int:
            return x

        sample_fn.__doc__ = docstring

        result = tool(sample_fn)

        assert isinstance(result, Tool)
        assert result.description == docstring.strip()


class TestToolNameDescriptionOverrides:
    """@tool uses override name/description when provided."""

    @settings(max_examples=100)
    @given(override_name=valid_identifiers, override_desc=descriptions)
    def test_overrides_take_precedence(
        self, override_name: str, override_desc: str
    ) -> None:
        """Override name and description take precedence over function defaults."""

        def original_fn(x: int) -> int:
            """Original docstring that should be overridden."""
            return x

        result = tool(original_fn, name=override_name, description=override_desc)

        assert isinstance(result, Tool)
        assert result.name == override_name
        assert result.description == override_desc

    @settings(max_examples=100)
    @given(override_name=valid_identifiers)
    def test_name_override_only(self, override_name: str) -> None:
        """When only name is overridden, description still comes from docstring."""

        def my_fn(x: int) -> int:
            """Keep this docstring."""
            return x

        result = tool(my_fn, name=override_name)

        assert result.name == override_name
        assert result.description == "Keep this docstring."

    @settings(max_examples=100)
    @given(override_desc=descriptions)
    def test_description_override_only(self, override_desc: str) -> None:
        """When only description is overridden, name still comes from fn.__name__."""

        def stable_name(x: int) -> int:
            """Overwritten."""
            return x

        result = tool(stable_name, description=override_desc)

        assert result.name == "stable_name"
        assert result.description == override_desc


# ---------------------------------------------------------------------------
# Property tests: Sync function invocation returns correct ToolResult
# ---------------------------------------------------------------------------


class TestSyncFunctionInvocation:
    """Sync function invocation returns a ToolResult derived from return value."""

    @settings(max_examples=100, deadline=None)
    @given(a=int_args, b=int_args)
    @pytest.mark.asyncio
    async def test_sync_add_returns_correct_result(self, a: int, b: int) -> None:
        """Sync function return value is wrapped in ToolResult.content."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        result = await add.invoke({"a": a, "b": b})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert result.content == a + b

    @settings(max_examples=100, deadline=None)
    @given(return_val=simple_return_values)
    @pytest.mark.asyncio
    async def test_sync_arbitrary_return_value(self, return_val: Any) -> None:
        """Any return value from a sync function is stored in ToolResult.content."""
        captured = {"val": return_val}

        @tool
        def return_anything() -> Any:
            """Return a captured value."""
            return captured["val"]

        result = await return_anything.invoke({})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert result.content == return_val

    @settings(max_examples=100, deadline=None)
    @given(name=str_args)
    @pytest.mark.asyncio
    async def test_sync_string_function(self, name: str) -> None:
        """Sync function with string arg returns correct string ToolResult."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        result = await greet.invoke({"name": name})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert result.content == f"Hello, {name}!"


# ---------------------------------------------------------------------------
# Property tests: Async function invocation returns correct ToolResult
# ---------------------------------------------------------------------------


class TestAsyncFunctionInvocation:
    """Async function invocation returns a ToolResult derived from return value."""

    @settings(max_examples=100)
    @given(a=int_args, b=int_args)
    @pytest.mark.asyncio
    async def test_async_add_returns_correct_result(self, a: int, b: int) -> None:
        """Async function return value is wrapped in ToolResult.content."""

        @tool
        async def async_add(a: int, b: int) -> int:
            """Add two integers asynchronously."""
            return a + b

        result = await async_add.invoke({"a": a, "b": b})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert result.content == a + b

    @settings(max_examples=100)
    @given(return_val=simple_return_values)
    @pytest.mark.asyncio
    async def test_async_arbitrary_return_value(self, return_val: Any) -> None:
        """Any return value from an async function is stored in ToolResult.content."""
        captured = {"val": return_val}

        @tool
        async def async_return_anything() -> Any:
            """Return a captured value asynchronously."""
            return captured["val"]

        result = await async_return_anything.invoke({})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert result.content == return_val

    @settings(max_examples=100)
    @given(name=str_args)
    @pytest.mark.asyncio
    async def test_async_string_function(self, name: str) -> None:
        """Async function with string arg returns correct string ToolResult."""

        @tool
        async def async_greet(name: str) -> str:
            """Greet someone asynchronously."""
            return f"Hi, {name}!"

        result = await async_greet.invoke({"name": name})

        assert isinstance(result, ToolResult)
        assert result.error is None
        assert result.content == f"Hi, {name}!"


# ---------------------------------------------------------------------------
# Property tests: Tool is a valid kernel Tool instance
# ---------------------------------------------------------------------------


class TestToolIsValidKernelTool:
    """@tool always produces an instance satisfying the kernel Tool contract."""

    @settings(max_examples=100)
    @given(
        use_override_name=st.booleans(),
        override_name=valid_identifiers,
        use_override_desc=st.booleans(),
        override_desc=descriptions,
        is_async=st.booleans(),
    )
    def test_decorated_function_is_tool_instance(
        self,
        use_override_name: bool,
        override_name: str,
        use_override_desc: bool,
        override_desc: str,
        is_async: bool,
    ) -> None:
        """Regardless of overrides or sync/async, result is always a Tool."""
        if is_async:
            async def fn(x: int) -> int:
                """Async function."""
                return x
        else:
            def fn(x: int) -> int:
                """Sync function."""
                return x

        kwargs: dict[str, Any] = {}
        if use_override_name:
            kwargs["name"] = override_name
        if use_override_desc:
            kwargs["description"] = override_desc

        result = tool(fn, **kwargs)

        # Must be a Tool instance
        assert isinstance(result, Tool)
        assert isinstance(result, FunctionTool)

        # Must have name and description as strings
        assert isinstance(result.name, str)
        assert len(result.name) > 0
        assert isinstance(result.description, str)

        # If overrides were provided, they should be used
        if use_override_name:
            assert result.name == override_name
        if use_override_desc:
            assert result.description == override_desc
