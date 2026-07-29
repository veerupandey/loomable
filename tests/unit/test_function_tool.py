"""Unit tests for loomable.agent.tools — @tool decorator and FunctionTool."""

from __future__ import annotations

import asyncio

import pytest

from loomable.agent.tools import FunctionTool, tool
from loomable.kernel.contracts import Tool
from loomable.kernel.models import ToolResult


# ---------------------------------------------------------------------------
# Basic decorator behavior
# ---------------------------------------------------------------------------


class TestToolDecorator:
    """Test that @tool produces a valid FunctionTool."""

    def test_decorator_without_args(self):
        @tool
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}!"

        assert isinstance(greet, FunctionTool)
        assert isinstance(greet, Tool)

    def test_decorator_with_args(self):
        @tool(name="greeter", description="Custom greeting tool")
        def greet(name: str) -> str:
            """Original docstring."""
            return f"Hello, {name}!"

        assert greet.name == "greeter"
        assert greet.description == "Custom greeting tool"

    def test_name_defaults_to_function_name(self):
        @tool
        def my_func():
            pass

        assert my_func.name == "my_func"

    def test_description_defaults_to_docstring(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        assert add.description == "Add two numbers together."

    def test_description_empty_when_no_docstring(self):
        @tool
        def no_docs(x: int) -> int:
            return x

        assert no_docs.description == ""


# ---------------------------------------------------------------------------
# Schema derivation
# ---------------------------------------------------------------------------


class TestSchemaDerived:
    """Test that the JSON schema is correctly derived from the function signature."""

    def test_basic_types(self):
        @tool
        def fn(a: str, b: int, c: float, d: bool) -> str:
            """Test."""
            return ""

        schema = fn.parameters
        assert schema["type"] == "object"
        assert schema["properties"]["a"] == {"type": "string"}
        assert schema["properties"]["b"] == {"type": "integer"}
        assert schema["properties"]["c"] == {"type": "number"}
        assert schema["properties"]["d"] == {"type": "boolean"}

    def test_list_and_dict_types(self):
        @tool
        def fn(items: list, data: dict) -> str:
            """Test."""
            return ""

        schema = fn.parameters
        assert schema["properties"]["items"] == {"type": "array"}
        assert schema["properties"]["data"] == {"type": "object"}

    def test_unannotated_defaults_to_string(self):
        @tool
        def fn(x):
            """Test."""
            return x

        schema = fn.parameters
        assert schema["properties"]["x"] == {"type": "string"}

    def test_required_params_without_defaults(self):
        @tool
        def fn(required_a: str, required_b: int, optional_c: str = "hi") -> str:
            """Test."""
            return ""

        schema = fn.parameters
        assert set(schema["required"]) == {"required_a", "required_b"}

    def test_no_required_when_all_have_defaults(self):
        @tool
        def fn(a: str = "x", b: int = 0) -> str:
            """Test."""
            return ""

        schema = fn.parameters
        assert "required" not in schema

    def test_var_args_excluded(self):
        @tool
        def fn(a: str, *args, **kwargs) -> str:
            """Test."""
            return ""

        schema = fn.parameters
        assert "args" not in schema["properties"]
        assert "kwargs" not in schema["properties"]
        assert "a" in schema["properties"]


# ---------------------------------------------------------------------------
# Invocation
# ---------------------------------------------------------------------------


class TestInvocation:
    """Test invoke() for sync and async functions."""

    @pytest.mark.asyncio
    async def test_sync_function_invocation(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        result = await add.invoke({"a": 3, "b": 5})
        assert isinstance(result, ToolResult)
        assert result.content == 8
        assert result.error is None

    @pytest.mark.asyncio
    async def test_async_function_invocation(self):
        @tool
        async def async_add(a: int, b: int) -> int:
            """Async add."""
            return a + b

        result = await async_add.invoke({"a": 10, "b": 20})
        assert isinstance(result, ToolResult)
        assert result.content == 30
        assert result.error is None

    @pytest.mark.asyncio
    async def test_sync_function_runs_in_thread(self):
        """Sync functions should run via asyncio.to_thread (non-blocking)."""
        import threading

        main_thread = threading.current_thread()
        invocation_thread = None

        @tool
        def check_thread() -> str:
            """Check thread."""
            nonlocal invocation_thread
            invocation_thread = threading.current_thread()
            return "done"

        await check_thread.invoke({})
        assert invocation_thread is not None
        assert invocation_thread != main_thread


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    """Test that exceptions are caught and returned as ToolResult errors."""

    @pytest.mark.asyncio
    async def test_sync_exception_isolated(self):
        @tool
        def failing_tool(x: int) -> int:
            """Fails."""
            raise ValueError("bad input")

        result = await failing_tool.invoke({"x": 1})
        assert result.is_error
        assert "failing_tool" in result.error
        assert "bad input" in result.error

    @pytest.mark.asyncio
    async def test_async_exception_isolated(self):
        @tool
        async def async_fail(x: int) -> int:
            """Fails async."""
            raise RuntimeError("network error")

        result = await async_fail.invoke({"x": 1})
        assert result.is_error
        assert "async_fail" in result.error
        assert "network error" in result.error

    @pytest.mark.asyncio
    async def test_error_names_overridden_tool(self):
        @tool(name="custom_name")
        def broken() -> str:
            """Broken."""
            raise Exception("oops")

        result = await broken.invoke({})
        assert result.is_error
        assert "custom_name" in result.error


# ---------------------------------------------------------------------------
# schema() helper
# ---------------------------------------------------------------------------


class TestSchemaMethod:
    """Test the schema() helper returns OpenAI-style schema."""

    def test_schema_format(self):
        @tool
        def lookup(id: str) -> dict:
            """Look up a record by id."""
            return {}

        schema = lookup.schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "lookup"
        assert schema["function"]["description"] == "Look up a record by id."
        assert schema["function"]["parameters"]["properties"]["id"] == {"type": "string"}


# ---------------------------------------------------------------------------
# Idempotent flag
# ---------------------------------------------------------------------------


class TestIdempotentFlag:
    """Test that the idempotent flag is correctly handled."""

    def test_default_idempotent_true(self):
        @tool
        def simple(x: int) -> int:
            """Simple tool."""
            return x

        assert simple.idempotent is True

    def test_decorator_with_idempotent_false(self):
        @tool(idempotent=False)
        def send_email(to: str, body: str) -> str:
            """Send an email."""
            return "sent"

        assert send_email.idempotent is False

    def test_decorator_with_idempotent_true_explicit(self):
        @tool(idempotent=True)
        def fetch(url: str) -> str:
            """Fetch a URL."""
            return ""

        assert fetch.idempotent is True

    def test_function_tool_direct_idempotent_false(self):
        def my_func(x: int) -> int:
            """A function."""
            return x

        ft = FunctionTool(my_func, idempotent=False)
        assert ft.idempotent is False

    def test_function_tool_direct_default_idempotent(self):
        def my_func(x: int) -> int:
            """A function."""
            return x

        ft = FunctionTool(my_func)
        assert ft.idempotent is True

    def test_idempotent_with_other_decorator_args(self):
        @tool(name="mailer", description="Send mail", idempotent=False)
        def send(to: str) -> str:
            """Send."""
            return "ok"

        assert send.name == "mailer"
        assert send.description == "Send mail"
        assert send.idempotent is False
