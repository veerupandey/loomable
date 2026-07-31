"""Tests for loomable.flow.flow_class decorators and metadata (Task 8.1).

Validates:
- Req 6.1: FlowClass provides @start(), @listen(source), @router(source) decorators
- Decorators attach correct metadata to methods via _flow_meta attribute
- Metadata dataclasses contain the expected fields
"""

from __future__ import annotations

import pytest

from loomable.flow.flow_class import (
    _ListenMeta,
    _RouterMeta,
    _StartMeta,
    listen,
    router,
    start,
)


# ---------------------------------------------------------------------------
# @start() decorator
# ---------------------------------------------------------------------------


class TestStartDecorator:
    """The @start() decorator attaches _StartMeta to methods."""

    def test_attaches_start_meta(self):
        """@start() attaches a _StartMeta instance to the function."""

        @start()
        def my_method(self, input):
            return input

        assert hasattr(my_method, "_flow_meta")
        assert isinstance(my_method._flow_meta, _StartMeta)

    def test_preserves_function_identity(self):
        """@start() returns the original function unchanged."""

        def my_method(self, input):
            return input

        decorated = start()(my_method)
        assert decorated is my_method

    def test_function_remains_callable(self):
        """Decorated function is still callable."""

        @start()
        def my_method(input):
            return f"result:{input}"

        assert my_method("hello") == "result:hello"

    def test_async_function_supported(self):
        """@start() works with async functions."""

        @start()
        async def my_method(self, input):
            return input

        assert hasattr(my_method, "_flow_meta")
        assert isinstance(my_method._flow_meta, _StartMeta)


# ---------------------------------------------------------------------------
# @listen(source) decorator
# ---------------------------------------------------------------------------


class TestListenDecorator:
    """The @listen(source) decorator attaches _ListenMeta to methods."""

    def test_attaches_listen_meta(self):
        """@listen(source) attaches a _ListenMeta with the source name."""

        @listen("begin")
        def my_method(self, input):
            return input

        assert hasattr(my_method, "_flow_meta")
        assert isinstance(my_method._flow_meta, _ListenMeta)
        assert my_method._flow_meta.source == "begin"

    def test_preserves_function_identity(self):
        """@listen() returns the original function unchanged."""

        def my_method(self, input):
            return input

        decorated = listen("source_method")(my_method)
        assert decorated is my_method

    def test_source_stored_correctly(self):
        """The source parameter is stored in the metadata."""

        @listen("analyze_data")
        def process(self, input):
            return input

        assert process._flow_meta.source == "analyze_data"

    def test_function_remains_callable(self):
        """Decorated function is still callable."""

        @listen("source")
        def my_method(input):
            return f"processed:{input}"

        assert my_method("data") == "processed:data"

    def test_async_function_supported(self):
        """@listen() works with async functions."""

        @listen("start_method")
        async def my_method(self, input):
            return input

        assert hasattr(my_method, "_flow_meta")
        assert isinstance(my_method._flow_meta, _ListenMeta)
        assert my_method._flow_meta.source == "start_method"


# ---------------------------------------------------------------------------
# @router(source) decorator
# ---------------------------------------------------------------------------


class TestRouterDecorator:
    """The @router(source) decorator attaches _RouterMeta to methods."""

    def test_attaches_router_meta(self):
        """@router(source) attaches a _RouterMeta with the source name."""

        @router("analyze")
        def route_decision(self, input):
            return "next_step"

        assert hasattr(route_decision, "_flow_meta")
        assert isinstance(route_decision._flow_meta, _RouterMeta)
        assert route_decision._flow_meta.source == "analyze"

    def test_preserves_function_identity(self):
        """@router() returns the original function unchanged."""

        def route_decision(self, input):
            return "next"

        decorated = router("source_method")(route_decision)
        assert decorated is route_decision

    def test_source_stored_correctly(self):
        """The source parameter is stored in the metadata."""

        @router("process_step")
        def my_router(self, input):
            return "branch_a"

        assert my_router._flow_meta.source == "process_step"

    def test_function_remains_callable(self):
        """Decorated function is still callable."""

        @router("source")
        def my_router(input):
            return "route_a" if input else "route_b"

        assert my_router(True) == "route_a"
        assert my_router(False) == "route_b"

    def test_async_function_supported(self):
        """@router() works with async functions."""

        @router("analyze")
        async def route_decision(self, input):
            return "next_step"

        assert hasattr(route_decision, "_flow_meta")
        assert isinstance(route_decision._flow_meta, _RouterMeta)
        assert route_decision._flow_meta.source == "analyze"


# ---------------------------------------------------------------------------
# Metadata dataclass behavior
# ---------------------------------------------------------------------------


class TestMetadataDataclasses:
    """Metadata dataclasses have the correct structure."""

    def test_start_meta_is_dataclass(self):
        """_StartMeta can be instantiated with no arguments."""
        meta = _StartMeta()
        assert meta is not None

    def test_listen_meta_requires_source(self):
        """_ListenMeta requires a source string."""
        meta = _ListenMeta(source="my_source")
        assert meta.source == "my_source"

    def test_router_meta_requires_source(self):
        """_RouterMeta requires a source string."""
        meta = _RouterMeta(source="my_source")
        assert meta.source == "my_source"

    def test_listen_meta_equality(self):
        """Two _ListenMeta with same source are equal."""
        assert _ListenMeta(source="a") == _ListenMeta(source="a")
        assert _ListenMeta(source="a") != _ListenMeta(source="b")

    def test_router_meta_equality(self):
        """Two _RouterMeta with same source are equal."""
        assert _RouterMeta(source="x") == _RouterMeta(source="x")
        assert _RouterMeta(source="x") != _RouterMeta(source="y")

    def test_start_meta_equality(self):
        """Two _StartMeta instances are equal."""
        assert _StartMeta() == _StartMeta()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDecoratorEdgeCases:
    """Edge cases for decorator usage."""

    def test_multiple_decorators_last_wins(self):
        """If multiple flow decorators are applied, the last one wins."""
        # This is documented behavior — only one _flow_meta per method.
        @start()
        @listen("source")
        def my_method(self, input):
            return input

        # @start() is applied last (outermost), so it overwrites
        assert isinstance(my_method._flow_meta, _StartMeta)

    def test_listen_with_empty_source(self):
        """@listen with empty string still attaches metadata (validation is elsewhere)."""

        @listen("")
        def my_method(self, input):
            return input

        assert my_method._flow_meta.source == ""

    def test_router_with_empty_source(self):
        """@router with empty string still attaches metadata (validation is elsewhere)."""

        @router("")
        def my_method(self, input):
            return input

        assert my_method._flow_meta.source == ""
