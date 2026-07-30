"""Tests for typed dependency injection end-to-end (Task 14).

Validates:
- Req 3.1: RunContext carries deps unchanged; tools/nodes that declare it receive it.
- Req 3.2: Tools/nodes that do NOT declare a deps parameter are invoked without it.
- Req 3.4: The same deps object is visible to every node in a Flow run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from loomable.agent.context import RunContext
from loomable.flow import Flow
from loomable.flow.runnable import FunctionRunnable


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeDeps:
    """A typed deps object used across tests."""

    db_url: str = "postgres://localhost/test"
    api_key: str = "secret-key-123"


# ---------------------------------------------------------------------------
# FunctionRunnable: deps injection based on signature
# ---------------------------------------------------------------------------


class TestFunctionRunnableDepsInjection:
    """FunctionRunnable injects context.deps when the function declares 'deps'."""

    @pytest.mark.asyncio
    async def test_deps_injected_when_declared(self):
        """A function declaring 'deps' receives context.deps."""
        received = {}

        def my_node(input, *, deps=None):
            received["deps"] = deps
            return f"got:{input}"

        runnable = FunctionRunnable(my_node)
        deps = FakeDeps()
        ctx = RunContext(deps=deps)

        result = await runnable.arun("hello", context=ctx)

        assert received["deps"] is deps
        assert result.output.parts[0].data == b"got:hello"

    @pytest.mark.asyncio
    async def test_deps_not_injected_when_not_declared(self):
        """A function without 'deps' parameter is called without it."""
        called_with_args = {}

        def simple_node(input):
            called_with_args["input"] = input
            return f"simple:{input}"

        runnable = FunctionRunnable(simple_node)
        deps = FakeDeps()
        ctx = RunContext(deps=deps)

        result = await runnable.arun("world", context=ctx)

        # Function was called with only 'input', no deps leaked in
        assert "deps" not in called_with_args
        assert called_with_args["input"] == "world"
        assert result.output.parts[0].data == b"simple:world"

    @pytest.mark.asyncio
    async def test_deps_is_none_when_context_is_none(self):
        """When no context is provided, deps is not injected; function uses its default."""
        received = {}

        def node_with_deps(input, *, deps=None):
            received["deps"] = deps
            return input

        runnable = FunctionRunnable(node_with_deps)

        # No context at all — deps kwarg not injected, function uses default=None
        await runnable.arun("test", context=None)

        # The function still ran with deps=None (its own default), not an injected value
        assert received["deps"] is None

    @pytest.mark.asyncio
    async def test_deps_none_when_context_deps_is_none(self):
        """When context.deps is None, deps=None is passed to functions that declare it."""
        received = {}

        def node_with_deps(input, *, deps=None):
            received["deps"] = deps
            return input

        runnable = FunctionRunnable(node_with_deps)
        ctx = RunContext(deps=None)

        await runnable.arun("test", context=ctx)

        assert received["deps"] is None

    @pytest.mark.asyncio
    async def test_async_function_receives_deps(self):
        """Async functions also receive deps when declared."""
        received = {}

        async def async_node(input, *, deps=None):
            received["deps"] = deps
            return f"async:{input}"

        runnable = FunctionRunnable(async_node)
        deps = FakeDeps(db_url="redis://localhost")
        ctx = RunContext(deps=deps)

        result = await runnable.arun("data", context=ctx)

        assert received["deps"] is deps
        assert received["deps"].db_url == "redis://localhost"
        assert result.output.parts[0].data == b"async:data"

    @pytest.mark.asyncio
    async def test_both_context_and_deps_injected(self):
        """A function declaring both 'context' and 'deps' receives both."""
        received = {}

        def full_node(input, *, context=None, deps=None):
            received["context"] = context
            received["deps"] = deps
            return f"full:{input}"

        runnable = FunctionRunnable(full_node)
        deps = FakeDeps()
        ctx = RunContext(deps=deps)

        await runnable.arun("x", context=ctx)

        assert received["context"] is ctx
        assert received["deps"] is deps

    @pytest.mark.asyncio
    async def test_deps_any_python_object(self):
        """Deps can be any Python object — dict, string, custom class, etc."""
        received_values = []

        def capture_deps(input, *, deps=None):
            received_values.append(deps)
            return input

        runnable = FunctionRunnable(capture_deps)

        # dict
        ctx1 = RunContext(deps={"key": "value"})
        await runnable.arun("a", context=ctx1)

        # string
        ctx2 = RunContext(deps="just-a-string")
        await runnable.arun("b", context=ctx2)

        # int
        ctx3 = RunContext(deps=42)
        await runnable.arun("c", context=ctx3)

        assert received_values[0] == {"key": "value"}
        assert received_values[1] == "just-a-string"
        assert received_values[2] == 42


# ---------------------------------------------------------------------------
# Flow end-to-end: same deps shared across all nodes
# ---------------------------------------------------------------------------


class TestFlowDepsSharing:
    """In a Flow run, all nodes that declare deps receive the same deps object."""

    @pytest.mark.asyncio
    async def test_same_deps_object_shared_across_all_nodes(self):
        """Every node in a flow receives the exact same deps instance (Req 3.4)."""
        seen_deps: list[Any] = []

        def node_a(input, *, deps=None):
            seen_deps.append(deps)
            return f"a:{input}"

        def node_b(input, *, deps=None):
            seen_deps.append(deps)
            return f"b:{input}"

        def node_c(input, *, deps=None):
            seen_deps.append(deps)
            return f"c:{input}"

        deps = FakeDeps(db_url="shared-db", api_key="shared-key")
        flow = Flow([node_a, node_b, node_c], deps=deps)

        await flow.arun("start")

        # All three nodes received deps
        assert len(seen_deps) == 3
        # All received the exact same object (identity, not just equality)
        assert seen_deps[0] is deps
        assert seen_deps[1] is deps
        assert seen_deps[2] is deps

    @pytest.mark.asyncio
    async def test_nodes_without_deps_unaffected_in_flow(self):
        """Nodes that don't declare deps still work normally in a flow."""
        call_log: list[str] = []

        def plain_node(input):
            call_log.append(f"plain:{input}")
            return f"plain:{input}"

        def deps_node(input, *, deps=None):
            call_log.append(f"deps:{input}:{deps}")
            return f"deps:{input}"

        deps = FakeDeps()
        flow = Flow([plain_node, deps_node], deps=deps)

        result = await flow.arun("hello")

        assert len(call_log) == 2
        assert call_log[0] == "plain:hello"
        # deps_node received the deps object
        assert f"deps:" in call_log[1]
        assert "FakeDeps" in call_log[1]

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async_nodes_receive_deps(self):
        """Both sync and async nodes in a flow receive the same deps."""
        seen_deps: list[Any] = []

        def sync_node(input, *, deps=None):
            seen_deps.append(("sync", deps))
            return f"sync:{input}"

        async def async_node(input, *, deps=None):
            seen_deps.append(("async", deps))
            return f"async:{input}"

        deps = FakeDeps(api_key="mixed-test")
        flow = Flow([sync_node, async_node], deps=deps)

        await flow.arun("data")

        assert len(seen_deps) == 2
        assert seen_deps[0][0] == "sync"
        assert seen_deps[0][1] is deps
        assert seen_deps[1][0] == "async"
        assert seen_deps[1][1] is deps

    @pytest.mark.asyncio
    async def test_deps_mutations_visible_downstream(self):
        """Because deps is shared by identity, mutations in one node are visible downstream."""
        shared_state_via_deps: dict[str, Any] = {"counter": 0}

        def incrementer(input, *, deps=None):
            deps["counter"] += 1
            return f"incremented:{deps['counter']}"

        def reader(input, *, deps=None):
            return f"counter={deps['counter']}"

        flow = Flow([incrementer, reader], deps=shared_state_via_deps)
        result = await flow.arun("go")

        # The reader should see the counter incremented by the first node
        assert shared_state_via_deps["counter"] == 1

    @pytest.mark.asyncio
    async def test_flow_without_deps_nodes_work_fine(self):
        """A flow with no deps configured still runs normally."""

        def add_prefix(input):
            return f"prefix:{input}"

        def add_suffix(input):
            return f"{input}:suffix"

        flow = Flow([add_prefix, add_suffix])

        result = await flow.arun("hello")
        # Should complete without errors
        assert result is not None

    @pytest.mark.asyncio
    async def test_deps_from_context_overrides_flow_deps(self):
        """If a context already has deps set, the flow does not override it."""
        seen_deps: list[Any] = []

        def capture(input, *, deps=None):
            seen_deps.append(deps)
            return input

        flow_deps = FakeDeps(db_url="flow-level")
        ctx_deps = FakeDeps(db_url="context-level")

        flow = Flow([capture], deps=flow_deps)
        ctx = RunContext(deps=ctx_deps)

        await flow.arun("test", context=ctx)

        # Context-level deps takes precedence (flow only sets if ctx.deps is None)
        assert seen_deps[0] is ctx_deps
        assert seen_deps[0].db_url == "context-level"
