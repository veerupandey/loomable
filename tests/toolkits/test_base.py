# Feature: built-in-toolkits, Property 1: Toolkit tools() returns all registered FunctionTool instances
# Feature: built-in-toolkits, Property 2: Agent flattens toolkits into individual tools
# Feature: built-in-toolkits, Property 3: include_tools filters to specified subset
# Feature: built-in-toolkits, Property 4: exclude_tools removes specified subset
"""Property tests for the Toolkit base class.

Property 1: For any Toolkit subclass that registers N tool functions, calling
tools() with no include/exclude filters SHALL return exactly N FunctionTool
instances, each with a unique name matching the registered function.

Property 2: For any list of Tool and Toolkit instances passed to Agent(tools=[...]),
the resulting ToolRuntime registry SHALL contain every individual FunctionTool from
each Toolkit's tools() output plus every directly-passed Tool, keyed by name.

Property 3: For any Toolkit with N registered tools and any subset S of those tool
names passed as include_tools, tools() SHALL return exactly the tools whose names
are in S, and no others.

Property 4: For any Toolkit with N registered tools and any subset S of those tool
names passed as exclude_tools, tools() SHALL return exactly the tools whose names
are NOT in S.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4**
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.agent import Agent
from loomable.agent.errors import AgentConfigError
from loomable.agent.tools import FunctionTool
from loomable.kernel.contracts import Tool
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.toolkits._base import Toolkit


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ModelProvider satisfying the structural protocol."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for valid Python identifier names (used as tool names)
valid_tool_names = st.from_regex(r"[a-z][a-z0-9_]{0,19}", fullmatch=True)


@st.composite
def unique_tool_names(draw: st.DrawFn, min_size: int = 1, max_size: int = 8) -> list[str]:
    """Generate a list of unique tool names."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    names: list[str] = []
    for _ in range(n):
        name = draw(valid_tool_names.filter(lambda x, ns=names: x not in ns))
        names.append(name)
    return names


def _make_function_tool(name: str) -> FunctionTool:
    """Create a FunctionTool with a given name."""

    def _fn(x: int = 0) -> int:
        return x

    _fn.__name__ = name
    _fn.__doc__ = f"Tool {name}"
    return FunctionTool(_fn, name=name)


class _DynamicToolkit(Toolkit):
    """A toolkit that holds a pre-built list of FunctionTool instances."""

    def __init__(self, tools_list: list[FunctionTool]) -> None:
        super().__init__()
        self._tools_list = tools_list

    def _register_tools(self) -> list[FunctionTool]:
        return self._tools_list


# ---------------------------------------------------------------------------
# Property test: Agent flattens toolkits into individual tools
# ---------------------------------------------------------------------------


class TestAgentFlattensToolkits:
    """Property 2: Agent flattens toolkits into individual tools."""

    @settings(max_examples=20)
    @given(data=st.data())
    def test_all_tools_from_toolkits_and_standalone_present_in_registry(
        self, data: st.DataObject
    ) -> None:
        """For any mix of standalone FunctionTool and Toolkit instances in
        Agent(tools=[...]), the registry contains every tool keyed by name."""

        # Generate between 0..4 standalone tool names
        num_standalone = data.draw(
            st.integers(min_value=0, max_value=4), label="num_standalone"
        )
        all_names_so_far: list[str] = []
        standalone_names: list[str] = []
        for _ in range(num_standalone):
            name = data.draw(
                valid_tool_names.filter(lambda x, ns=all_names_so_far: x not in ns),
                label="standalone_name",
            )
            standalone_names.append(name)
            all_names_so_far.append(name)

        # Generate between 1..3 toolkits, each with 1..4 tools
        num_toolkits = data.draw(
            st.integers(min_value=1, max_value=3), label="num_toolkits"
        )
        toolkit_tool_names: list[list[str]] = []
        for tk_idx in range(num_toolkits):
            num_tools_in_tk = data.draw(
                st.integers(min_value=1, max_value=4),
                label=f"toolkit_{tk_idx}_size",
            )
            tk_names: list[str] = []
            for _ in range(num_tools_in_tk):
                name = data.draw(
                    valid_tool_names.filter(
                        lambda x, ns=all_names_so_far: x not in ns
                    ),
                    label=f"toolkit_{tk_idx}_tool_name",
                )
                tk_names.append(name)
                all_names_so_far.append(name)
            toolkit_tool_names.append(tk_names)

        # Build FunctionTool instances and Toolkit instances
        standalone_tools = [_make_function_tool(n) for n in standalone_names]
        toolkits = [
            _DynamicToolkit([_make_function_tool(n) for n in tk_names])
            for tk_names in toolkit_tool_names
        ]

        # Shuffle the tools= list to interleave standalone and toolkit items
        items: list[Any] = list(standalone_tools) + list(toolkits)
        shuffled = data.draw(st.permutations(items), label="tools_order")

        # Create Agent and call _build_tool_registry directly
        agent = Agent(model=_FakeProvider(), tools=list(shuffled))
        registry, _errors = agent._build_tool_registry()

        # Verify: every standalone tool name is in the registry
        for name in standalone_names:
            assert name in registry, (
                f"Standalone tool '{name}' missing from registry"
            )
            assert isinstance(registry[name], FunctionTool)

        # Verify: every toolkit tool name is in the registry
        for tk_names in toolkit_tool_names:
            for name in tk_names:
                assert name in registry, (
                    f"Toolkit tool '{name}' missing from registry"
                )
                assert isinstance(registry[name], FunctionTool)

        # Verify: no extra tools beyond what we provided
        expected_names = set(all_names_so_far)
        assert set(registry.keys()) == expected_names, (
            f"Registry keys mismatch. Expected {expected_names}, got {set(registry.keys())}"
        )

# ---------------------------------------------------------------------------
# Property 1: Toolkit tools() returns all registered FunctionTool instances
# ---------------------------------------------------------------------------


class TestToolkitToolsReturnsAllRegistered:
    """Property 1: Toolkit tools() returns all registered FunctionTool instances.

    For any Toolkit subclass that registers N tool functions, calling tools()
    with no include/exclude filters SHALL return exactly N FunctionTool instances,
    each with a unique name matching the registered function.

    **Validates: Requirements 1.1**
    """

    @settings(max_examples=20)
    @given(data=st.data())
    def test_tools_returns_all_registered_tools(self, data: st.DataObject) -> None:
        """tools() with no filters returns all registered FunctionTool instances."""
        # Generate a list of unique tool names (1..10)
        names = data.draw(unique_tool_names(min_size=1, max_size=10), label="tool_names")

        # Build a toolkit that registers those tools
        toolkit = _DynamicToolkit([_make_function_tool(n) for n in names])

        result = toolkit.tools()

        # Exactly N FunctionTool instances
        assert len(result) == len(names)

        # All are FunctionTool instances
        for t in result:
            assert isinstance(t, FunctionTool)

        # Names match exactly (order-independent)
        result_names = {t.name for t in result}
        assert result_names == set(names)

    @settings(max_examples=20)
    @given(data=st.data())
    def test_each_tool_has_unique_name(self, data: st.DataObject) -> None:
        """All FunctionTool instances returned by tools() have unique names."""
        names = data.draw(unique_tool_names(min_size=1, max_size=10), label="tool_names")

        toolkit = _DynamicToolkit([_make_function_tool(n) for n in names])

        result = toolkit.tools()

        result_names = [t.name for t in result]
        # No duplicates
        assert len(result_names) == len(set(result_names))


# ---------------------------------------------------------------------------
# Property 3: include_tools filters to specified subset
# ---------------------------------------------------------------------------


class TestIncludeToolsFilters:
    """Property 3: include_tools filters to specified subset.

    For any Toolkit with N registered tools and any subset S of those tool names
    passed as include_tools, tools() SHALL return exactly the tools whose names
    are in S, and no others.

    **Validates: Requirements 1.3**
    """

    @settings(max_examples=20)
    @given(data=st.data())
    def test_include_tools_returns_only_specified_subset(
        self, data: st.DataObject
    ) -> None:
        """tools() with include_tools returns only the named tools."""
        # Generate all tool names for the toolkit
        all_names = data.draw(
            unique_tool_names(min_size=1, max_size=10), label="all_names"
        )

        # Draw a subset of those names to include (can be empty or full)
        include_subset = data.draw(
            st.lists(
                st.sampled_from(all_names),
                min_size=0,
                max_size=len(all_names),
                unique=True,
            ),
            label="include_subset",
        )

        # Build toolkit with include_tools filter
        tools_list = [_make_function_tool(n) for n in all_names]
        toolkit = _DynamicToolkit.__new__(_DynamicToolkit)
        toolkit._tools_list = tools_list
        Toolkit.__init__(toolkit, include_tools=include_subset)

        result = toolkit.tools()

        # Result should contain exactly the included tools
        result_names = {t.name for t in result}
        assert result_names == set(include_subset)

        # All returned tools are FunctionTool instances
        for t in result:
            assert isinstance(t, FunctionTool)

    @settings(max_examples=20)
    @given(data=st.data())
    def test_include_tools_empty_list_returns_nothing(
        self, data: st.DataObject
    ) -> None:
        """include_tools=[] returns an empty list."""
        all_names = data.draw(
            unique_tool_names(min_size=1, max_size=10), label="all_names"
        )

        tools_list = [_make_function_tool(n) for n in all_names]
        toolkit = _DynamicToolkit.__new__(_DynamicToolkit)
        toolkit._tools_list = tools_list
        Toolkit.__init__(toolkit, include_tools=[])

        result = toolkit.tools()
        assert result == []


# ---------------------------------------------------------------------------
# Property 4: exclude_tools removes specified subset
# ---------------------------------------------------------------------------


class TestExcludeToolsFilters:
    """Property 4: exclude_tools removes specified subset.

    For any Toolkit with N registered tools and any subset S of those tool names
    passed as exclude_tools, tools() SHALL return exactly the tools whose names
    are NOT in S.

    **Validates: Requirements 1.4**
    """

    @settings(max_examples=20)
    @given(data=st.data())
    def test_exclude_tools_removes_specified_subset(
        self, data: st.DataObject
    ) -> None:
        """tools() with exclude_tools returns all tools NOT in the excluded set."""
        # Generate all tool names for the toolkit
        all_names = data.draw(
            unique_tool_names(min_size=1, max_size=10), label="all_names"
        )

        # Draw a subset of those names to exclude (can be empty or full)
        exclude_subset = data.draw(
            st.lists(
                st.sampled_from(all_names),
                min_size=0,
                max_size=len(all_names),
                unique=True,
            ),
            label="exclude_subset",
        )

        # Build toolkit with exclude_tools filter
        tools_list = [_make_function_tool(n) for n in all_names]
        toolkit = _DynamicToolkit.__new__(_DynamicToolkit)
        toolkit._tools_list = tools_list
        Toolkit.__init__(toolkit, exclude_tools=exclude_subset)

        result = toolkit.tools()

        # Result should contain exactly the tools NOT excluded
        expected_names = set(all_names) - set(exclude_subset)
        result_names = {t.name for t in result}
        assert result_names == expected_names

        # All returned tools are FunctionTool instances
        for t in result:
            assert isinstance(t, FunctionTool)

    @settings(max_examples=20)
    @given(data=st.data())
    def test_exclude_tools_empty_list_returns_all(
        self, data: st.DataObject
    ) -> None:
        """exclude_tools=[] returns all tools (nothing excluded)."""
        all_names = data.draw(
            unique_tool_names(min_size=1, max_size=10), label="all_names"
        )

        tools_list = [_make_function_tool(n) for n in all_names]
        toolkit = _DynamicToolkit.__new__(_DynamicToolkit)
        toolkit._tools_list = tools_list
        Toolkit.__init__(toolkit, exclude_tools=[])

        result = toolkit.tools()

        result_names = {t.name for t in result}
        assert result_names == set(all_names)


# ---------------------------------------------------------------------------
# Additional: both include_tools and exclude_tools raises AgentConfigError
# ---------------------------------------------------------------------------


class TestIncludeExcludeMutualExclusion:
    """Both include_tools and exclude_tools raises AgentConfigError.

    **Validates: Requirements 1.5**
    """

    @settings(max_examples=20)
    @given(data=st.data())
    def test_both_include_and_exclude_raises_config_error(
        self, data: st.DataObject
    ) -> None:
        """Providing both include_tools and exclude_tools raises AgentConfigError."""
        all_names = data.draw(
            unique_tool_names(min_size=2, max_size=10), label="all_names"
        )

        # Split names into two non-empty groups for include and exclude
        split_point = data.draw(
            st.integers(min_value=1, max_value=len(all_names) - 1),
            label="split_point",
        )
        include_set = all_names[:split_point]
        exclude_set = all_names[split_point:]

        with pytest.raises(AgentConfigError):
            _DynamicToolkit.__new__(_DynamicToolkit)
            # Use Toolkit.__init__ directly to trigger the validation
            tk = _DynamicToolkit.__new__(_DynamicToolkit)
            Toolkit.__init__(
                tk,
                include_tools=include_set,
                exclude_tools=exclude_set,
            )
