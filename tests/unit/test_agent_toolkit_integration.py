"""Integration tests for Agent with Toolkit instances in the tools= list.

Verifies that the Agent builder correctly handles Toolkit instances:
1. Flattens a single Toolkit into individual FunctionTool instances
2. Works with a mix of Tool and Toolkit in tools= list
3. Respects include_tools filtering end-to-end through the agent
4. Respects exclude_tools filtering end-to-end through the agent
5. Handles multiple toolkits together

**Validates: Requirements 1.2, 1.3, 1.4**
"""

from __future__ import annotations

import pytest

from loomable.agent import Agent
from loomable.agent.tools import FunctionTool
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.toolkits.file_tools import FileTools
from loomable.toolkits.sql_tools import SQLTools


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ModelProvider satisfying the structural protocol."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentWithSingleToolkit:
    """Test Agent with a single Toolkit flattens into individual tools."""

    def test_file_tools_flattened_into_registry(self, tmp_path):
        """FileTools passed to Agent(tools=[toolkit]) produces individual tools."""
        toolkit = FileTools(base_dir=str(tmp_path))
        agent = Agent(model=_FakeProvider(), tools=[toolkit])
        registry, errors = agent._build_tool_registry()

        assert not errors
        # FileTools registers: read_file, write_file, list_directory
        assert "read_file" in registry
        assert "write_file" in registry
        assert "list_directory" in registry
        assert len(registry) == 3

        # Each entry is a FunctionTool
        for tool in registry.values():
            assert isinstance(tool, FunctionTool)

    def test_sql_tools_flattened_into_registry(self):
        """SQLTools passed to Agent(tools=[toolkit]) produces individual tools."""
        toolkit = SQLTools()
        agent = Agent(model=_FakeProvider(), tools=[toolkit])
        registry, errors = agent._build_tool_registry()

        assert not errors
        # SQLTools registers: run_sql, list_tables, describe_table
        assert "run_sql" in registry
        assert "list_tables" in registry
        assert "describe_table" in registry
        assert len(registry) == 3

        for tool in registry.values():
            assert isinstance(tool, FunctionTool)


class TestAgentWithMixedToolsAndToolkits:
    """Test Agent with a mix of standalone FunctionTool and Toolkit."""

    def test_standalone_tool_plus_toolkit(self, tmp_path):
        """Both a standalone FunctionTool and a Toolkit's tools are registered."""

        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}"

        standalone = FunctionTool(greet, name="greet")
        toolkit = FileTools(base_dir=str(tmp_path))

        agent = Agent(model=_FakeProvider(), tools=[standalone, toolkit])
        registry, errors = agent._build_tool_registry()

        assert not errors
        # Standalone tool
        assert "greet" in registry
        assert isinstance(registry["greet"], FunctionTool)

        # Toolkit tools
        assert "read_file" in registry
        assert "write_file" in registry
        assert "list_directory" in registry

        # Total: 1 standalone + 3 from FileTools
        assert len(registry) == 4

    def test_toolkit_before_standalone_tool(self, tmp_path):
        """Order doesn't matter: toolkit first, standalone second."""

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        standalone = FunctionTool(add, name="add")
        toolkit = SQLTools()

        agent = Agent(model=_FakeProvider(), tools=[toolkit, standalone])
        registry, errors = agent._build_tool_registry()

        assert not errors
        assert "add" in registry
        assert "run_sql" in registry
        assert "list_tables" in registry
        assert "describe_table" in registry
        assert len(registry) == 4


class TestAgentWithIncludeToolsFiltering:
    """Test include_tools filtering end-to-end through the agent."""

    def test_include_only_read_file(self, tmp_path):
        """Only 'read_file' is registered when include_tools limits the toolkit."""
        toolkit = FileTools(
            base_dir=str(tmp_path),
            include_tools=["read_file"],
        )
        agent = Agent(model=_FakeProvider(), tools=[toolkit])
        registry, errors = agent._build_tool_registry()

        assert not errors
        assert "read_file" in registry
        assert "write_file" not in registry
        assert "list_directory" not in registry
        assert len(registry) == 1

    def test_include_multiple_tools(self, tmp_path):
        """Multiple tools can be included from a toolkit."""
        toolkit = FileTools(
            base_dir=str(tmp_path),
            include_tools=["read_file", "list_directory"],
        )
        agent = Agent(model=_FakeProvider(), tools=[toolkit])
        registry, errors = agent._build_tool_registry()

        assert not errors
        assert "read_file" in registry
        assert "list_directory" in registry
        assert "write_file" not in registry
        assert len(registry) == 2


class TestAgentWithExcludeToolsFiltering:
    """Test exclude_tools filtering end-to-end through the agent."""

    def test_exclude_write_file(self, tmp_path):
        """'write_file' is excluded; others remain."""
        toolkit = FileTools(
            base_dir=str(tmp_path),
            exclude_tools=["write_file"],
        )
        agent = Agent(model=_FakeProvider(), tools=[toolkit])
        registry, errors = agent._build_tool_registry()

        assert not errors
        assert "read_file" in registry
        assert "list_directory" in registry
        assert "write_file" not in registry
        assert len(registry) == 2

    def test_exclude_multiple_tools(self):
        """Multiple tools can be excluded from a toolkit."""
        toolkit = SQLTools(exclude_tools=["run_sql", "describe_table"])
        agent = Agent(model=_FakeProvider(), tools=[toolkit])
        registry, errors = agent._build_tool_registry()

        assert not errors
        assert "list_tables" in registry
        assert "run_sql" not in registry
        assert "describe_table" not in registry
        assert len(registry) == 1


class TestAgentWithMultipleToolkits:
    """Test Agent with multiple toolkits combined."""

    def test_file_tools_and_sql_tools_combined(self, tmp_path):
        """All tools from FileTools and SQLTools are registered together."""
        file_toolkit = FileTools(base_dir=str(tmp_path))
        sql_toolkit = SQLTools()

        agent = Agent(model=_FakeProvider(), tools=[file_toolkit, sql_toolkit])
        registry, errors = agent._build_tool_registry()

        assert not errors
        # FileTools: read_file, write_file, list_directory
        assert "read_file" in registry
        assert "write_file" in registry
        assert "list_directory" in registry
        # SQLTools: run_sql, list_tables, describe_table
        assert "run_sql" in registry
        assert "list_tables" in registry
        assert "describe_table" in registry

        assert len(registry) == 6

    def test_multiple_toolkits_with_filtering(self, tmp_path):
        """Toolkits with different filters combine correctly."""
        file_toolkit = FileTools(
            base_dir=str(tmp_path),
            include_tools=["read_file"],
        )
        sql_toolkit = SQLTools(exclude_tools=["describe_table"])

        agent = Agent(model=_FakeProvider(), tools=[file_toolkit, sql_toolkit])
        registry, errors = agent._build_tool_registry()

        assert not errors
        # FileTools with include_tools=["read_file"]
        assert "read_file" in registry
        assert "write_file" not in registry
        assert "list_directory" not in registry
        # SQLTools with exclude_tools=["describe_table"]
        assert "run_sql" in registry
        assert "list_tables" in registry
        assert "describe_table" not in registry

        assert len(registry) == 3
