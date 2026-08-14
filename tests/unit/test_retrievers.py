"""Unit tests for the RetrieverTool adapter.

Validates that a Retriever can be exposed as a Tool (MCP/API style) without
Kernel changes. Covers:
- Successful retrieval returns content as ToolResult (Req 16.1, 16.2)
- No Kernel modification needed to wire a retriever (Req 16.4)
- Failure yields an error naming the retriever (Req 16.5)
"""

from __future__ import annotations

from typing import Any

import pytest

from loomable.kernel.contracts import Retriever
from loomable.kernel.retrievers import RetrieverTool


# ---------------------------------------------------------------------------
# Test Retriever implementations
# ---------------------------------------------------------------------------


class FakeRetriever(Retriever):
    """A simple in-memory retriever for testing."""

    def __init__(self, name: str, docs: list[dict[str, Any]] | None = None) -> None:
        self.name = name
        self._docs = docs or []

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        # Return up to k docs that contain the query string
        matches = [d for d in self._docs if query.lower() in d.get("content", "").lower()]
        return matches[:k]


class FailingRetriever(Retriever):
    """A retriever that always raises an error."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        raise RuntimeError("Connection lost to vector store")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetrieverTool:
    """Tests for RetrieverTool adapter wiring retrievers as tools."""

    async def test_wraps_retriever_as_tool(self) -> None:
        """RetrieverTool exposes a Retriever with the Tool interface."""
        retriever = FakeRetriever("doc-search")
        tool = RetrieverTool(retriever)

        assert tool.name == "doc-search"
        assert "doc-search" in tool.description

    async def test_invoke_returns_retrieved_content(self) -> None:
        """Invoking the tool returns retrieved documents as ToolResult content."""
        docs = [
            {"content": "Python is great", "id": "1"},
            {"content": "Python async patterns", "id": "2"},
            {"content": "Java basics", "id": "3"},
        ]
        retriever = FakeRetriever("code-docs", docs)
        tool = RetrieverTool(retriever)

        result = await tool.invoke({"query": "Python", "k": 10})

        assert result.is_error is False
        assert len(result.content) == 2
        assert result.content[0]["id"] == "1"
        assert result.content[1]["id"] == "2"

    async def test_invoke_respects_k_parameter(self) -> None:
        """The k parameter limits the number of returned results."""
        docs = [
            {"content": "match a", "id": "1"},
            {"content": "match b", "id": "2"},
            {"content": "match c", "id": "3"},
        ]
        retriever = FakeRetriever("limited", docs)
        tool = RetrieverTool(retriever)

        result = await tool.invoke({"query": "match", "k": 2})

        assert result.is_error is False
        assert len(result.content) == 2

    async def test_invoke_defaults_k_to_5(self) -> None:
        """When k is not specified in args, defaults to 5."""
        docs = [{"content": f"item {i}", "id": str(i)} for i in range(10)]
        retriever = FakeRetriever("default-k", docs)
        tool = RetrieverTool(retriever)

        result = await tool.invoke({"query": "item"})

        assert result.is_error is False
        assert len(result.content) == 5
        assert result.metadata["k"] == 5

    async def test_invoke_metadata_includes_query_and_retriever(self) -> None:
        """Result metadata includes the retriever name, query, and k."""
        retriever = FakeRetriever("meta-test", [{"content": "hello"}])
        tool = RetrieverTool(retriever)

        result = await tool.invoke({"query": "hello", "k": 3})

        assert result.metadata["retriever_name"] == "meta-test"
        assert result.metadata["query"] == "hello"
        assert result.metadata["k"] == 3

    async def test_failure_yields_error_naming_retriever(self) -> None:
        """When retriever raises, ToolResult has error naming the retriever."""
        retriever = FailingRetriever("broken-retriever")
        tool = RetrieverTool(retriever)

        result = await tool.invoke({"query": "anything"})

        assert result.is_error is True
        assert "broken-retriever" in result.error
        assert result.metadata["retriever_name"] == "broken-retriever"

    async def test_custom_description(self) -> None:
        """RetrieverTool accepts a custom description override."""
        retriever = FakeRetriever("custom")
        tool = RetrieverTool(retriever, description="Search custom docs")

        assert tool.description == "Search custom docs"

    async def test_parameters_schema_advertises_query_and_k(self) -> None:
        """Agent models need a real JSON schema to call the tool correctly."""
        tool = RetrieverTool(FakeRetriever("search_kb"))
        assert tool.parameters["type"] == "object"
        assert "query" in tool.parameters["properties"]
        assert "k" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["query"]

    async def test_no_kernel_change_needed(self) -> None:
        """RetrieverTool extends Tool without any Kernel modification.

        This test validates Req 16.4: a new retriever can be wired in
        purely by instantiating RetrieverTool with any Retriever impl.
        """
        from loomable.kernel.contracts import Tool

        retriever = FakeRetriever("plug-and-play")
        tool = RetrieverTool(retriever)

        # The tool is a proper Tool subclass
        assert isinstance(tool, Tool)

        # It can be invoked through the standard interface
        result = await tool.invoke({"query": "test"})
        assert result.is_error is False

    async def test_empty_query_returns_empty_results(self) -> None:
        """An empty query that matches nothing returns an empty list."""
        retriever = FakeRetriever("empty", [{"content": "specific content"}])
        tool = RetrieverTool(retriever)

        result = await tool.invoke({"query": "nonexistent"})

        assert result.is_error is False
        assert result.content == []
