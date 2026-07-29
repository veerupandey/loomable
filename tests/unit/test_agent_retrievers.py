"""Unit tests for high-level knowledge/retriever attachment (task 9.3, Req 16).

Verify that:
- A Retriever attached through the builder is exposed to the agent as an invocable
  tool in the default ToolRuntime, keyed by the retriever's name (Req 16.1/16.2).
- Invoking that tool with a query returns the retrieved content (Req 16.3).
- Retriever failures surface as an error naming the retriever, via the kernel
  RetrieverTool adapter unchanged (Req 16.4).
- Name collisions between an explicit tool and a retriever are surfaced eagerly.
- Explicit tools are still registered alongside retriever tools.
"""

from __future__ import annotations

from typing import Any

import pytest

from loomable.agent import Agent
from loomable.agent.errors import AgentConfigError
from loomable.kernel.contracts import Retriever, Tool
from loomable.kernel.models import (
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResult,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ModelProvider implementation (satisfies the structural protocol)."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


class _FakeRetriever(Retriever):
    """A Retriever returning a fixed set of docs, echoing the query back."""

    def __init__(self, name: str, docs: list[dict[str, Any]] | None = None) -> None:
        self.name = name
        self._docs = docs if docs is not None else [{"content": "doc-a"}, {"content": "doc-b"}]
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        self.calls.append((query, k))
        return self._docs[:k]


class _BrokenRetriever(Retriever):
    """A Retriever that always raises to exercise the error contract."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        raise RuntimeError("boom")


class _EchoTool(Tool):
    """A trivial explicit tool used to test collisions/coexistence."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"echo tool {name}"

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(content={"echo": args})


# ---------------------------------------------------------------------------
# Attachment + invocation (Req 16.1, 16.2, 16.3)
# ---------------------------------------------------------------------------


class TestRetrieverAttachment:
    def test_retriever_registered_as_tool_by_name(self):
        retriever = _FakeRetriever("kb")
        built = Agent(model=_FakeProvider(), retrievers=[retriever]).build()

        # Exposed in the default ToolRuntime keyed by the retriever's name.
        assert "kb" in built.tool_runtime._tools
        from loomable.kernel.retrievers import RetrieverTool

        assert isinstance(built.tool_runtime._tools["kb"], RetrieverTool)

    async def test_invoking_retriever_tool_returns_retrieved_content(self):
        retriever = _FakeRetriever("kb", docs=[{"content": "hello"}, {"content": "world"}])
        built = Agent(model=_FakeProvider(), retrievers=[retriever]).build()

        call = ToolCall(id="c1", tool_name="kb", args={"query": "greet", "k": 2})
        [outcome] = await built.tool_runtime.dispatch([call])

        assert outcome.call_id == "c1"
        assert outcome.error is None
        assert outcome.result is not None
        assert outcome.result.content == [{"content": "hello"}, {"content": "world"}]
        # The retriever actually received the query/k.
        assert retriever.calls == [("greet", 2)]

    async def test_multiple_retrievers_each_invocable(self):
        r1 = _FakeRetriever("docs", docs=[{"content": "d1"}])
        r2 = _FakeRetriever("faqs", docs=[{"content": "f1"}])
        built = Agent(model=_FakeProvider(), retrievers=[r1, r2]).build()

        calls = [
            ToolCall(id="a", tool_name="docs", args={"query": "q"}),
            ToolCall(id="b", tool_name="faqs", args={"query": "q"}),
        ]
        outcomes = {o.call_id: o for o in await built.tool_runtime.dispatch(calls)}

        assert outcomes["a"].result.content == [{"content": "d1"}]
        assert outcomes["b"].result.content == [{"content": "f1"}]

    async def test_retrievers_coexist_with_explicit_tools(self):
        built = Agent(
            model=_FakeProvider(),
            tools=[_EchoTool("echo")],
            retrievers=[_FakeRetriever("kb")],
        ).build()

        assert "echo" in built.tool_runtime._tools
        assert "kb" in built.tool_runtime._tools


# ---------------------------------------------------------------------------
# Error contract via the kernel RetrieverTool adapter (Req 16.4)
# ---------------------------------------------------------------------------


class TestRetrieverFailure:
    async def test_retriever_failure_names_the_retriever(self):
        built = Agent(
            model=_FakeProvider(), retrievers=[_BrokenRetriever("flaky")]
        ).build()

        call = ToolCall(id="c1", tool_name="flaky", args={"query": "q"})
        [outcome] = await built.tool_runtime.dispatch([call])

        # The RetrieverTool adapter returns a ToolResult carrying an error (not an
        # exception), so the outcome is a successful dispatch whose result is an error.
        assert outcome.result is not None
        assert outcome.result.is_error
        assert "flaky" in outcome.result.error
        assert outcome.result.metadata.get("retriever_name") == "flaky"


# ---------------------------------------------------------------------------
# Collision handling
# ---------------------------------------------------------------------------


class TestRetrieverCollisions:
    def test_name_collision_with_explicit_tool_raises(self):
        agent = Agent(
            model=_FakeProvider(),
            tools=[_EchoTool("kb")],
            retrievers=[_FakeRetriever("kb")],
        )
        with pytest.raises(AgentConfigError) as exc:
            agent.build()
        assert "kb" in exc.value.field

    def test_duplicate_retriever_names_raise(self):
        agent = Agent(
            model=_FakeProvider(),
            retrievers=[_FakeRetriever("kb"), _FakeRetriever("kb")],
        )
        with pytest.raises(AgentConfigError) as exc:
            agent.build()
        assert "kb" in exc.value.field
