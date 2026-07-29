"""Retriever-to-Tool adapters for the loomable agent framework.

Exposes any Retriever as an invocable Tool (MCP-style or API-style) with no
Kernel source changes. The adapter wraps a Retriever instance and delegates
invoke() to the retriever's retrieve() method.

Error contract:
- If the retriever raises an exception, the adapter produces a ToolResult
  with an error naming the retriever (Req 16.5).
"""

from __future__ import annotations

from typing import Any

from loomable.kernel.contracts import Retriever, Tool
from loomable.kernel.models import ToolResult


class RetrieverTool(Tool):
    """Adapter that wraps a Retriever as an invocable Tool.

    This allows any Retriever to be exposed to an Agent as an MCP-style or
    API-style tool without modifying Kernel source code. The agent invokes
    the tool with a query (and optional k), and receives retrieved content.

    Args:
        retriever: The Retriever instance to wrap.
        description: Optional description override. Defaults to a generated
            description based on the retriever name.
    """

    def __init__(self, retriever: Retriever, description: str | None = None) -> None:
        self._retriever = retriever
        self.name: str = retriever.name
        self.description: str = description or f"Retriever tool: {retriever.name}"

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        """Invoke the wrapped retriever with query and k from args.

        Args:
            args: Must contain 'query' (str). May contain 'k' (int, default 5).

        Returns:
            ToolResult with retrieved content on success, or ToolResult with
            error naming the retriever on failure.
        """
        query: str = args.get("query", "")
        k: int = args.get("k", 5)

        try:
            results = await self._retriever.retrieve(query, k)
        except Exception as exc:
            return ToolResult(
                error=f"Retriever '{self._retriever.name}' failed: {exc}",
                metadata={"retriever_name": self._retriever.name},
            )

        return ToolResult(
            content=results,
            metadata={"retriever_name": self._retriever.name, "query": query, "k": k},
        )
