"""Retriever-to-Tool adapters for the loomable agent framework.

Expose any :class:`~loomable.kernel.contracts.Retriever` as an invocable Tool
with a proper JSON schema so the agent LLM can call it. Registration happens
via ``Agent(retrievers=[...])`` — the tool name is ``retriever.name``.

Error contract:
- If the retriever raises, the adapter returns a ToolResult error naming the
  retriever (Req 16.5).
"""

from __future__ import annotations

from typing import Any

from loomable.kernel.contracts import Retriever, Tool
from loomable.kernel.models import ToolResult

_RETRIEVER_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Natural-language search query to look up in this knowledge base.",
        },
        "k": {
            "type": "integer",
            "description": "Maximum number of results to return (default 5).",
            "default": 5,
        },
    },
    "required": ["query"],
}


class RetrieverTool(Tool):
    """Adapter that wraps any Retriever as an agent tool.

    The model sees ``name``, ``description``, and ``parameters`` (query + k).
    ``invoke({"query": ..., "k": ...})`` delegates to ``retriever.retrieve``.

    Args:
        retriever: Any object with ``.name`` and ``async retrieve(query, k)``.
        description: Optional override; otherwise uses ``retriever.description``
            or a generated search hint.
    """

    def __init__(self, retriever: Retriever, description: str | None = None) -> None:
        self._retriever = retriever
        self.name: str = retriever.name
        self.parameters: dict[str, Any] = dict(_RETRIEVER_PARAMETERS)
        self.description: str = description or _default_description(retriever)

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        """Invoke the wrapped retriever with query and k from args."""
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


def _default_description(retriever: Retriever) -> str:
    custom = (getattr(retriever, "description", None) or "").strip()
    if custom:
        return custom
    return (
        f"Search knowledge base '{retriever.name}'. "
        "Call this tool to retrieve documents, facts, or context needed to answer the user."
    )
