"""Agent-facing retriever tool names.

Any :class:`~loomable.kernel.contracts.Retriever` passed to
``Agent(retrievers=[...])`` is registered as a tool under ``retriever.name``.
Names should read as search actions for the model (``search_docs``,
``search_auth``, …). Corpus identity (for multi-corpus routing) stays separate.
"""

from __future__ import annotations

__all__ = ["ensure_search_tool_name", "DEFAULT_SEARCH_DOCS", "DEFAULT_SEARCH_KNOWLEDGE"]

DEFAULT_SEARCH_DOCS = "search_docs"
DEFAULT_SEARCH_KNOWLEDGE = "search_knowledge"

_VERB_PREFIXES = ("search_", "retrieve_", "lookup_", "find_", "query_")


def ensure_search_tool_name(name: str | None, *, default: str = DEFAULT_SEARCH_DOCS) -> str:
    """Normalize a retriever name into an agent tool name.

    - ``\"docs\"`` → ``\"search_docs\"``
    - ``\"search_auth\"`` / ``\"retrieve_kb\"`` / ``\"knowledge_search\"`` kept as-is
    - empty / None → ``default``
    """
    raw = (name or "").strip()
    if not raw:
        return default
    lower = raw.lower()
    if lower.startswith(_VERB_PREFIXES):
        return raw
    if lower.endswith(("_search", "_retrieve", "_lookup", "_find", "_query")):
        return raw
    if lower == "retrieve":
        return DEFAULT_SEARCH_DOCS
    return f"search_{raw}"
