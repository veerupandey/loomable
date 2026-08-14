"""Personalized agent + knowledge bases (personal vs company).

``create_personalized_agent`` ingest user notes and org docs as named
``search_*`` retrievers, then builds a normal :class:`Agent` or deep agent.
Discovery never defers those search tools.
"""

from __future__ import annotations

from typing import Any, Sequence

from loomable.agent.builder import Agent
from loomable.kernel.contracts import Retriever
from loomable.providers.vector_store import open_vector_store
from loomable.retrieval import AgenticRetriever, ingest

__all__ = ["create_personalized_agent", "build_knowledge_retriever"]


async def build_knowledge_retriever(
    sources: Sequence[Any],
    *,
    name: str,
    description: str = "",
    user_id: str | None = None,
    scope: str = "knowledge",
    embedder: Any | None = None,
    store: Any | None = None,
    metadata: dict[str, Any] | None = None,
    base_mode: str = "hybrid",
    strategy: str = "auto",
) -> Retriever:
    """Ingest *sources* into an agentic search tool named ``name`` (search_*)."""
    from loomable.retrieval.naming import ensure_search_tool_name

    tool_name = ensure_search_tool_name(name)
    meta = dict(metadata or {})
    if user_id:
        meta.setdefault("user_id", user_id)
    meta.setdefault("scope", scope)
    corpus = await ingest(
        sources,
        name=scope,
        description=description,
        store=store or open_vector_store(engine="memory"),
        embedder=embedder,
        strategy=strategy,
        base_mode=base_mode,
        metadata=meta,
    )
    return AgenticRetriever(
        corpus,
        name=tool_name,
        description=description
        or f"Search the '{scope}' knowledge base and cite filename/page.",
        mode="auto",
        rewrite="off",
        rerank="mmr",
    )


async def create_personalized_agent(
    model: Any,
    *,
    user_id: str,
    personal: Sequence[Any],
    knowledge: Sequence[Any] | None = None,
    deep: bool = False,
    embedder: Any | None = None,
    personal_store: Any | None = None,
    knowledge_store: Any | None = None,
    name: str | None = None,
    instructions: str | None = None,
    **agent_kwargs: Any,
) -> Agent:
    """Build an agent with a personal KB (+ optional company KB).

    ::

        agent = await create_personalized_agent(
            model,
            user_id="avery",
            personal=["./notes", "I never commit secrets."],
            knowledge=["./handbook.pdf", "./runbooks"],
            deep=True,
        )
        await agent.arun("Can I commit the staging token? Cite sources.")
    """
    retrievers: list[Retriever] = []
    personal_r = await build_knowledge_retriever(
        personal,
        name="search_personal",
        description=(
            f"Search {user_id}'s personal notes, preferences, and private facts. "
            "Call this for user-specific constraints (diet, timezone, secret handling)."
        ),
        user_id=user_id,
        scope="personal",
        embedder=embedder,
        store=personal_store,
    )
    retrievers.append(personal_r)

    if knowledge:
        company_r = await build_knowledge_retriever(
            knowledge,
            name="search_company",
            description=(
                "Search company policy, runbooks, and product docs. "
                "Cite filename and page. If personal notes conflict, flag the conflict."
            ),
            user_id=user_id,
            scope="company",
            embedder=embedder,
            store=knowledge_store,
        )
        retrievers.append(company_r)

    prompt = (
        f"You are a personalized assistant for user '{user_id}'.\n"
        "Always search knowledge tools before answering factual questions.\n"
        "Cite source filename (and page when present).\n"
        "If personal preferences conflict with company policy, state both, "
        "then follow the stricter personal safety/security constraint.\n"
        "Never invent identifiers (SKU, keys, tokens) that tools did not return."
    )
    if instructions:
        prompt = f"{prompt}\n\n{instructions.strip()}"

    agent_name = name or f"assistant-{user_id}"
    require = list(agent_kwargs.pop("require_tools", None) or [])
    require.extend(r.name for r in retrievers)

    common: dict[str, Any] = dict(
        model=model,
        name=agent_name,
        role="Personalized assistant",
        goal=f"Help {user_id} using their notes and the company knowledge base",
        instructions=prompt,
        retrievers=retrievers,
        user_id=user_id,
        require_tools=require or None,
        **agent_kwargs,
    )
    common = {k: v for k, v in common.items() if v is not None}

    if deep:
        from loomable.agent.deep import create_deep_agent

        common.setdefault("web_search", False)
        common.setdefault("url_fetch", False)
        common.setdefault("discovery", True)
        common.setdefault("use_llm_summarizer", False)
        return create_deep_agent(**common)

    common.setdefault("use_llm_summarizer", False)
    return Agent(**common)
