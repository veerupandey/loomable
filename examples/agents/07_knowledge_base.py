"""Searchable knowledge base — a vector store the agent queries as tools.

``knowledge_base=`` is a vector DB (optionally ingested from files/dirs).
It accepts a store / URI, sources to ingest, a ``KnowledgeBase``, a Corpus,
a retriever, or a name→collection mapping.

``retrievers=`` attaches extra ``search_*`` tools on the same Agent.
``create_deep_agent`` is Agent, so it takes the same kwargs.

This script is offline (scripted model). For a live model, swap in Gemini /
Azure OpenAI and keep the same ``knowledge_base=``.

Run::

    python examples/agents/07_knowledge_base.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loomable.agent import Agent, ModelSpec, create_deep_agent
from loomable.kernel.contracts import Retriever
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.providers.vector_store import open_vector_store
from loomable.retrieval import KnowledgeBase

ROOT = Path(__file__).resolve().parent / ".knowledge_base_demo"
ROOT.mkdir(parents=True, exist_ok=True)


def _seed() -> dict[str, list[Path]]:
    personal = ROOT / "personal"
    company = ROOT / "company"
    personal.mkdir(exist_ok=True)
    company.mkdir(exist_ok=True)
    (personal / "prefs.md").write_text(
        "# Avery preferences\n\n"
        "Never commit secrets. No API tokens in git, including .env files.\n",
        encoding="utf-8",
    )
    (company / "policy.md").write_text(
        "# Credential policy\n\n"
        "Staging credentials MAY be stored in a committed internal .env file.\n",
        encoding="utf-8",
    )
    (company / "runbook.md").write_text(
        "# Webhooks\n\n"
        "Current signing secret is DEMO-WH-4419. Rotate after any leak.\n",
        encoding="utf-8",
    )
    return {"personal": [personal], "company": [company]}


class _CatalogRetriever(Retriever):
    """Extra search tool shipped alongside the knowledge base."""

    name = "search_catalog"
    description = "Search the internal product catalog for SKUs."

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        rows = [
            {"content": "SKU-100 widget — $9", "id": "1"},
            {"content": "SKU-200 sprocket — $12", "id": "2"},
        ]
        q = (query or "").lower()
        hits = [r for r in rows if any(t in r["content"].lower() for t in q.split())]
        return (hits or rows)[: max(1, int(k))]


class _Scripted:
    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        if self.n == 1:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        tool_name="search_personal",
                        args={"query": "commit secrets tokens git", "k": 3},
                    )
                ],
            )
        if self.n == 2:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="2",
                        tool_name="search_company",
                        args={"query": "webhook DEMO-WH .env policy", "k": 5},
                    )
                ],
            )
        return ModelResponse(
            content=(
                "Do not commit the staging token (personal notes are stricter "
                "than company .env policy). Webhook key is DEMO-WH-4419 (runbook.md)."
            )
        )


def _tool_names(request: ModelRequest) -> set[str]:
    names: set[str] = set()
    for t in request.tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = (fn or {}).get("name") or t.get("name")
        if name:
            names.add(str(name))
    return names


class _KbObjectScripted:
    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        names = _tool_names(request)
        if self.n == 1:
            assert "search_handbook" in names, names
            assert "search_catalog" in names, names
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        tool_name="search_handbook",
                        args={"query": "OAuth2", "k": 2},
                    )
                ],
            )
        return ModelResponse(content="KnowledgeBase + retrievers demo complete.")


async def _ask(agent: Agent, label: str) -> None:
    result = await agent.arun(
        "Can I commit STAGING_TOKEN=demo-not-a-secret per policy? "
        "What is the webhook signing secret? Cite sources."
    )
    print(f"[{label}] {(result.output.text() or '').strip()}")


async def main() -> None:
    # --- Named collections → search_personal, search_company ---
    kb = _seed()
    agent = Agent(
        ModelSpec(provider="scripted", provider_impl=_Scripted()),
        user_id="avery",
        knowledge_base=kb,
        use_llm_summarizer=False,
        max_tool_iterations=8,
    )
    await _ask(agent, "Agent named collections")

    deep = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Scripted()),
        user_id="avery",
        knowledge_base=kb,
        workspace=ROOT / "workspace",
        web_search=False,
        url_fetch=False,
        citations=False,
        think_tool=False,
        board=False,
        use_llm_summarizer=False,
        max_tool_iterations=8,
    )
    await _ask(deep, "create_deep_agent")

    # --- KnowledgeBase(store=..., sources=...) + extra retrievers= ---
    handbook = ROOT / "handbook.md"
    handbook.write_text(
        "# Auth\n\nUse OAuth2 bearer tokens for API login.\n",
        encoding="utf-8",
    )
    store = open_vector_store(engine="memory")
    kb_obj = KnowledgeBase(
        store=store,
        sources=[handbook],
        name="handbook",
        description="Internal engineering handbook.",
    )
    combo = Agent(
        ModelSpec(provider="scripted", provider_impl=_KbObjectScripted()),
        knowledge_base=kb_obj,
        retrievers=[_CatalogRetriever()],
        use_llm_summarizer=False,
        max_tool_iterations=4,
    )
    result = await combo.arun("How do we authenticate?")
    print(f"[KnowledgeBase+retrievers] {(result.output.text() or '').strip()}")


if __name__ == "__main__":
    asyncio.run(main())
