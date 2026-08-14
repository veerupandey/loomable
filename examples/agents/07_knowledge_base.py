"""Searchable knowledge base — live ``Agent(knowledge_base=...)``.

``knowledge_base=`` is a vector DB (optionally ingested from files/dirs).
``retrievers=`` attaches extra ``search_*`` tools on the same Agent.
``create_deep_agent`` takes the same kwargs.

Requires a real LLM key — see ``.env.example``.

Run::

    python examples/agents/07_knowledge_base.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, create_deep_agent, open_vector_store
from loomable.kernel.contracts import Retriever
from loomable.retrieval import KnowledgeBase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

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


class CatalogRetriever(Retriever):
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


async def main() -> None:
    model = require_provider()
    kb = _seed()
    question = (
        "Can I commit STAGING_TOKEN=demo-not-a-secret per policy? "
        "What is the webhook signing secret? Cite which collection you used."
    )

    agent = Agent(
        model,
        user_id="avery",
        knowledge_base=kb,
        instructions=(
            "Use search_personal and search_company. Prefer personal prefs over "
            "company policy when they conflict. Cite sources."
        ),
        max_tool_iterations=8,
    )
    result = await agent.arun(question)
    print("[Agent named collections]")
    print((result.output.text() or "").strip())

    deep = create_deep_agent(
        model,
        user_id="avery",
        knowledge_base=kb,
        workspace=ROOT / "workspace",
        web_search=False,
        url_fetch=False,
        citations=False,
        think_tool=False,
        board=False,
        max_tool_iterations=8,
    )
    deep_result = await deep.arun(question)
    print("\n[create_deep_agent]")
    print((deep_result.output.text() or "").strip())

    handbook = ROOT / "handbook.md"
    handbook.write_text(
        "# Auth\n\nUse OAuth2 bearer tokens for API login.\n",
        encoding="utf-8",
    )
    combo = Agent(
        model,
        knowledge_base=KnowledgeBase(
            store=open_vector_store(engine="memory"),
            sources=[handbook],
            name="handbook",
            description="Internal engineering handbook.",
        ),
        retrievers=[CatalogRetriever()],
        instructions="Use search_handbook and search_catalog when relevant.",
        max_tool_iterations=6,
    )
    combo_result = await combo.arun(
        "How do we authenticate to the API? What is the widget SKU?"
    )
    print("\n[KnowledgeBase + retrievers]")
    print((combo_result.output.text() or "").strip())


if __name__ == "__main__":
    asyncio.run(main())
