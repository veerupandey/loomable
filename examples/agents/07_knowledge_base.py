"""Searchable knowledge base — high-level ``knowledge_base=`` / ``retrievers=``.

``knowledge_base=`` is a vector DB (optionally ingested from files/dirs).
``retrievers=`` attaches extra ``search_*`` tools on the same Agent.
``create_deep_agent`` is Agent, so it takes the same kwargs.

Offline (scripted model). For live use, swap in Gemini / Azure and keep the
same ``knowledge_base=`` kwargs.

Run::

    python examples/agents/07_knowledge_base.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from loomable import Agent, create_deep_agent, open_vector_store
from loomable.agent import ModelSpec
from loomable.kernel.contracts import Retriever
from loomable.retrieval import KnowledgeBase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _offline import scripted_model  # noqa: E402

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


async def _ask(agent: Agent, label: str) -> None:
    result = await agent.arun(
        "Can I commit STAGING_TOKEN=demo-not-a-secret per policy? "
        "What is the webhook signing secret? Cite sources."
    )
    print(f"[{label}] {(result.output.text() or '').strip()}")


def _kb_script() -> ModelSpec:
    return scripted_model(
        [
            {"tool": "search_personal", "args": {"query": "commit secrets tokens git", "k": 3}},
            {"tool": "search_company", "args": {"query": "webhook DEMO-WH .env policy", "k": 5}},
            "Do not commit the staging token (personal notes are stricter "
            "than company .env policy). Webhook key is DEMO-WH-4419 (runbook.md).",
        ]
    )


async def main() -> None:
    kb = _seed()

    # Named collections → search_personal, search_company
    await _ask(
        Agent(
            _kb_script(),
            user_id="avery",
            knowledge_base=kb,
            use_llm_summarizer=False,
            max_tool_iterations=8,
        ),
        "Agent named collections",
    )

    await _ask(
        create_deep_agent(
            _kb_script(),
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
        ),
        "create_deep_agent",
    )

    # KnowledgeBase(store=..., sources=...) + extra retrievers=
    handbook = ROOT / "handbook.md"
    handbook.write_text("# Auth\n\nUse OAuth2 bearer tokens for API login.\n", encoding="utf-8")
    combo = Agent(
        scripted_model(
            [
                {"tool": "search_handbook", "args": {"query": "OAuth2", "k": 2}},
                "KnowledgeBase + retrievers demo complete.",
            ]
        ),
        knowledge_base=KnowledgeBase(
            store=open_vector_store(engine="memory"),
            sources=[handbook],
            name="handbook",
            description="Internal engineering handbook.",
        ),
        retrievers=[CatalogRetriever()],
        use_llm_summarizer=False,
        max_tool_iterations=4,
    )
    result = await combo.arun("How do we authenticate?")
    print(f"[KnowledgeBase+retrievers] {(result.output.text() or '').strip()}")


if __name__ == "__main__":
    asyncio.run(main())
