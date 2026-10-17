"""Ingest a small handbook and have an agent search it before answering."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from examples._provider import make_embedder, make_provider
from loomable import Agent, open_vector_store
from loomable.retrieval import KnowledgeBase


async def main():
    with TemporaryDirectory() as directory:
        handbook = Path(directory) / "handbook.md"
        handbook.write_text("# Returns\nUnused items can be returned within 45 days.\n", encoding="utf-8")
        knowledge = KnowledgeBase(
            store=open_vector_store(engine="memory"), sources=[handbook], name="handbook",
            description="Search the shop policy handbook.",
        )
        async with make_provider() as model:
            agent = Agent(
                model, knowledge_base=knowledge, embedder=make_embedder(),
                require_tools=["search_handbook"], strict_require_tools=True,
                instructions="Answer from the handbook and name the source.",
            )
            print((await agent.arun("How long do I have to return an unused item?")).text)


if __name__ == "__main__":
    asyncio.run(main())
