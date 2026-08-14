"""Compose memory backends: conversation L1/L2 + vector L3 (live LLM).

Two independent axes — mix freely:

  Conversation (L1/L2)  → session_store
      sqlite | file | postgres | memory
  Long-term notes (L3)  → note_store (vector backend)
      default Alibaba zvec, or faiss / memory / postgres

Requires a live LLM key — see ``.env.example``.
Optional: ``POSTGRES_URL`` and ``pip install 'loomable[postgres,zvec]'``.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable import Agent
from loomable.agent import NoteStore
from loomable.kernel.long_term import LongTermStore
from loomable.memory import open_session_store
from loomable.providers import GeminiEmbedder
from loomable.providers.vector_store import open_vector_store


async def main() -> None:
    model = require_provider()
    embedder = GeminiEmbedder()
    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    session_id = f"compose-demo-{uuid.uuid4().hex[:8]}"

    if dsn:
        conversation = open_session_store("postgres", url=dsn, user_id="alice")
        print("L1/L2: Postgres")
    else:
        conversation = open_session_store("memory")
        print("L1/L2: in-memory (set POSTGRES_URL for Postgres)")

    try:
        notes = NoteStore(long_term=LongTermStore(), embedder=embedder)
        print("L3: Alibaba zvec (default)")
    except ImportError:
        notes = NoteStore(
            long_term=open_vector_store(engine="memory"),
            embedder=embedder,
        )
        print("L3: in-memory fallback (pip install loomable[zvec] for default)")

    agent = Agent(
        model=model,
        session_id=session_id,
        session_store=conversation,
        note_store=notes,
        memory_tool=True,
        modalities="text",
        memory_window=8,
        user_id="alice",
        instructions="Remember user preferences.",
    )
    print("turn1:", (await agent.arun("I prefer dark mode and Python.")).output.text())

    await notes.write(
        "prefs",
        "User prefers dark mode and Python.",
        tags=["preferences"],
    )
    recalled = await notes.recall("preferences", k=1)
    print("L3 recall:", [n.text for n in recalled])

    agent2 = Agent(
        model=model,
        session_id=session_id,
        session_store=conversation,
        resume=True,
        note_store=notes,
        memory_tool=True,
        modalities="text",
        user_id="alice",
        instructions="Remember user preferences.",
    )
    print("turn2:", (await agent2.arun("What are my preferences?")).output.text())
    notes._store.close()


if __name__ == "__main__":
    asyncio.run(main())
