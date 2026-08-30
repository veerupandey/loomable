"""Compose memory backends via Memory.compose (live LLM).

Conversation (L1/L2) and long-term notes (L3) are independent layers::

    memory = Memory.compose(
        conversation=ConversationMemory(store=...),
        user=UserMemory(note_store=..., memory_tool=True),
    )
    Agent(model=..., memory=memory, session_id=..., user_id=...)

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
from _provider import make_embedder, require_provider  # noqa: E402

from loomable import Agent, ConversationMemory, Memory, UserMemory, open_session_store
from loomable.memory import MemoryScope, NoteStore, ScopedNoteStore
from loomable.kernel.long_term import LongTermStore
from loomable.providers.vector_store import open_vector_store


async def main() -> None:
    model = require_provider()
    embedder = make_embedder()
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

    # Seed through the same scope the Agent will use (user_id=alice).
    scoped_notes = ScopedNoteStore(notes, scope=MemoryScope.of(user_id="alice"))
    memory = Memory.compose(
        conversation=ConversationMemory(store=conversation, window=8),
        user=UserMemory(note_store=notes, memory_tool=True),
    )

    agent = Agent(
        model=model,
        memory=memory,
        session_id=session_id,
        user_id="alice",
        modalities="text",
        instructions="Remember user preferences.",
    )
    print("turn1:", (await agent.arun("I prefer dark mode and Python.")).output.text())

    await scoped_notes.write(
        "prefs",
        "User prefers dark mode and Python.",
        tags=["preferences"],
    )
    recalled = await scoped_notes.recall("preferences", k=1)
    print("L3 recall:", [n.text for n in recalled])

    agent2 = Agent(
        model=model,
        memory=memory,
        session_id=session_id,
        resume=True,
        user_id="alice",
        modalities="text",
        instructions="Remember user preferences.",
    )
    print("turn2:", (await agent2.arun("What are my preferences?")).output.text())
    notes._store.close()


if __name__ == "__main__":
    asyncio.run(main())
