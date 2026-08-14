"""Unified Memory.compose — conversation + user long-term (live LLM).

    memory = Memory.compose(
        conversation=ConversationMemory(store=...),
        user=UserMemory(note_store=..., auto_extract=True),
    )
    Agent(model=..., memory=memory, session_id=..., user_id=...)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import make_embedder, require_provider  # noqa: E402

from loomable import Agent, ConversationMemory, Memory, UserMemory, open_session_store
from loomable.agent import NoteStore
from loomable.kernel.long_term import LongTermStore

ROOT = Path(__file__).resolve().parent / ".compose_sessions"
ROOT.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    model = require_provider()
    store = open_session_store("file", path=str(ROOT))
    notes = NoteStore(long_term=LongTermStore(), embedder=make_embedder())
    memory = Memory.compose(
        conversation=ConversationMemory(store=store, window=8),
        user=UserMemory(note_store=notes, memory_tool=True, auto_extract=True),
    )

    agent = Agent(
        model=model,
        memory=memory,
        session_id="compose-demo",
        user_id="alex",
        instructions="Remember the user's name and preferences.",
    )
    print((await agent.arun("Hi, my name is Alex and I prefer dark mode.")).output.text())
    print((await agent.arun("What do you know about me?")).output.text())


if __name__ == "__main__":
    asyncio.run(main())
