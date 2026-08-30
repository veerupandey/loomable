"""User / conversation memory via Memory.compose (preferred).

Requires for postgres: pip install 'loomable[postgres]' && docker compose up -d
Requires a live LLM key — see ``.env.example``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import make_embedder, require_provider  # noqa: E402

from loomable import Agent, ConversationMemory, Memory, UserMemory, open_session_store
from loomable.memory import NoteStore
from loomable.kernel.long_term import LongTermStore

DSN = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
ROOT = Path(__file__).resolve().parent / ".sessions_demo"
ROOT.mkdir(exist_ok=True)

if DSN:
    store = open_session_store("postgres", url=DSN, user_id="alice")
    label = "postgres"
else:
    store = open_session_store("file", path=str(ROOT))
    label = f"file:{ROOT}"


async def main() -> None:
    model = require_provider()
    print(f"Using conversation store: {label}")
    # L3 default = Alibaba zvec (.loomable/memory_zvec); pip install loomable[zvec]
    notes = NoteStore(long_term=LongTermStore(), embedder=make_embedder())
    memory = Memory.compose(
        conversation=ConversationMemory(store=store, window=8),
        user=UserMemory(note_store=notes, memory_tool=True, auto_extract=True),
    )

    a1 = Agent(
        model=model,
        memory=memory,
        session_id="demo-session",
        user_id="alice",
        instructions="Remember user preferences.",
    )
    print((await a1.arun("I prefer dark mode and Python.")).output.text())

    a2 = Agent(
        model=model,
        memory=memory,
        session_id="demo-session",
        user_id="alice",
        resume=True,
        instructions="Remember user preferences.",
    )
    print((await a2.arun("What are my preferences?")).output.text())


if __name__ == "__main__":
    asyncio.run(main())
