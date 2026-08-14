"""User / conversation memory via Memory.compose (preferred) or session_store.

Requires for postgres: pip install 'loomable[postgres]' && docker compose up -d
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, ConversationMemory, Memory, UserMemory, open_session_store
from loomable.agent import NoteStore
from loomable.kernel.long_term import LongTermStore
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()
DSN = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
ROOT = Path(__file__).resolve().parent / ".sessions_demo"
ROOT.mkdir(exist_ok=True)

if DSN:
    store = open_session_store("postgres", url=DSN, user_id="alice")
    label = "postgres"
else:
    store = open_session_store("file", path=str(ROOT))
    label = f"file:{ROOT}"


class _FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.0]


async def main() -> None:
    print(f"Using conversation store: {label}")
    # Optional long-term layer (skip if you only need chat history)
    notes = NoteStore(long_term=LongTermStore(), embedder=_FakeEmbedder())
    memory = Memory.compose(
        conversation=ConversationMemory(store=store, window=8),
        user=UserMemory(note_store=notes, memory_tool=True, auto_extract=True),
    )

    a1 = Agent(
        model=provider,
        memory=memory,
        session_id="demo-session",
        user_id="alice",
        # Extra isolation keys when needed, e.g. insurance:
        # scopes={"claim_id": "CLM-4421"},
        instructions="Remember user preferences.",
    )
    print((await a1.arun("I prefer dark mode and Python.")).output.text())

    a2 = Agent(
        model=provider,
        memory=memory,
        session_id="demo-session",
        user_id="alice",
        resume=True,
        instructions="Remember user preferences.",
    )
    print((await a2.arun("What are my preferences?")).output.text())


asyncio.run(main())
