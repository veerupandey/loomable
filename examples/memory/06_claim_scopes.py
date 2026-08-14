"""Claim-scoped memory — same user, different claim_id, no leakage (live LLM).

Shows MemoryScope / Agent(scopes=) for insurance-style isolation.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import make_embedder, require_provider  # noqa: E402

from loomable import (
    Agent,
    ConversationMemory,
    Memory,
    MemoryScope,
    UserMemory,
    open_session_store,
)
from loomable.agent import NoteStore
from loomable.kernel.long_term import LongTermStore
from loomable.memory import ScopedNoteStore


async def main() -> None:
    model = require_provider()
    base = NoteStore(long_term=LongTermStore(), embedder=make_embedder())
    store = open_session_store("memory")

    memory = Memory.compose(
        conversation=ConversationMemory(store=store),
        user=UserMemory(note_store=base, memory_tool=True, auto_extract=False),
    )

    # Seed two claims for the same insured
    s1 = ScopedNoteStore(base, scope=MemoryScope.of(user_id="alice", claim_id="CLM-1"))
    s2 = ScopedNoteStore(base, scope=MemoryScope.of(user_id="alice", claim_id="CLM-2"))
    await s1.write("injury", "soft-tissue neck injury")
    await s2.write("injury", "rear bumper only")

    a1 = Agent(
        model=model,
        memory=memory,
        session_id="claim:CLM-1",
        user_id="alice",
        scopes={"claim_id": "CLM-1"},
        modalities="text",
        instructions="Summarize only the facts for this claim. Do not invent other claims.",
    )
    print("CLM-1:", (await a1.arun("Summarize this claim's injury notes.")).output.text())

    a2 = Agent(
        model=model,
        memory=memory,
        session_id="claim:CLM-2",
        user_id="alice",
        scopes={"claim_id": "CLM-2"},
        modalities="text",
        instructions="Summarize only the facts for this claim. Do not invent other claims.",
    )
    print("CLM-2:", (await a2.arun("Summarize this claim's injury notes.")).output.text())


if __name__ == "__main__":
    asyncio.run(main())
