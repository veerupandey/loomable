"""Claim-scoped memory — same user, different claim_id, no leakage.

Shows MemoryScope / Agent(scopes=) for insurance-style isolation.
"""

from __future__ import annotations

import asyncio

from loomable import (
    Agent,
    ConversationMemory,
    Memory,
    MemoryScope,
    UserMemory,
    open_session_store,
)
from loomable.agent import ModelSpec, NoteStore
from loomable.kernel.long_term import LongTermStore
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.memory import ScopedNoteStore


class _Script:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        blob = str(request.messages).lower()
        if "known facts" in blob and "neck" in blob:
            return ModelResponse(content="Claim notes: soft-tissue neck injury.")
        if "known facts" in blob and "bumper" in blob:
            return ModelResponse(content="Claim notes: rear bumper only.")
        return ModelResponse(content="ack")


class _Emb:
    async def embed(self, text: str) -> list[float]:
        return [float(hash(text) % 97) / 97.0, 1.0, 0.0]


async def main() -> None:
    base = NoteStore(long_term=LongTermStore(), embedder=_Emb())
    model = ModelSpec(provider="scripted", provider_impl=_Script())
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
    )
    print("CLM-1:", (await a1.arun("Summarize this claim")).output.text())

    a2 = Agent(
        model=model,
        memory=memory,
        session_id="claim:CLM-2",
        user_id="alice",
        scopes={"claim_id": "CLM-2"},
        modalities="text",
    )
    print("CLM-2:", (await a2.arun("Summarize this claim")).output.text())


if __name__ == "__main__":
    asyncio.run(main())
