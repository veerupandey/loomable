"""Unified Memory.compose — conversation + user long-term in one object.

Demonstrates the productized API::

    memory = Memory.compose(
        conversation=ConversationMemory(store=...),
        user=UserMemory(note_store=..., auto_extract=True),
    )
    Agent(model=..., memory=memory, session_id=..., user_id=...)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, ConversationMemory, Memory, UserMemory, open_session_store
from loomable.agent import ModelSpec, NoteStore
from loomable.kernel.long_term import LongTermStore
from loomable.kernel.models import ModelRequest, ModelResponse

ROOT = Path(__file__).resolve().parent / ".compose_sessions"
ROOT.mkdir(parents=True, exist_ok=True)


class _Script:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        blob = str(request.messages).lower()
        if "known facts" in blob and "alex" in blob:
            return ModelResponse(content="I remember you — Alex, prefers dark mode.")
        if "prefer" in blob or "name is" in blob:
            return ModelResponse(content="Got it, I'll remember that.")
        if "remember" in blob or "know about" in blob:
            if "alex" in blob or "dark" in blob:
                return ModelResponse(content="You are Alex and you prefer dark mode.")
            return ModelResponse(content="I don't have facts yet.")
        return ModelResponse(content="ok")


class _Emb:
    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7), 1.0, 0.0, 0.0]


async def main() -> None:
    DSN = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if DSN:
        conv_store = open_session_store("postgres", url=DSN, user_id="alice")
        print("conversation: postgres")
    else:
        conv_store = open_session_store("file", path=str(ROOT))
        print(f"conversation: file:{ROOT}")

    notes = NoteStore(long_term=LongTermStore(), embedder=_Emb())
    memory = Memory.compose(
        conversation=ConversationMemory(store=conv_store, window=8),
        user=UserMemory(note_store=notes, memory_tool=True, auto_extract=True),
    )

    model = ModelSpec(provider="scripted", provider_impl=_Script())
    a1 = Agent(
        model=model,
        memory=memory,
        session_id="compose-demo",
        user_id="alice",
        modalities="text",
    )
    print((await a1.arun("My name is Alex. I prefer dark mode.")).output.text())

    # New session id — conversation is fresh, but user facts survive via UserMemory
    a2 = Agent(
        model=model,
        memory=memory,
        session_id="compose-demo-2",
        user_id="alice",
        modalities="text",
    )
    print((await a2.arun("What do you know about me?")).output.text())


if __name__ == "__main__":
    asyncio.run(main())
