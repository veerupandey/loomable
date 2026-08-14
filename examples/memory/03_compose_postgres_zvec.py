"""Compose memory backends: Postgres (or in-memory) L1/L2 + vector L3.

Two independent axes — mix freely:

  Conversation (L1/L2)  → session_store / memory_backend
      sqlite | file | postgres | memory
  Long-term notes (L3)  → note_store (vector backend)
      Alibaba zvec (file) | PgVectorBackend | any VectorBackend
      (omit path → in-memory for zero-dep demos)

This demo:
  1. Creates a session (no resume=True — that requires an existing row)
  2. Persists chat turns into the conversation store
  3. Writes an episodic note into L3 (in-memory here; use path= for Alibaba zvec)
  4. New Agent(resume=True) reloads L1/L2; same note_store keeps L3

Requires for Postgres: pip install 'loomable[postgres]' && docker compose up -d
Requires for Alibaba zvec: pip install 'loomable[zvec]' and pass
  open_vector_store(path="./.loomable/notes_zvec", dimensions=…)
"""

from __future__ import annotations

import asyncio
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, NoteStore
from loomable.kernel.long_term import LongTermStore
from loomable.kernel.models import ModelRequest, ModelResponse
from loomable.memory import open_session_store


class _DemoProvider:
    """Scripted provider so the example runs without a live LLM key."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        blob = str(request.messages).lower()
        # Ask-about-prefs first (otherwise "preferences" matches "prefer")
        if "what are my" in blob or "my preferences" in blob:
            if "dark" in blob or "python" in blob:
                return ModelResponse(content="You prefer dark mode and Python.")
            return ModelResponse(content="I don't see preferences in history.")
        if "prefer" in blob or "dark" in blob:
            return ModelResponse(content="Got it — I'll remember your preferences.")
        return ModelResponse(content="ok")


class _FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        # Tiny deterministic vector — enough for in-process demos
        return [float(len(text) % 7), 1.0, 0.0, 0.0]


async def main() -> None:
    from loomable.agent import ModelSpec

    DSN = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    model = ModelSpec(provider="demo", provider_impl=_DemoProvider())
    embedder = _FakeEmbedder()
    session_id = f"compose-demo-{uuid.uuid4().hex[:8]}"

    if DSN:
        conversation = open_session_store("postgres", url=DSN, user_id="alice")
        print("L1/L2: Postgres")
    else:
        conversation = open_session_store("memory")
        print("L1/L2: in-memory (set POSTGRES_URL for Postgres)")

    # L3: in-memory by default. For Alibaba zvec on disk:
    #   open_vector_store(path="./.loomable/notes_zvec", dimensions=4)
    # For Postgres vectors:
    #   open_vector_store(postgres_url=DSN, dimensions=4, user_id="alice")
    notes = NoteStore(long_term=LongTermStore(), embedder=embedder)
    print("L3: in-memory VectorBackend (see open_vector_store for zvec/Postgres)")

    # First Agent: create session (do NOT pass resume=True).
    agent = Agent(
        model=model,
        session_id=session_id,
        session_store=conversation,
        note_store=notes,
        memory_tool=True,
        modalities="text",
        memory_window=8,
        user_id="alice",
    )
    print("turn1:", (await agent.arun("I prefer dark mode and Python.")).output.text())

    # Explicit L3 write (in a live LLM run the model would call the memory tool)
    await notes.write(
        "prefs",
        "User prefers dark mode and Python.",
        tags=["preferences"],
    )
    recalled = await notes.recall("preferences", k=1)
    print("L3 recall:", [n.text for n in recalled])

    # New Agent: same session_id + store + resume=True reloads L1/L2.
    # Reuse the same note_store object.
    agent2 = Agent(
        model=model,
        session_id=session_id,
        session_store=conversation,
        resume=True,
        note_store=notes,
        memory_tool=True,
        modalities="text",
        user_id="alice",
    )
    print("turn2:", (await agent2.arun("What are my preferences?")).output.text())


if __name__ == "__main__":
    asyncio.run(main())
