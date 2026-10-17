"""Recall a scoped user preference in a different conversation."""

import asyncio

from examples._provider import make_embedder, make_provider
from loomable import Agent, Memory, UserMemory, open_vector_store
from loomable.memory import NoteStore


async def main():
    notes = NoteStore(open_vector_store(engine="memory"), make_embedder())
    memory = Memory.compose(user=UserMemory(note_store=notes, auto_extract=True, memory_tool=False))
    async with make_provider() as model:
        first = Agent(model, memory=memory, user_id="alex", session_id="preferences")
        await first.arun("My favorite color is persimmon. Please acknowledge.")
        followup = Agent(model, memory=memory, user_id="alex", session_id="new-ticket")
        print((await followup.arun("What is my favorite color?")).text)


if __name__ == "__main__":
    asyncio.run(main())
