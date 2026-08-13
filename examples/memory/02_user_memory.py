"""Postgres KV + vector memory demo.

Requires: pip install 'loomable[postgres]' && docker compose up -d
Env: POSTGRES_URL=postgresql://loomable:loomable@127.0.0.1:5432/loomable
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.kernel.long_term import LongTermStore
from loomable.kernel.stores import ShortTermStore
from loomable.providers.openai import AzureOpenAIProvider

DSN = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
provider = AzureOpenAIProvider()

if DSN:
    from loomable.providers.backends.postgres import PgVectorBackend, PostgresMemoryBackend

    kv = PostgresMemoryBackend(DSN, user_id="alice")
    short_term = ShortTermStore(backend=kv)
    vectors = PgVectorBackend(DSN, dimensions=1536, user_id="alice")
    long_term = LongTermStore(backend=vectors, backend_name="postgres")
else:
    kv = vectors = None
    short_term = ShortTermStore()
    long_term = LongTermStore()
    print("POSTGRES_URL unset — using in-memory stores")

agent = Agent(
    model=provider,
    role="Personal Assistant",
    instructions="Remember user preferences.",
    session_id="demo-session",
    user_id="alice",
)


async def main() -> None:
    r1 = await agent.arun("I prefer dark mode and Python.")
    print("Session 1:", r1.output.text())
    r2 = await agent.arun("What are my preferences?")
    print("Session 2:", r2.output.text())
    if kv is not None:
        await short_term.write("prefs", {"theme": "dark", "lang": "python"})
        print("KV:", await short_term.read("prefs"))
        await kv.aclose()
        await vectors.aclose()


asyncio.run(main())
