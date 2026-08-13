"""User Memory with Postgres — Cross-session persistence.

USE WHEN: You need durable KV / vector memory across restarts.

Requires: pip install 'loomable[postgres]'
Requires: A running PostgreSQL instance (POSTGRES_URL).
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
    from loomable.providers.backends.postgres import (
        PgVectorBackend,
        PostgresMemoryBackend,
    )

    kv = PostgresMemoryBackend(url=DSN, user_id="alice")
    short_term = ShortTermStore(backend=kv)
    vectors = PgVectorBackend(url=DSN, dimensions=1536, user_id="alice")
    long_term = LongTermStore(backend=vectors, backend_name="postgres")
    print(f"Using Postgres backends at {DSN.split('@')[-1]}")
else:
    short_term = ShortTermStore()  # SQLite :memory:
    long_term = LongTermStore()
    print("POSTGRES_URL not set — using in-memory short/long-term stores")

# Demo agent (conversation turns still use Agent session_store; KV/vector
# backends are ready for ShortTermStore / LongTermStore / knowledge wiring).
agent = Agent(
    model=provider,
    role="Personal Assistant",
    goal="Remember user preferences across sessions",
    instructions="Remember everything the user tells you about their preferences.",
    session_id="demo-session",
    user_id="alice",
)

result = asyncio.run(agent.arun("I prefer dark mode and Python over JavaScript."))
print(f"Session 1: {result.output.text()}\n")

result = asyncio.run(agent.arun("What are my preferences?"))
print(f"Session 2: {result.output.text()}")

# Exercise KV backend when Postgres is configured
async def _demo_kv() -> None:
    if not DSN:
        return
    await short_term.write("alice:prefs", {"theme": "dark", "lang": "python"})
    print("KV readback:", await short_term.read("alice:prefs"))
    await kv.aclose()
    await vectors.aclose()

asyncio.run(_demo_kv())
