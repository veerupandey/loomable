"""Pass Agent memory via sqlite / file / postgres.

Requires for postgres: pip install 'loomable[postgres]' && docker compose up -d
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.memory import open_session_store
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()
DSN = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
ROOT = Path(__file__).resolve().parent / ".sessions_demo"
ROOT.mkdir(exist_ok=True)

# Pick one:
if DSN:
    store = open_session_store("postgres", url=DSN, user_id="alice")
    label = "postgres"
else:
    store = open_session_store("file", path=str(ROOT))
    label = f"file:{ROOT}"


async def main() -> None:
    print(f"Using session store: {label}")
    a1 = Agent(
        model=provider,
        session_id="demo-session",
        session_store=store,
        instructions="Remember user preferences.",
    )
    print((await a1.arun("I prefer dark mode and Python.")).output.text())

    # New Agent instance — same store + resume=True restores L1/L2
    a2 = Agent(
        model=provider,
        session_id="demo-session",
        session_store=store,
        resume=True,
        instructions="Remember user preferences.",
    )
    print((await a2.arun("What are my preferences?")).output.text())


asyncio.run(main())
