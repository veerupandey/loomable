"""Passive knowledge recall — short strings injected into context.

USE WHEN: You have a few FAQ / policy snippets and do not need the model
to call a search tool. Docs are embedded at build and the top-k snippets
are prepended each turn.

For a searchable vector-DB knowledge base (files, PDFs, named collections),
see ``07_knowledge_base.py`` (``Agent(knowledge_base=...)``).
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import make_embedder, require_provider  # noqa: E402

from loomable.agent import Agent

provider = require_provider()
embedder = make_embedder()

# Passive snippets — embedded at build, recalled into context (no search tool)
knowledge_docs = [
    "Loomable is a lightweight Python agent framework. It uses a kernel/agent "
    "architecture where the kernel provides stable contracts and the agent layer "
    "composes them into runnable agents.",
    "The Runnable protocol requires an `arun(input, context=None) -> RunResult` "
    "method. Any object satisfying this protocol can be a Workflow step or Team member.",
    "Memory in loomable composes conversation (L1/L2), user notes (L3), and "
    "optional knowledge. Prefer Memory.compose for durable chat + long-term facts.",
]

agent = Agent(
    model=provider,
    role="Documentation Assistant",
    goal="Answer questions about the loomable framework",
    instructions="Answer based on the provided knowledge. Say 'I don't know' if unsure.",
    knowledge=knowledge_docs,
    embedder=embedder,
    knowledge_top_k=2,
)

result = asyncio.run(agent.arun("What is the Runnable protocol in loomable?"))
print(result.output.text())
