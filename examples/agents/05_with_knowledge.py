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
from _provider import require_provider  # noqa: E402

from loomable.agent import Agent
from loomable.providers import GeminiEmbedder

provider = require_provider()
embedder = GeminiEmbedder()

# Passive snippets — embedded at build, recalled into context (no search tool)
knowledge_docs = [
    "Loomable is a lightweight Python agent framework. It uses a kernel/agent "
    "architecture where the kernel provides stable contracts and the agent layer "
    "composes them into runnable agents.",
    "The Runnable protocol requires an `arun(input, context=None) -> RunResult` "
    "method. Any object satisfying this protocol can be used as a Flow node.",
    "Memory in loomable has three tiers: L1 (raw turns), L2 (summaries), and "
    "L3 (long-term semantic via vectors). Compaction automatically moves L1 "
    "turns into L2 summaries when the window overflows.",
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
