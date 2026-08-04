"""Agent with Knowledge (RAG) — Retrieval-augmented generation.

USE WHEN: Your agent needs to answer questions about specific
documents or data it wasn't trained on.

Knowledge docs are embedded at build time and recalled at runtime
via vector similarity search.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider
from loomable.providers import AzureOpenAIEmbedder

provider = AzureOpenAIProvider()
embedder = AzureOpenAIEmbedder()

# Knowledge base: these docs get embedded and indexed at build time
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
