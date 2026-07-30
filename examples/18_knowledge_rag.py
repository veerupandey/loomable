"""18 — Knowledge RAG (Embedder + LongTermStore)

Demonstrates attaching knowledge documents to an agent:
- Documents are embedded and indexed at build time (using Azure OpenAI embeddings)
- At run time, the input is embedded and relevant docs are recalled into context
- The model sees the recalled knowledge alongside the user's question
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.providers.openai import AzureOpenAIProvider
from loomable.providers.embedders import AzureOpenAIEmbedder

# --- Knowledge documents (your project's docs, policies, etc.) ---

knowledge_docs = [
    "Loomable is a lightweight agent framework with 3 tiers: Agent (single call), Loop (retry with verification), and Flow (directed graph of runnables).",
    "The @tool decorator turns any Python function into an agent tool. It auto-derives a JSON schema from the function signature and type hints.",
    "Flows support four patterns: sequential (pipeline), parallel (concurrent branches), route (conditional), and coordinate (workers + manager).",
    "Memory in loomable has 4 tiers: WORKING (current run scratch), EPISODIC (past events), SEMANTIC (long-term knowledge), and PROCEDURAL (learned rules).",
    "MCP (Model Context Protocol) servers can be connected to any agent via mcp_servers= config. Failed connections are isolated — other tools still work.",
    "The plan tool dynamically creates a plan→map→synthesize flow, enabling an agent to decompose complex tasks into parallel subtasks at runtime.",
    "Tiered model routing allows configuring multiple models (e.g., GPT-4 primary, GPT-3.5 fallback) with automatic failover and substitution recording.",
]

# --- Build agent with knowledge RAG ---

provider = AzureOpenAIProvider()

# Use the Azure embeddings endpoint (reads AZURE_OPENAI_EMBED_DEPLOYMENT_NAME from .env)
import os
embedder = AzureOpenAIEmbedder(
    deployment=os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME", "text-embedding-3-large"),
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_EMBED_API_VERSION", "2023-05-15"),
)

agent = Agent(
    model=provider,
    instructions=(
        "You are a loomable framework expert. Answer questions using the knowledge "
        "provided in your context. If the answer is in your knowledge base, cite it. "
        "If not, say you don't have information about that."
    ),
    knowledge=knowledge_docs,
    embedder=embedder,
    knowledge_top_k=3,  # Recall top 3 most relevant docs
)

# --- Query ---

queries = [
    "What patterns do Flows support?",
    "How does the @tool decorator work?",
    "What happens when an MCP server connection fails?",
]

for query in queries:
    result = asyncio.run(agent.arun(query))
    print(f"Q: {query}")
    print(f"A: {result.output.text()}\n")
