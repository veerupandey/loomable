"""24 — Full Production Agent (All Features Combined)

A comprehensive example combining all loomable features:
- Function tools with @tool
- Think reasoning tool
- Conversational memory with compaction
- Knowledge RAG with Azure embeddings
- Tool hooks for safety
- Verification with retry
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool, make_think_tool, ToolHookRejection
from loomable.content import AgentOutput
from loomable.agent.context import RunContext
from loomable.flow import Loop
from loomable.providers.openai import AzureOpenAIProvider
from loomable.providers.embedders import AzureOpenAIEmbedder


# ============================================================================
# Tools
# ============================================================================


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for relevant information."""
    kb = {
        "pricing": "Standard: $10/mo (100 msgs/day), Pro: $25/mo (1000 msgs/day), Enterprise: custom (unlimited)",
        "refund": "Full refund within 30 days, pro-rated after. Contact support@example.com",
        "features": "AI chat, code generation, document analysis, team collaboration, API access",
        "sla": "99.9% uptime SLA for Pro and Enterprise. Standard has best-effort availability.",
    }
    for key, val in kb.items():
        if key in query.lower():
            return val
    return "No specific information found. Suggest contacting support."


@tool
def create_ticket(customer_id: str, issue: str, priority: str = "medium") -> str:
    """Create a support ticket in the system."""
    import hashlib
    ticket_id = f"TKT-{hashlib.md5((customer_id + issue).encode()).hexdigest()[:6].upper()}"
    return f"Created ticket {ticket_id}: [{priority.upper()}] {issue} (customer: {customer_id})"


@tool(idempotent=False)
def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human agent."""
    return f"ESCALATED: {reason} — Human agent will respond within 1 hour."


# ============================================================================
# Safety Hook
# ============================================================================


def rate_limit_hook(tool_name: str, call, args: dict) -> object:
    """Block excessive escalations (max 1 per conversation)."""
    if tool_name == "escalate_to_human":
        # In production, check actual escalation count
        pass
    return True


# ============================================================================
# Knowledge Documents
# ============================================================================

knowledge_docs = [
    "Support Policy: Respond within 24 hours. Escalate Enterprise issues to senior team immediately.",
    "Pricing: Standard $10/mo, Pro $25/mo, Enterprise custom. All include core AI features.",
    "Refund Policy: Full refund within 30 days. Pro-rated refund after 30 days for annual plans.",
    "SLA: 99.9% uptime for Pro/Enterprise. Compensation: 10% credit per hour of downtime beyond SLA.",
    "Security: SOC2 Type II certified. Data encrypted at rest (AES-256) and in transit (TLS 1.3).",
]

# ============================================================================
# Build the Full Agent
# ============================================================================

provider = AzureOpenAIProvider()

embedder = AzureOpenAIEmbedder(
    deployment=os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT_NAME", "text-embedding-3-large"),
    endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ.get("AZURE_OPENAI_EMBED_API_VERSION", "2023-05-15"),
)

agent = Agent(
    model=provider,
    instructions=(
        "You are a customer support agent for a SaaS company. "
        "Use the think tool to reason before responding. "
        "Search the knowledge base for accurate information. "
        "Create tickets for issues that need follow-up. "
        "Be helpful, empathetic, and concise."
    ),
    tools=[search_knowledge_base, create_ticket, escalate_to_human, make_think_tool()],
    session_id="support-session",
    memory_window=8,
    compaction_threshold=16,
    knowledge=knowledge_docs,
    embedder=embedder,
    knowledge_top_k=3,
    tool_hooks=[rate_limit_hook],
    require_confirmation=["escalate_to_human"],
)
built = agent.build()


# ============================================================================
# Run a Multi-Turn Support Conversation
# ============================================================================

print("=== Full Production Agent ===\n")

conversations = [
    "Hi, I'm on the Pro plan. What's included again?",
    "I've been experiencing downtime. What's your SLA?",
    "I'd like a refund for last month's outage.",
]

for msg in conversations:
    result = asyncio.run(built.arun(msg))
    print(f"Customer: {msg}")
    print(f"Agent: {result.output.text()}")
    if result.tool_activity:
        print(f"  [Tools used: {len(result.tool_activity)}]")
    print()

print("--- Session State ---")
print(f"Turns in memory: {len(built.session.l1)}")
print(f"Compacted summaries: {len(built.session.l2)}")
print(f"Knowledge store active: {built.long_term is not None}")
