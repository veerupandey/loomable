"""Production-Ready Agent — All hardening features enabled.

USE WHEN: Building a real application that needs resilience,
observability, HITL gates, and structured output.

Combines: retry policy, tool hooks, event tracing, verification.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool, AgentEvents, JSONTracer
from loomable.providers import AzureOpenAIProvider, RetryPolicy

# --- Tools ---


@tool(idempotent=False)
def send_notification(recipient: str, message: str) -> str:
    """Send a notification to a user (simulated)."""
    return f"Notification sent to {recipient}: {message}"


# --- Approval gate ---


def approval_gate(tool_call) -> bool:
    """Auto-approve in demo; in production, prompt the user."""
    print(f"  [HITL] Auto-approving: {tool_call.name}({tool_call.arguments})")
    return True


# --- Build production agent ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    role="Operations Assistant",
    goal="Manage notifications with safety gates",
    instructions="Send notifications when asked. Always confirm the recipient.",
    tools=[send_notification],
    require_confirmation=["send_notification"],
    resilience=RetryPolicy(max_attempts=3, base_delay=1.0),
    events=JSONTracer(),
    tool_timeout=10.0,
    tool_concurrency=2,
)
built = agent.build()
built.approver = approval_gate

result = asyncio.run(built.arun("Send a notification to alice@co saying 'deploy complete'"))
print(result.output.text())
print(f"\nTrace events: {len(result.trace)}")
