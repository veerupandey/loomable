"""Production-Ready Agent — All hardening features enabled.

USE WHEN: Building a real application that needs resilience,
observability, HITL gates, and structured output.

Combines: retry policy, tool hooks, event tracing, verification.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402

from loomable.agent import Agent, tool, JSONTracer
from loomable.providers import RetryPolicy

# --- Tools ---


@tool(idempotent=False)
def send_notification(recipient: str, message: str) -> str:
    """Send a notification to a user (simulated)."""
    return f"Notification sent to {recipient}: {message}"


# --- Approval gate ---


def approval_gate(tool_call) -> bool:
    """Auto-approve in demo; in production, prompt the user."""
    print(f"  [HITL] Auto-approving: {tool_call.tool_name}({tool_call.args})")
    return True


# --- Build production agent ---

provider = require_provider()

agent = Agent(
    model=provider,
    role="Operations Assistant",
    goal="Manage notifications with safety gates",
    instructions="Send notifications when asked. Always confirm the recipient.",
    tools=[send_notification],
    require_confirmation=["send_notification"],
    approver=approval_gate,
    resilience=RetryPolicy(max_attempts=3, base_delay=1.0),
    events=JSONTracer(),
    tool_timeout=10.0,
    tool_concurrency=2,
)
result = asyncio.run(agent.arun("Send a notification to alice@co saying 'deploy complete'"))
print(result.output.text())
print(f"\nTrace events: {len(result.trace)}")
