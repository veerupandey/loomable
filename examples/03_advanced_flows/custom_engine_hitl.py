"""23 — Human-in-the-Loop (HITL) Tool Approval

Demonstrates safety mechanisms:
- require_confirmation: tools that need human approval before execution
- Tool hooks: pre-hooks that can block dangerous tool calls
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool, ToolHookRejection
from loomable.providers.openai import AzureOpenAIProvider


# --- Tools with different safety levels ---


@tool
def read_data(table: str) -> str:
    """Read data from a table (safe, no approval needed)."""
    data = {"users": "10 records", "orders": "156 records", "products": "42 records"}
    return f"Data from {table}: {data.get(table, 'empty')}"


@tool(idempotent=False)
def delete_record(table: str, record_id: str) -> str:
    """Delete a record (dangerous, needs approval!)."""
    return f"DELETED {record_id} from {table}"


@tool(idempotent=False)
def send_notification(channel: str, message: str) -> str:
    """Send a notification to a channel (side-effecting)."""
    return f"Notification sent to #{channel}: {message}"


# --- Pre-hook: block any tool targeting production ---


def safety_hook(tool_name: str, call, args: dict) -> object:
    """Block tool calls that target production data."""
    if "prod" in str(args).lower():
        raise ToolHookRejection(f"BLOCKED: Cannot operate on production data via '{tool_name}'")
    return True  # Allow


# --- Build agent with HITL ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions=(
        "You are a database admin assistant. You can read data freely, "
        "but destructive operations require confirmation."
    ),
    tools=[read_data, delete_record, send_notification],
    require_confirmation=["delete_record", "send_notification"],
    tool_hooks=[safety_hook],
)

# --- Run (delete/send will be blocked by default approver) ---

result = asyncio.run(agent.arun("How many users do we have? Also clean up record user-999."))
print("Answer:", result.output.text())
print(f"\nTools executed: {len(result.tool_activity)}")
for activity in result.tool_activity:
    print(f"  ✓ {activity.result.content}")

print("\n--- HITL Configuration ---")
print("""
  require_confirmation = ["delete_record", "send_notification"]
  tool_hooks = [safety_hook]  # Blocks "prod" targets

  Default approver denies all confirmation-required tools (headless-safe).
  In production, inject your own approver after building:
    built = agent.build()
    built.approver = lambda call: input(f"Approve {call.tool_name}? ") == "y"
""")
