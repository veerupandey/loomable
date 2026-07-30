"""20 — MCP Tools in a Flow

Demonstrates MCP-style tools available to agents within a flow.
Different agents in the pipeline have access to different tool sets.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool
from loomable.flow import sequential
from loomable.providers.openai import AzureOpenAIProvider


# --- MCP-style tools for different stages ---


@tool
def search_codebase(query: str) -> str:
    """Search the codebase for relevant code (simulates MCP code-search server)."""
    results = {
        "auth": "Found: src/auth/jwt.py - JWT validation middleware, 45 lines",
        "user": "Found: src/models/user.py - User model with email, name fields",
        "api": "Found: src/routes/api.py - REST endpoints for /users, /posts",
    }
    for key, val in results.items():
        if key in query.lower():
            return val
    return f"No code found matching: {query}"


@tool
def create_pr(title: str, description: str, branch: str) -> str:
    """Create a pull request (simulates MCP GitHub server)."""
    return f"PR created: '{title}' on branch '{branch}' - Ready for review"


# --- Flow: research → implement → ship ---

provider = AzureOpenAIProvider()

researcher = Agent(
    model=provider,
    instructions="You are a code researcher. Use search_codebase to understand the relevant code, then summarize what you found.",
    tools=[search_codebase],
)

implementer = Agent(
    model=provider,
    instructions="You are a developer. Based on the research, describe what code changes you would make (2-3 bullet points).",
)

shipper = Agent(
    model=provider,
    instructions="You are a release engineer. Use create_pr to ship the changes described. Provide a clear PR title and description.",
    tools=[create_pr],
)

# --- Compose flow ---

flow = sequential(researcher, implementer, shipper, session_id="dev-flow")

print("Running: research(MCP) → implement → ship(MCP)\n")
result = asyncio.run(flow.arun("Add input validation to the user registration endpoint"))
print("=== Result ===")
print(result.output.text())
