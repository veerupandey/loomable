"""11 — Route Flow: Dynamic Branching

A RouterNode evaluates input and routes to the appropriate handler.
Only the selected branch executes — efficient for multi-purpose agents.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow import route
from loomable.providers.openai import AzureOpenAIProvider

# --- Router: classifies intent ---


async def classify_intent(input, **kwargs):
    """Simple keyword-based classifier (in production, use a model)."""
    text = str(input).lower()
    if any(w in text for w in ["bug", "error", "fix", "crash", "broken"]):
        return "bugfix_handler"
    elif any(w in text for w in ["feature", "add", "new", "implement", "create"]):
        return "feature_handler"
    return "general_handler"


# --- Handler agents ---

provider = AzureOpenAIProvider()

bugfix_agent = Agent(
    model=provider,
    instructions="You are a debugging expert. Analyze the bug, suggest root cause, and provide a fix.",
)

feature_agent = Agent(
    model=provider,
    instructions="You are a product engineer. Design the feature with clear implementation steps.",
)

general_agent = Agent(
    model=provider,
    instructions="You are a helpful engineering assistant. Answer concisely.",
)


# --- Build routed flow ---

flow = route(
    chooser=classify_intent,
    choices={
        "bugfix_handler": bugfix_agent,
        "feature_handler": feature_agent,
        "general_handler": general_agent,
    },
)

# Test different routes
queries = [
    "The login page crashes when the password is empty",
    "Add a dark mode toggle to the settings page",
    "What's the difference between REST and GraphQL?",
]

for query in queries:
    result = asyncio.run(flow.arun(query))
    print(f"Query: {query}")
    print(f"Response: {result.output.text()[:150]}...\n")
