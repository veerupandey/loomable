"""Router — Intent-based routing to specialists.

USE WHEN: Different types of input should go to different agents
based on what the user is asking (intent classification).

Uses the `route` flow helper with a chooser function.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.flow.helpers import route
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

# --- Specialist agents ---

code_agent = Agent(
    model=provider,
    role="Code Assistant",
    goal="Write and explain code",
    instructions="Help with programming questions. Include code examples.",
)

math_agent = Agent(
    model=provider,
    role="Math Tutor",
    goal="Solve math problems step by step",
    instructions="Solve math problems showing your work.",
)

general_agent = Agent(
    model=provider,
    role="General Assistant",
    goal="Answer general knowledge questions",
    instructions="Answer clearly and concisely.",
)


# --- Router: classifies intent ---

def classify_intent(input_text) -> str:
    """Simple keyword-based routing (use an LLM classifier in production)."""
    text = str(input_text).lower()
    if any(kw in text for kw in ["code", "python", "function", "program", "bug"]):
        return "code"
    if any(kw in text for kw in ["math", "calculate", "equation", "solve"]):
        return "math"
    return "general"


router = route(
    classify_intent,
    {"code": code_agent, "math": math_agent, "general": general_agent},
)

result = asyncio.run(router.arun("Solve the quadratic equation x^2 - 5x + 6 = 0"))
print(result.output.text())
