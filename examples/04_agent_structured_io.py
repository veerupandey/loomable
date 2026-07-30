"""04 — Structured Input and Output

Demonstrates:
- input_schema: validate inputs against a Pydantic model before any model call
- output_schema: parse model responses into typed objects
"""

import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel

from loomable.agent import Agent, InputValidationError
from loomable.providers.openai import AzureOpenAIProvider


# --- Pydantic input schema ---


class BookQuery(BaseModel):
    title: str
    author: str = ""
    genre: str = ""
    max_results: int = 3


# --- Build agent with input validation ---

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    instructions=(
        "You are a book recommendation engine. When given a book query, "
        "recommend similar books. Respond with a brief list."
    ),
    input_schema=BookQuery,
)

# --- Valid input (dict conforming to BookQuery) ---

print("=== Valid structured input ===")
result = asyncio.run(agent.arun({
    "title": "Dune",
    "author": "Frank Herbert",
    "genre": "sci-fi",
    "max_results": 3,
}))
print(result.output.text())

# --- String input bypasses schema validation ---

print("\n=== String input (bypasses schema) ===")
result = asyncio.run(agent.arun("Recommend books similar to 1984 by Orwell"))
print(result.output.text())

# --- Invalid input raises InputValidationError BEFORE any model call ---

print("\n=== Invalid input (missing required field) ===")
try:
    asyncio.run(agent.arun({"author": "only author, no title"}))
except InputValidationError as e:
    print(f"Caught validation error: {e}")
