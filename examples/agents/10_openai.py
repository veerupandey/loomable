"""Run a Loomable agent against OpenAI. Set OPENAI_API_KEY first."""
import os

from loomable import Agent
from loomable.providers.openai import OpenAIProvider


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run this example.")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    agent = Agent(OpenAIProvider(model), instructions="Answer in one sentence.")
    print(agent.run("What is the capital of France?").text)
