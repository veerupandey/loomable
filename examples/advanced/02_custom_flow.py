"""Custom Flow — Build your own graph with Node + Edge.

USE WHEN: The built-in patterns (sequential, parallel, route)
don't fit your topology. You need conditional edges, cycles,
or custom routing logic.

For most branching, prefer ``Workflow(...).branch(...)`` or
``route(...)`` from ``loomable.flow.helpers``. This example is the
power-user layer for explicit DAG construction.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent
from loomable.content import AgentOutput
from loomable.flow.flow import Flow
from loomable.flow.nodes import Node, Edge
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()

# --- Define nodes ---

classifier = Agent(
    model=provider,
    role="Intent Classifier",
    goal="Classify the input into a category",
    instructions="Classify as 'technical' or 'creative'. Output only the category word.",
)

technical_handler = Agent(
    model=provider,
    role="Technical Expert",
    goal="Handle technical questions",
    instructions="Provide a clear technical answer.",
)

creative_handler = Agent(
    model=provider,
    role="Creative Writer",
    goal="Handle creative requests",
    instructions="Provide an imaginative, creative response.",
)


def _node_text(state, key: str) -> str:
    """SharedState stores AgentOutput (or plain values) — read text safely."""
    value = state.get(key)
    if value is None:
        return ""
    if isinstance(value, AgentOutput):
        return (value.text() or "").lower()
    if hasattr(value, "text") and callable(value.text):
        return (value.text() or "").lower()
    return str(value).lower()


# --- Build custom flow with conditional edges ---

nodes = {
    "classify": Node(node_id="classify", runnable=classifier),
    "technical": Node(node_id="technical", runnable=technical_handler),
    "creative": Node(node_id="creative", runnable=creative_handler),
}

edges = [
    Edge(
        source="classify",
        target="technical",
        condition=lambda state: "technical" in _node_text(state, "classify"),
    ),
    Edge(
        source="classify",
        target="creative",
        condition=lambda state: "creative" in _node_text(state, "classify"),
    ),
]

flow = Flow(nodes, edges=edges, engine="sequential")

result = asyncio.run(flow.arun("Write a poem about recursion"))
print(result.output.text())
