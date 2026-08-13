"""03 — Structured output brief

USE WHEN: You need typed JSON output (not free-form text) from a simple agent.

Builds a structured India briefing and a structured topic card.
"""

from __future__ import annotations


import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from loomable.agent import Agent
from loomable.toolkits import WebSearchTools

from _provider import make_provider


class NewsBrief(BaseModel):
    headline: str
    summary: str
    mood: Literal["positive", "mixed", "negative", "unclear"]
    key_points: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class TopicCard(BaseModel):
    name: str
    what_it_is: str
    origin_or_context: str
    why_it_matters: str
    confidence: Literal["high", "medium", "low"]


async def news_brief() -> None:
    agent = Agent(
        model=make_provider(),
        role="Briefing writer",
        goal="Return a structured news brief",
        instructions=(
            "Use web_search. Return ONLY JSON matching the schema fields: "
            "headline, summary, mood, key_points, sources."
        ),
        tools=[WebSearchTools()],
        response_model=NewsBrief,
    )
    result = await agent.arun(
        "Give a structured brief on how the Modi government is doing this month."
    )
    brief = result.structured
    assert isinstance(brief, NewsBrief)
    print("=== NewsBrief ===")
    print(brief.model_dump_json(indent=2))


async def topic_card() -> None:
    agent = Agent(
        model=make_provider(),
        role="Researcher",
        goal="Return a structured topic card",
        instructions=(
            "Use web_search. Return ONLY JSON matching: "
            "name, what_it_is, origin_or_context, why_it_matters, confidence."
        ),
        tools=[WebSearchTools()],
        response_model=TopicCard,
    )
    result = await agent.arun("Make a structured topic card for kageha mactha / matcha.")
    card = result.structured
    assert isinstance(card, TopicCard)
    print("\n=== TopicCard ===")
    print(card.model_dump_json(indent=2))


async def main() -> None:
    await news_brief()
    await topic_card()


if __name__ == "__main__":
    asyncio.run(main())
