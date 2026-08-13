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
    headline: str = Field(description="Short news headline")
    summary: str = Field(description="2-4 sentence summary")
    mood: Literal["positive", "mixed", "negative", "unclear"] = Field(
        description="Overall tone of coverage"
    )
    key_points: list[str] = Field(default_factory=list, description="Bullet facts")
    sources: list[str] = Field(default_factory=list, description="Source titles or URLs")


class TopicCard(BaseModel):
    name: str = Field(description="Canonical topic/product name")
    what_it_is: str = Field(description="Plain-English definition")
    origin_or_context: str = Field(description="Where it comes from / context")
    why_it_matters: str = Field(description="Why people care")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the identification"
    )


async def news_brief() -> None:
    agent = Agent(
        model=make_provider(),
        role="Briefing writer",
        goal="Return a structured news brief",
        instructions=(
            "Use web_search. Final answer MUST be raw JSON with EXACT keys: "
            'headline, summary, mood ("positive"|"mixed"|"negative"|"unclear"), '
            "key_points (array of strings), sources (array of strings). "
            "No markdown fences."
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
            "You may use web_search. After researching, your FINAL message must be "
            "ONLY a raw JSON object (no prose before/after) with EXACT keys: "
            "name, what_it_is, origin_or_context, why_it_matters, "
            'confidence ("high"|"medium"|"low").'
        ),
        tools=[WebSearchTools()],
        response_model=TopicCard,
    )
    result = await agent.arun(
        "Make a structured topic card for kageha mactha / matcha. "
        "Reply with JSON only, for example: "
        '{"name":"...","what_it_is":"...","origin_or_context":"...",'
        '"why_it_matters":"...","confidence":"medium"}'
    )
    card = result.structured
    assert isinstance(card, TopicCard)
    print("\n=== TopicCard ===")
    print(card.model_dump_json(indent=2))
    if result.tool_activity:
        print(f"[tools used: {len(result.tool_activity)}]")


async def main() -> None:
    await news_brief()
    await topic_card()


if __name__ == "__main__":
    asyncio.run(main())
