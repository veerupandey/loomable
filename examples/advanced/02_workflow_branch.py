"""Binary Workflow branch — ``Workflow.branch(when=…, then=…, else_=…)``.

USE WHEN: After a step (often a classifier Agent), you need a **yes/no** fork
inside a Workflow. Each Agent receives the previous output automatically.

Do **not** use this for:

- LLM picks a specialist Team member → ``Team(mode="route")``
  (``patterns/04_router.py``)
- Three or more named arms / ``Command(goto=…)`` → ``Workflow.route``
  (``patterns/08_route_command.py``)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, Workflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import require_provider  # noqa: E402


async def main() -> None:
    model = require_provider()

    classifier = Agent(
        model,
        role="Intent Classifier",
        goal="Classify the input into a category",
        instructions="Classify as 'technical' or 'creative'. Output only that one word.",
    )
    technical = Agent(
        model,
        role="Technical Expert",
        goal="Handle technical questions",
        instructions="Provide a clear technical answer.",
    )
    creative = Agent(
        model,
        role="Creative Writer",
        goal="Handle creative requests",
        instructions="Provide an imaginative, creative response.",
    )

    def is_technical(state) -> bool:
        value = state.get("classify")
        text = value.text() if hasattr(value, "text") else str(value or "")
        return "technical" in text.lower()

    wf = (
        Workflow("intent-branch")
        .step("classify", classifier)
        .branch(when=is_technical, then=technical, else_=creative)
    )
    result = await wf.arun("Write a poem about recursion")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
