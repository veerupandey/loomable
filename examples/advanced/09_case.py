"""Give a Case a goal and validate the handoff after planning and execution."""

import asyncio

from examples._provider import make_provider
from loomable import Case, VerdictResult


def accept(output, _context):
    text = output.text()
    passed = "REL-208" in text and "rollback" in text.lower()
    return VerdictResult(ok=passed, detail="Include REL-208 and a rollback check.")


async def main():
    async with make_provider() as model:
        case = Case(
            model=model,
            goal="Write a short deployment handoff for REL-208.",
            accept=accept,
            max_rounds=2,
            max_steps=3,
        )
        result = await case.arun("REL-208 changes the checkout schema. Include a rollback check.")
        print(result.text)
        print(f"Accepted {result.metadata.get('loop_verified')}")


if __name__ == "__main__":
    asyncio.run(main())
