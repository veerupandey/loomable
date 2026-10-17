"""Review a change in parallel, then combine both reviews in a final step."""

import asyncio

from examples._provider import make_provider
from loomable import Agent, Workflow


async def main():
    async with make_provider() as model:
        performance = Agent(model, instructions="Identify one performance risk.")
        security = Agent(model, instructions="Identify one security risk.")
        writer = Agent(model, instructions="Combine the reviews into a short recommendation.")

        workflow = (
            Workflow("API review")
            .parallel(performance=performance, security=security)
            .step("recommend", writer)
        )
        result = await workflow.arun("We plan to add an endpoint that exports all orders as CSV.")
        for name in ("performance", "security"):
            print(f"{name}: {workflow.state.get(name).text()}")
        print(f"Recommendation: {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
