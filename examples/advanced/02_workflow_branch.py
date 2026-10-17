"""Route a support ticket through a yes/no workflow branch."""

import asyncio

from examples._provider import make_provider
from loomable import Agent, Workflow


async def main():
    async with make_provider() as model:
        billing = Agent(model, instructions="Acknowledge the billing issue in one sentence without asking for payment details.")
        general = Agent(model, instructions="Draft a short general support reply.")

        def is_billing(state):
            return "charged" in (state.get("_workflow_input") or "").lower()

        workflow = Workflow("support").branch(when=is_billing, then=billing, else_=general)
        print((await workflow.arun("I was charged twice for order QZ-8421.")).text)


if __name__ == "__main__":
    asyncio.run(main())
