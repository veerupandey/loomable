"""Turn a typed model plan into independent work items, then merge the results."""

import asyncio

from pydantic import BaseModel, Field

from examples._provider import make_provider
from loomable import Agent, Workflow


class Plan(BaseModel):
    tasks: list[str] = Field(min_length=2, max_length=3)


async def main():
    async with make_provider() as model:
        planner = Agent(model, response_model=Plan, instructions="Create two independent review tasks.")
        writer = Agent(model, instructions="Combine the review findings into a short recommendation.")

        async def plan(question):
            result = await planner.arun(question)
            return {"plan_steps": result.structured.tasks}

        async def review(task):
            # Each item gets its own runtime and shares the model's HTTP pool.
            worker = Agent(model, instructions="Answer the assigned review task in one sentence.")
            return await worker.arun(task)

        workflow = Workflow("release review").map(review, planner=plan, synthesizer=writer)
        result = await workflow.arun("Review a plan to cache checkout prices for five minutes.")
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
