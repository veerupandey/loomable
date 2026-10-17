"""Write and recall a record from a workflow blackboard. Runs offline."""

import asyncio

from loomable import Workflow


async def record_evidence(incident, *, context):
    await context.memory.write(f"{incident}: checkout recovered at 14:30 UTC.")
    return incident


async def recall_evidence(incident, *, context):
    records = await context.memory.recall(incident.text())
    return records[0]["record"]


async def main():
    workflow = (Workflow("incident notes", memory=True)
                .step("record", record_evidence)
                .step("recall", recall_evidence))
    print((await workflow.arun("INC-2041")).text)


if __name__ == "__main__":
    asyncio.run(main())
