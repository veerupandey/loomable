"""LangGraph / Agno-style control plane — route + Command + get_state.

USE WHEN: Migrating from LangGraph multi-edge routing or Agno Router workflows.

No live model required.
"""

from __future__ import annotations

import asyncio

from loomable import Command, Workflow
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.persist.checkpoint import InMemoryCheckpointer


def _text(s: str) -> RunResult:
    return RunResult(
        output=AgentOutput(
            parts=[
                MediaPart(
                    modality=Modality.TEXT,
                    media_type="text/plain",
                    data=s.encode("utf-8"),
                )
            ]
        ),
        session_id="",
    )


def classify(change):
    text = str(change).lower()
    if "sev1" in text or "outage" in text:
        return Command(goto="full", update={"severity": "high"})
    if "question" in text:
        return Command(goto="human", update={"severity": "unknown"})
    return Command(goto="quick", update={"severity": "low"})


async def quick(change, *, context=None):
    return _text(f"quick-ack:{change}")


async def full(change, *, context=None):
    sev = context.shared_state.get("severity") if context else "?"
    return _text(f"full-audit[{sev}]:{change}")


async def human(change, *, context=None):
    return _text(f"needs-human:{change}")


async def main() -> None:
    cp = InMemoryCheckpointer()
    wf = Workflow(
        "triage",
        session_id="inc-1",
        checkpointer=cp,
    ).route(classify, quick=quick, full=full, human=human)

    result = await wf.arun("SEV1 database outage")
    print(result.output.text())
    print("state:", await wf.get_state())
    print("decision:", wf.state.get("_route_decision"))


if __name__ == "__main__":
    asyncio.run(main())
