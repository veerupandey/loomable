"""Graph engineering patterns — failure policy, verify gate, edge reads.

USE WHEN: You want an explicit execution map instead of one long agent chain.

Patterns demonstrated (no live model required):

1. Fan-out independent work, join for synthesis
2. Local failure policy (``on_failure="skip"``) so one branch cannot freeze the job
3. Edge data contract (``reads=``) so draft consumes structured evidence, not
   ambient previous-node text
4. ``Workflow.verify`` — generate → check → repair with a hard budget
"""

from __future__ import annotations

import asyncio

from loomable import Step, Workflow
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality


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


def _as_text(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "text") and callable(value.text):
        return value.text() or ""
    return str(value)


async def company_sources(topic, *, context=None):
    return _text(f"company:{topic}")


async def papers(topic, *, context=None):
    return _text(f"papers:{topic}")


async def flaky_expert(topic, *, context=None):
    # Optional lane — failure stays local via on_failure="skip"
    raise RuntimeError("expert feed unavailable")


async def dedupe(topic, *, context=None):
    state = context.shared_state if context else None
    parts: list[str] = []
    if state:
        # Parallel group writes each branch under its step name
        for key in ("company", "papers", "expert"):
            text = _as_text(state.get(key)).strip()
            if text:
                parts.append(text)
    evidence = " | ".join(parts) or str(topic)
    if state:
        state.write(
            "evidence",
            AgentOutput(
                parts=[
                    MediaPart(
                        modality=Modality.TEXT,
                        media_type="text/plain",
                        data=evidence.encode("utf-8"),
                    )
                ]
            ),
        )
    return _text(evidence)


_draft_attempts = {"n": 0}


async def draft(evidence, *, context=None):
    _draft_attempts["n"] += 1
    text = _as_text(evidence)
    # First attempt is incomplete; verifier forces one repair round
    if _draft_attempts["n"] == 1:
        return _text(f"DRAFT (incomplete): {text}")
    return _text(f"FINAL: {text}\nSources checked.")


def has_sources(output, ctx) -> bool:
    return "Sources checked" in output.text()


async def main() -> None:
    wf = (
        Workflow("research_publish")
        .parallel(
            Step("company", company_sources),
            Step("papers", papers),
            Step("expert", flaky_expert, on_failure="skip"),
        )
        .step("dedupe", dedupe)
        .verify(
            Step("draft", draft, reads="evidence"),
            check=has_sources,
            max_retries=2,
        )
    )

    result = await wf.arun("graph engineering")
    print(result.output.text())
    print("---")
    print("verified:", result.metadata.get("loop_verified"))
    print("plan:", wf.explain())


if __name__ == "__main__":
    asyncio.run(main())
