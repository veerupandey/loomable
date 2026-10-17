"""Run a release review through Workflow and Team with real model calls."""

import asyncio
from collections import Counter
from contextlib import asynccontextmanager

import pytest
from pydantic import BaseModel

from loomable import Agent, Command, FlowPaused, SQLiteCheckpointer, Team, Workflow, tool
from loomable.persist import CheckpointBusy
from loomable.providers import ResilientModel, RetryPolicy
from loomable.stream import RUN_ERROR, RUN_FINISHED, RUN_PAUSED
from tests.integration.test_model_live import live_provider

pytestmark = pytest.mark.live

PACKET = "Release REL-208 changes the checkout database schema. Prepare a release review."


@asynccontextmanager
async def release_provider():
    async with live_provider() as provider:
        yield ResilientModel(provider, RetryPolicy(max_attempts=2, per_call_timeout=22))


def specialist(provider, role, evidence, calls):
    @tool
    def read_evidence() -> str:
        """Read the specialist's release evidence."""
        calls[role] += 1
        return evidence

    return Agent(
        provider, role=role, tools=[read_evidence], require_tools=["read_evidence"],
        strict_require_tools=True, instructions="Call read_evidence and return its evidence label verbatim.",
    )


def text(value):
    return value if isinstance(value, str) else value.text()


async def test_release_review_fans_out_routes_and_resumes(tmp_path):
    calls = Counter()
    published = []
    publishing = asyncio.Event()
    release = asyncio.Event()
    db = str(tmp_path / "review.sqlite")

    async def publish(value):
        published.append(text(value))
        publishing.set()
        await release.wait()
        return "Handoff recorded"

    def choose(_value, *, context):
        original = context.shared_state.get("_workflow_input")
        return Command(goto="schema" if "schema" in original else "ordinary")

    async def ordinary(_value):
        pytest.fail("schema changes must use the schema review")

    async with release_provider() as provider, asyncio.timeout(75):
        def app():
            reviewers = Team([
                specialist(provider, "Rollback reviewer", "ROLLBACK-204", calls),
                specialist(provider, "Data reviewer", "DATA-508", calls),
            ], model=provider, mode="broadcast", strict=True)
            release_notes = specialist(provider, "Release notes", "NOTES-387", calls)
            return (
                Workflow("release review", checkpointer=SQLiteCheckpointer(db), session_id="REL-208")
                .parallel(review=reviewers, notes=release_notes)
                .route(choose, schema=lambda _: "Schema review selected", ordinary=ordinary)
                .step("publish", publish, reads="review", confirm=True)
            )

        workflow = app()
        with pytest.raises(FlowPaused) as paused:
            await workflow.arun(PACKET)
        assert paused.value.node_id == "publish"
        assert published == []
        assert calls == {"Rollback reviewer": 1, "Data reviewer": 1, "Release notes": 1}
        reopened = app()
        await reopened.approve("publish")
        owner = asyncio.create_task(reopened.arun(resume=True))
        try:
            await asyncio.wait_for(publishing.wait(), 10)
            with pytest.raises(CheckpointBusy):
                await app().arun(resume=True)
            with pytest.raises(CheckpointBusy):
                await app().approve("publish", status="rejected")
        finally:
            release.set()
            result = await owner
        assert result.text == "Handoff recorded"
        assert len(published) == 1 and all(label in published[0] for label in ("ROLLBACK-204", "DATA-508"))
        assert calls == {"Rollback reviewer": 1, "Data reviewer": 1, "Release notes": 1}
        checkpoint = await SQLiteCheckpointer(db).get("REL-208")
        assert checkpoint.complete


async def test_release_team_routes_only_to_the_relevant_specialist():
    calls = Counter()
    async with release_provider() as provider, asyncio.timeout(60):
        team = Team([
            specialist(provider, "Documentation", "DOCS-136", calls),
            specialist(provider, "Database migration", "MIGRATION-816", calls),
        ], model=provider, mode="route")
        result = await team.arun("Get the database migration evidence label. This is a schema change.")
        assert calls == {"Database migration": 1}
        assert "MIGRATION-816" in result.text
        events = [event async for event in team.astream_events("Get the documentation evidence label.")]
    assert calls == {"Database migration": 1, "Documentation": 1}
    assert events[-1].type == RUN_FINISHED
    assert "DOCS-136" in events[-1].data["text"]


class Handoff(BaseModel):
    label: str


async def test_release_team_sequential_stream_returns_typed_handoff():
    calls = Counter()
    async with release_provider() as provider, asyncio.timeout(60):
        reviewer = specialist(provider, "Change reviewer", "CHANGE-619", calls)
        formatter = Agent(provider, role="Handoff formatter", instructions=(
            "Use the previous member's evidence label as the label in your JSON handoff."
        ))
        team = Team([reviewer, formatter], model=provider, mode="sequential", strict=True)
        chunks = [chunk async for chunk in team.astream(PACKET, output_schema=Handoff)]
    assert calls == {"Change reviewer": 1}
    assert chunks[-1].done
    assert chunks[-1].result.structured.label == "CHANGE-619"


async def test_release_workflow_stream_pauses_then_resumes(tmp_path):
    calls = Counter()
    store = SQLiteCheckpointer(str(tmp_path / "stream.sqlite"))
    async with release_provider() as provider, asyncio.timeout(60):
        reviewer = specialist(provider, "Rollback reviewer", "ROLLBACK-204", calls)
        workflow = (Workflow("streamed release", checkpointer=store, session_id="release-stream")
                    .step("review", reviewer)
                    .step("publish", lambda output: text(output), confirm=True))
        events = [event async for event in workflow.astream_events(PACKET)]
        assert events[-1].type == RUN_PAUSED and events[-1].data["node_id"] == "publish"
        await workflow.approve("publish")
        events = [event async for event in workflow.astream_events(resume=True)]
    assert events[-1].type == RUN_FINISHED
    assert "ROLLBACK-204" in events[-1].data["text"]
    assert calls == {"Rollback reviewer": 1}


async def test_approval_before_first_step_restores_original_release_packet(tmp_path):
    store = SQLiteCheckpointer(str(tmp_path / "approval.sqlite"))
    async with release_provider() as provider, asyncio.timeout(60):
        def app():
            reviewer = Agent(provider, instructions="Return only the release label from the packet.")
            return (Workflow("approved review", checkpointer=store, session_id="approved-release")
                    .step("review", reviewer, confirm=True))

        with pytest.raises(FlowPaused):
            await app().arun(PACKET)
        resumed = app()
        await resumed.approve("review")
        result = await resumed.arun(resume=True)
    assert "REL-208" in result.text


async def test_strict_team_stream_reports_member_failure():
    calls = Counter()

    @tool
    def unavailable_evidence() -> str:
        """Read release evidence from a temporarily unavailable source."""
        raise RuntimeError("evidence unavailable")

    async with release_provider() as provider, asyncio.timeout(60):
        unavailable = Agent(
            provider, role="Unavailable reviewer", tools=[unavailable_evidence],
            require_tools=["unavailable_evidence"], strict_require_tools=True, max_tool_iterations=2,
            instructions="Call unavailable_evidence before responding.",
        )
        team = Team([
            specialist(provider, "Change reviewer", "CHANGE-619", calls), unavailable,
        ], model=provider, mode="broadcast", strict=True)
        events = [event async for event in team.astream_events(PACKET)]
    assert events[-1].type == RUN_ERROR
    assert all(event.type != RUN_FINISHED for event in events)


async def test_release_coordinator_delegates_to_a_nested_team():
    calls = Counter()
    async with release_provider() as provider, asyncio.timeout(75):
        nested = Team([
            specialist(provider, "Rollback reviewer", "ROLLBACK-204", calls),
            specialist(provider, "Data reviewer", "DATA-508", calls),
        ], model=provider, mode="broadcast", strict=True)
        outer = Team([
            nested, specialist(provider, "Release notes", "NOTES-387", calls),
        ], model=provider, mode="coordinate", instructions="Return every evidence label verbatim.")
        result = await outer.arun(PACKET + " Return the three exact evidence labels from the specialists.")
    labels = ("ROLLBACK-204", "DATA-508", "NOTES-387")
    if not all(label in result.text for label in labels):
        evidence = [outcome.result.content for outcome in result.tool_activity if outcome.result]
        pytest.fail(f"Missing evidence label. Handoff: {result.text}\nDelegations: {evidence}\nCalls: {calls}")
    assert calls == {"Rollback reviewer": 1, "Data reviewer": 1, "Release notes": 1}


async def test_release_task_team_completes_its_checklist():
    calls = Counter()
    async with release_provider() as provider, asyncio.timeout(75):
        team = Team([
            specialist(provider, "Database migration", "MIGRATION-816", calls),
        ], model=provider, mode="tasks", max_iterations=3)
        result = await team.arun(
            "Create one todo to obtain the database migration evidence. "
            "Delegate it, mark the todo completed, and report the exact evidence label."
        )
    assert "MIGRATION-816" in result.text
    assert calls == {"Database migration": 1}
    items = team._todo_tools.store.list()
    assert len(items) == 1 and items[0]["status"] == "completed"
