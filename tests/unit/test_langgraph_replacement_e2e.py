"""Tough LangGraph-replacement E2E: incident war-room control plane.

Simulates what teams usually build in LangGraph without a live LLM:

  ingest → parallel gather (reducers + skip) → Command.route triage
        → verify SEV packet → HITL publish → checkpoint resume / fork

Also probes known sharp edges (multi-goto, Command.goto outside route,
nested SharedState + reads=, update_state mid-flight).
"""

from __future__ import annotations

import pytest

from loomable import Command, FlowPaused, Step, StepFailed, Workflow
from loomable.agent.run import RunResult
from loomable.content import AgentOutput, MediaPart, Modality
from loomable.flow.state import extend
from loomable.persist.checkpoint import Checkpoint, InMemoryCheckpointer


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


# ---------------------------------------------------------------------------
# Full war-room pipeline
# ---------------------------------------------------------------------------


class TestIncidentWarRoomE2E:
    @pytest.mark.asyncio
    async def test_full_sev1_path_with_hitl_and_fork(self):
        cp = InMemoryCheckpointer()
        drafts = {"n": 0}

        async def ingest(ticket, *, context=None):
            if context and context.shared_state:
                context.shared_state.write("ticket_id", str(ticket))
            return _text(f"ingested:{ticket}")

        async def logs(ticket, *, context=None):
            return RunResult(
                output=_text("logs:oom").output,
                session_id="",
                metadata={"state_updates": {"evidence": ["logs:oom"]}},
            )

        async def metrics(ticket, *, context=None):
            return RunResult(
                output=_text("metrics:p99").output,
                session_id="",
                metadata={"state_updates": {"evidence": ["metrics:p99"]}},
            )

        async def flaky_vendor(ticket, *, context=None):
            raise RuntimeError("vendor API down")

        async def summarize(ticket, *, context=None):
            evidence = []
            if context and context.shared_state:
                evidence = list(context.shared_state.get("evidence") or [])
                context.shared_state.write(
                    "brief",
                    AgentOutput(
                        parts=[
                            MediaPart(
                                modality=Modality.TEXT,
                                media_type="text/plain",
                                data=(" | ".join(evidence)).encode("utf-8"),
                            )
                        ]
                    ),
                )
            return _text(f"brief:{'|'.join(evidence)}")

        def triage(change, *, context=None):
            # After gather, ambient ``change`` is the brief — not the ticket.
            # Prefer ``_workflow_input`` / ``ticket_id`` (LangGraph-style channels).
            text = str(change).lower()
            if context is not None and context.shared_state is not None:
                orig = context.shared_state.get("_workflow_input")
                if orig is not None:
                    text = str(orig).lower()
                ticket = context.shared_state.get("ticket_id")
                if ticket:
                    text = f"{text} {ticket}".lower()
            if "sev1" in text or "outage" in text:
                return Command(goto="full", update={"severity": "sev1"})
            if "question" in text:
                return Command(goto="human", update={"severity": "unknown"})
            return Command(goto="quick", update={"severity": "sev3"})

        async def quick(change, *, context=None):
            return _text("quick-ack")

        async def full(change, *, context=None):
            sev = context.shared_state.get("severity") if context else "?"
            brief = _as_text(
                context.shared_state.get("brief") if context else None
            )
            return _text(f"full[{sev}]:{brief}")

        async def human(change, *, context=None):
            return _text("needs-human")

        async def draft_sev(packet, *, context=None):
            drafts["n"] += 1
            brief = _as_text(
                context.shared_state.get("brief") if context else None
            )
            sev = context.shared_state.get("severity") if context else "?"
            body = f"SEV-{sev} {brief}"
            if drafts["n"] == 1:
                return _text(f"DRAFT:{body}")  # missing Sources
            return _text(f"PACKET:{body}\nSources checked.")

        def packet_ok(output, ctx) -> bool:
            return "Sources checked" in output.text()

        async def publish(packet, *, context=None):
            return _text(f"PUBLISHED:{_as_text(packet)}")

        gather = (
            Workflow("gather", reducers={"evidence": extend})
            .parallel(
                Step("logs", logs),
                Step("metrics", metrics),
                Step("vendor", flaky_vendor, on_failure="skip"),
            )
            .step("summarize", summarize)
        )

        wf = (
            Workflow(
                "war_room",
                session_id="inc-88421",
                checkpointer=cp,
                reducers={"evidence": extend},
            )
            .step("ingest", ingest)
            .step("gather", gather)
            .route(triage, quick=quick, full=full, human=human)
            .verify(
                Step("draft", draft_sev, reads="brief"),
                check=packet_ok,
                max_retries=2,
            )
            .step("publish", publish, confirm=True)
        )

        # --- Hit HITL pause ---
        with pytest.raises(FlowPaused) as paused:
            await wf.arun("SEV1 database outage")
        assert paused.value.node_id == "publish"

        mid = await wf.get_state()
        assert mid["complete"] is False
        assert mid["values"].get("severity") == "sev1"
        assert set(mid["values"].get("evidence") or []) == {
            "logs:oom",
            "metrics:p99",
        }
        assert "vendor" not in (mid["values"].get("evidence") or [])
        assert drafts["n"] == 2  # one repair round

        # Human edits severity note then approves
        await wf.update_state({"operator_note": "customer-facing"})
        await wf.approve("publish")
        result = await wf.arun(resume=True)
        assert "PUBLISHED:PACKET:" in result.output.text()
        assert "Sources checked" in result.output.text()

        final = await wf.get_state()
        assert final["complete"] is True
        assert final["values"].get("operator_note") == "customer-facing"
        assert drafts["n"] == 2  # verify not re-run on resume

        # Time-travel fork for an alternate operator path
        forked = await wf.fork_session("inc-88421-alt")
        assert forked["thread_id"] == "inc-88421-alt"
        assert wf._session_id == "inc-88421-alt"
        alt_cp = await cp.get("inc-88421-alt")
        assert alt_cp is not None
        src_cp = await cp.get("inc-88421")
        assert src_cp is not None

    @pytest.mark.asyncio
    async def test_sev3_skips_full_and_still_publishes(self):
        cp = InMemoryCheckpointer()

        def triage(change):
            return Command(goto="quick", update={"severity": "sev3"})

        async def quick(change, *, context=None):
            if context and context.shared_state:
                context.shared_state.write(
                    "brief",
                    AgentOutput(
                        parts=[
                            MediaPart(
                                modality=Modality.TEXT,
                                media_type="text/plain",
                                data=b"minor blip",
                            )
                        ]
                    ),
                )
            return _text("quick-ack")

        async def full(change, *, context=None):
            return _text("SHOULD_NOT_RUN")

        async def draft(packet, *, context=None):
            return _text("PACKET:minor\nSources checked.")

        async def publish(packet, *, context=None):
            return _text(f"OUT:{_as_text(packet)}")

        wf = (
            Workflow("triage_low", session_id="t-low", checkpointer=cp)
            .route(triage, quick=quick, full=full)
            .verify(Step("draft", draft, reads="brief"), check=lambda o, c: True)
            .step("publish", publish, confirm=True)
        )

        with pytest.raises(FlowPaused):
            await wf.arun("sev3 latency blip")

        # Unselected "full" arm must not be marked completed
        st = await wf.get_state()
        completed = st["completed"]
        assert not any(c.endswith("_full") or c == "full" for c in completed)

        await wf.approve("publish")
        result = await wf.arun(resume=True)
        assert "OUT:PACKET:minor" in result.output.text()
        assert "SHOULD_NOT_RUN" not in result.output.text()


# ---------------------------------------------------------------------------
# Sharp-edge probes (expect honest failures / document behavior)
# ---------------------------------------------------------------------------


    @pytest.mark.asyncio
    async def test_route_after_pipeline_needs_workflow_input(self):
        """Ambient route input is the previous step — use ``_workflow_input``."""

        async def gather(inp, *, context=None):  # noqa: A002
            return _text("brief-only-no-sev-keyword")

        seen: dict[str, str] = {}

        def triage(change, *, context=None):
            ambient = str(change)
            orig = ""
            if context and context.shared_state:
                orig = str(context.shared_state.get("_workflow_input") or "")
            seen["ambient"] = ambient
            seen["orig"] = orig
            text = orig.lower() or ambient.lower()
            if "sev1" in text:
                return Command(goto="full", update={"severity": "sev1"})
            return Command(goto="quick", update={"severity": "sev3"})

        async def quick(change, *, context=None):
            return _text("Q")

        async def full(change, *, context=None):
            return _text("F")

        wf = (
            Workflow("r")
            .step("gather", gather)
            .route(triage, quick=quick, full=full)
        )
        result = await wf.arun("SEV1 outage")
        assert "brief-only" in seen["ambient"]
        assert "SEV1" in seen["orig"]
        assert "F" in result.output.text()
        assert wf.state.get("severity") == "sev1"
        assert wf.state.get("_workflow_input") == "SEV1 outage"


class TestSharpEdges:
    @pytest.mark.asyncio
    async def test_multi_goto_list_selects_multiple_route_arms(self):
        """LangGraph multi-Send-ish: Command(goto=[a,b]) should fan out."""
        ran = {"a": 0, "b": 0, "c": 0}

        def choose(inp):  # noqa: A002
            return Command(goto=["a", "b"], update={"fan": True})

        async def a(inp, *, context=None):  # noqa: A002
            ran["a"] += 1
            return _text("A")

        async def b(inp, *, context=None):  # noqa: A002
            ran["b"] += 1
            return _text("B")

        async def c(inp, *, context=None):  # noqa: A002
            ran["c"] += 1
            return _text("C")

        wf = Workflow("multi").route(choose, a=a, b=b, c=c)
        result = await wf.arun("x")
        # Both a and b should run; c must not
        assert ran["a"] == 1
        assert ran["b"] == 1
        assert ran["c"] == 0
        # Join output should reflect executed arms (order may vary)
        text = result.output.text()
        assert "A" in text or "B" in text

    @pytest.mark.asyncio
    async def test_command_goto_outside_route_does_not_skip_ahead(self):
        """Docstring claims sequential skip-ahead; verify actual behavior."""
        order: list[str] = []

        async def a(inp, *, context=None):  # noqa: A002
            order.append("a")
            return Command(goto="c", update={"jumped": True})

        async def b(inp, *, context=None):  # noqa: A002
            order.append("b")
            return _text("B")

        async def c(inp, *, context=None):  # noqa: A002
            order.append("c")
            jumped = context.shared_state.get("jumped") if context else None
            return _text(f"C:{jumped}")

        wf = Workflow("skip").step("a", a).step("b", b).step("c", c)
        result = await wf.arun("x")
        # If skip-ahead worked: order == ["a","c"]. Today engines ignore goto.
        assert order == ["a", "b", "c"]
        assert "C:True" in result.output.text()  # update still applied

    @pytest.mark.asyncio
    async def test_nested_reads_sees_parent_key(self):
        async def writer(inp, *, context=None):  # noqa: A002
            if context and context.shared_state:
                context.shared_state.write(
                    "evidence",
                    AgentOutput(
                        parts=[
                            MediaPart(
                                modality=Modality.TEXT,
                                media_type="text/plain",
                                data=b"parent-evidence",
                            )
                        ]
                    ),
                )
            return _text("wrote")

        async def reader(evidence, *, context=None):
            return _text(f"read:{_as_text(evidence)}")

        inner = Workflow("inner").step(
            Step("reader", reader, reads="evidence")
        )
        wf = Workflow("outer").step("writer", writer).step("inner", inner)
        result = await wf.arun("x")
        assert "read:parent-evidence" in result.output.text()

    @pytest.mark.asyncio
    async def test_hitl_rejected_inside_route(self):
        def choose(inp):  # noqa: A002
            return "arm"

        async def arm(inp, *, context=None):  # noqa: A002
            return _text("ARM")

        with pytest.raises(Exception) as exc:
            Workflow("bad").route(
                choose,
                arm=Step("arm", arm, confirm=True),
            ).build()
        assert "confirm" in str(exc.value).lower() or "hitl" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_parallel_stop_preserves_sibling_evidence(self):
        async def good(inp, *, context=None):  # noqa: A002
            return RunResult(
                output=_text("GOOD").output,
                session_id="",
                metadata={"state_updates": {"evidence": ["good"]}},
            )

        async def bad(inp, *, context=None):  # noqa: A002
            raise RuntimeError("boom")

        wf = Workflow("stop", reducers={"evidence": extend}).parallel(
            Step("good", good),
            Step("bad", bad, on_failure="stop"),
        )
        with pytest.raises(StepFailed):
            await wf.arun("x")
        assert wf.state.get("evidence") == ["good"]
        assert _as_text(wf.state.get("good")) == "GOOD"

    @pytest.mark.asyncio
    async def test_kill_resume_after_gather_before_route(self):
        """Simulate process death after nested gather, resume into route."""
        cp = InMemoryCheckpointer()

        async def gather_step(inp, *, context=None):  # noqa: A002
            if context and context.shared_state:
                context.shared_state.write("severity_hint", "sev1")
                context.shared_state.write(
                    "brief",
                    AgentOutput(
                        parts=[
                            MediaPart(
                                modality=Modality.TEXT,
                                media_type="text/plain",
                                data=b"precomputed",
                            )
                        ]
                    ),
                )
            return _text("gathered")

        def triage(change):
            return Command(goto="full", update={"severity": "sev1"})

        async def quick(change, *, context=None):
            return _text("Q")

        async def full(change, *, context=None):
            brief = _as_text(context.shared_state.get("brief") if context else None)
            return _text(f"FULL:{brief}")

        wf = (
            Workflow("kill", session_id="k1", checkpointer=cp)
            .step("gather", gather_step)
            .route(triage, quick=quick, full=full)
        )

        # Manually seed checkpoint as if gather completed and process died
        await cp.put(
            Checkpoint(
                thread_id="k1",
                step=1,
                session_state={
                    "shared_state": {
                        "gather": {
                            "__type__": "AgentOutput",
                            "parts": [
                                {
                                    "__type__": "MediaPart",
                                    "modality": "text",
                                    "media_type": "text/plain",
                                    "data_b64": "Z2F0aGVyZWQ=",  # gathered
                                    "uri": None,
                                }
                            ],
                        },
                        "severity_hint": "sev1",
                        "brief": {
                            "__type__": "AgentOutput",
                            "parts": [
                                {
                                    "__type__": "MediaPart",
                                    "modality": "text",
                                    "media_type": "text/plain",
                                    "data_b64": "cHJlY29tcHV0ZWQ=",  # precomputed
                                    "uri": None,
                                }
                            ],
                        },
                    },
                    "completed_node_ids": ["gather"],
                },
                complete=False,
            )
        )

        result = await wf.arun("SEV1 outage", resume=True)
        assert "FULL:precomputed" in result.output.text()
        st = await wf.get_state()
        assert st["complete"] is True
        assert "gather" in st["completed"]
