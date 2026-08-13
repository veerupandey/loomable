"""Unit tests for ToughTask / plan_act_verify / SharedState plan glue."""

from __future__ import annotations

import pytest

from loomable import Agent, ToughTask, plan_act_verify
from loomable.agent import ModelSpec
from loomable.tough import parse_plan_steps
from loomable.flow.loop import VerdictResult
from loomable.kernel.models import ModelRequest, ModelResponse


class _Scripted:
    """Deterministic provider driven by call count + prompt keywords."""

    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        text_parts: list[str] = []
        for msg in request.messages:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        text_parts.append(str(p.get("text", "")))
            elif isinstance(content, str):
                text_parts.append(content)
        blob = "\n".join(text_parts).lower()

        if "json array" in blob or "break the user task" in blob or "planner" in blob:
            return ModelResponse(
                content='["Gather ticket facts", "Assess SLA risk", "Draft SEV packet"]',
                usage={"input_tokens": 5, "output_tokens": 10},
            )
        if "verification failed" in blob or "failed verification" in blob:
            return ModelResponse(
                content="FINAL SEV-1 packet with evidence and next actions.",
                usage={"input_tokens": 5, "output_tokens": 8},
            )
        if "integrate" in blob or "synthesizer" in blob or "specialist/worker" in blob:
            # First synthesis may omit SEV- to exercise verify loop.
            if self.n < 6:
                return ModelResponse(
                    content="Draft packet without severity label yet.",
                    usage={"input_tokens": 5, "output_tokens": 6},
                )
            return ModelResponse(
                content="FINAL SEV-1 packet ready for war room.",
                usage={"input_tokens": 5, "output_tokens": 8},
            )
        # Worker / specialist steps
        return ModelResponse(
            content=f"done-step-{self.n}",
            usage={"input_tokens": 3, "output_tokens": 4},
        )


def _model() -> ModelSpec:
    return ModelSpec(provider="scripted", provider_impl=_Scripted())


def test_parse_plan_steps_json_and_bullets() -> None:
    assert parse_plan_steps('["A", "B"]') == ["A", "B"]
    assert parse_plan_steps("- one\n- two") == ["one", "two"]


@pytest.mark.asyncio
async def test_function_runnable_dict_writes_state_updates() -> None:
    from loomable.flow.runnable import FunctionRunnable

    async def planner(inp):
        return {"plan_steps": ["a", "b"]}

    result = await FunctionRunnable(planner).arun("task")
    assert result.metadata["state_updates"]["plan_steps"] == ["a", "b"]
    assert result.structured["plan_steps"] == ["a", "b"]


@pytest.mark.asyncio
async def test_plan_and_execute_shared_state_glue() -> None:
    from loomable.flow.helpers import plan_and_execute

    async def planner(inp):
        return {"plan_steps": ["step-one", "step-two"]}

    async def worker(inp):
        return f"worked:{inp}"

    async def synthesizer(inp, *, context=None):
        pieces = []
        if context is not None and context.shared_state is not None:
            pieces = context.shared_state.get("map") or []
        return " | ".join(pieces)

    flow = plan_and_execute(planner, worker, synthesizer)
    result = await flow.arun("do it")
    assert "worked:step-one" in result.output.text()
    assert "worked:step-two" in result.output.text()


def _sev_verifier(output, context) -> VerdictResult:
    ok = "SEV-" in (output.text() or "")
    return VerdictResult(ok=ok, detail="missing SEV-" if not ok else "")


@pytest.mark.asyncio
async def test_tough_task_map_with_verify_loop() -> None:
    task = ToughTask(
        model=_model(),
        fan_out="map",
        verify=_sev_verifier,
        max_iterations=4,
        max_steps=3,
    )
    result = await task.arun("Handle INC-88421 for BharatNova")
    assert result.metadata.get("tough") is True
    assert "SEV-" in result.output.text()
    assert result.metadata.get("loop_verified") is True


@pytest.mark.asyncio
async def test_agent_mode_tough() -> None:
    agent = Agent(
        model=_model(),
        mode="tough",
        fan_out="map",
        verifier=_sev_verifier,
        max_verify_retries=3,
        max_plan_steps=3,
        modalities="text",
    )
    result = await agent.arun("Escalate INC-88421")
    assert result.metadata.get("tough") is True
    assert "SEV-" in result.output.text()


@pytest.mark.asyncio
async def test_plan_act_verify_returns_workflow() -> None:
    wf = plan_act_verify(model=_model(), fan_out="map", max_steps=2)
    assert wf.name == "tough"
    result = await wf.arun("plan a caching strategy")
    assert result.output.text()
