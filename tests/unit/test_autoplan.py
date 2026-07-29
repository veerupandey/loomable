"""Unit tests for autonomous PLAN mode (loomable.agent.autoplan).

With OrchestrationMode.PLAN a single agent decomposes a task into steps, runs each
step as a concurrent internal subagent, and synthesizes — all through one agent, so
the agent's own session/memory captures the task and final answer.
"""

from __future__ import annotations

import json

from loomable.agent import Agent, ModelSpec, OrchestrationMode
from loomable.agent.autoplan import _parse_steps
from loomable.kernel.models import ModelRequest, ModelResponse


class ScriptedProvider:
    """A provider that returns a plan (JSON array) first, then per-call replies.

    The first call (the planning request) returns a JSON array of steps. Every
    subsequent call echoes the last user text so step/synthesis outputs are
    identifiable and deterministic.
    """

    def __init__(self, steps: list[str]) -> None:
        self._plan = json.dumps(steps)
        self.calls = 0
        self.seen_user_texts: list[str] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        # Extract the last user text part.
        user_text = ""
        for message in request.messages:
            if message["role"] == "user":
                for part in message["content"]:
                    if part.get("type") == "text":
                        user_text = part["text"]
        self.seen_user_texts.append(user_text)

        if self.calls == 1:
            # Planning call → return the JSON plan.
            return ModelResponse(content=self._plan, usage={"output_tokens": 1})
        # Step / synthesis calls → echo a short marker of the user text.
        return ModelResponse(content=f"done:{user_text[:20]}", usage={"output_tokens": 1})


# ---------------------------------------------------------------------------
# _parse_steps
# ---------------------------------------------------------------------------


class TestParseSteps:
    def test_parses_json_array(self):
        assert _parse_steps('["a", "b", "c"]', 5) == ["a", "b", "c"]

    def test_parses_json_in_code_fence(self):
        assert _parse_steps('```json\n["x", "y"]\n```', 5) == ["x", "y"]

    def test_falls_back_to_numbered_lines(self):
        text = "1. First thing\n2. Second thing\n3. Third thing"
        assert _parse_steps(text, 5) == ["First thing", "Second thing", "Third thing"]

    def test_falls_back_to_bullets_and_skips_headings(self):
        text = "### Plan\n- do a\n- do b\nObjective:\n"
        assert _parse_steps(text, 5) == ["do a", "do b"]

    def test_caps_to_max_steps(self):
        assert _parse_steps('["a","b","c","d"]', 2) == ["a", "b"]


# ---------------------------------------------------------------------------
# PLAN mode end-to-end (with a scripted provider)
# ---------------------------------------------------------------------------


class TestPlanMode:
    async def test_plan_mode_decomposes_and_synthesizes(self):
        provider = ScriptedProvider(steps=["Research X", "Draft Y", "Review Z"])
        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            mode=OrchestrationMode.PLAN,
            max_plan_steps=5,
        )

        result = await agent.arun("Build a thing")

        # One planning call + one call per step + one synthesis call = 1 + 3 + 1.
        assert provider.calls == 5
        # Each step ran as an internal subagent, keyed by step id.
        assert set(result.sub_results) == {"step-1", "step-2", "step-3"}
        # The final answer is the synthesized output.
        assert result.output.text().startswith("done:")

    async def test_plan_mode_respects_max_steps(self):
        provider = ScriptedProvider(steps=["a", "b", "c", "d", "e"])
        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            mode=OrchestrationMode.PLAN,
            max_plan_steps=2,
        )

        result = await agent.arun("do work")
        # Only 2 steps executed (capped) → 1 plan + 2 steps + 1 synth = 4 calls.
        assert provider.calls == 4
        assert set(result.sub_results) == {"step-1", "step-2"}

    async def test_plan_mode_persists_to_single_agent_memory(self):
        """The one PLAN agent records the task + final answer in its own session."""
        provider = ScriptedProvider(steps=["one", "two"])
        agent = Agent(
            model=ModelSpec(provider="scripted", provider_impl=provider),
            mode=OrchestrationMode.PLAN,
            session_id="plan-session",
        )

        built = agent._get_built()
        await agent.arun("centralized memory task")

        # Exactly one user+assistant turn pair recorded on THIS agent's session,
        # regardless of how many internal subagents ran.
        assert len(built.session.l1) == 2
        assert built.session.l1[0].role == "user"
        assert built.session.l1[0].content == "centralized memory task"
        assert built.session.l1[1].role == "assistant"
        assert built.session.step == 1
