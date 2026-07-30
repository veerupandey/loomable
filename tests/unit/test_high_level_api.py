"""Tests for high-level API improvements: model strings, name, debug, response_model, etc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from loomable.agent import Agent, ModelSpec, AgentConfigError
from loomable.kernel.models import ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeProvider:
    """Minimal provider for testing."""
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content='{"city": "Paris", "pop": 2161000}', usage={"input_tokens": 5, "output_tokens": 10})


# ---------------------------------------------------------------------------
# Model string shorthand
# ---------------------------------------------------------------------------


class TestModelStringShorthand:
    """Test that Agent(model="provider:model") resolves correctly."""

    def test_openai_string(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        agent = Agent(model="openai:gpt-4o-mini")
        assert agent._model.provider == "openai"
        assert agent._model.provider_impl.model == "gpt-4o-mini"

    def test_groq_string(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "test-key")
        agent = Agent(model="groq:llama-3.3-70b-versatile")
        assert agent._model.provider == "groq"
        assert agent._model.provider_impl.model == "llama-3.3-70b-versatile"

    def test_ollama_string(self):
        agent = Agent(model="ollama:mistral")
        assert agent._model.provider == "ollama"
        assert agent._model.provider_impl.model == "mistral"

    def test_gemini_string(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        agent = Agent(model="gemini:gemini-2.0-flash")
        assert agent._model.provider == "gemini"
        assert agent._model.provider_impl.model == "gemini-2.0-flash"

    def test_anthropic_string(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        agent = Agent(model="anthropic:claude-sonnet-4-20250514")
        assert agent._model.provider == "anthropic"
        assert agent._model.provider_impl.model == "claude-sonnet-4-20250514"

    def test_bare_model_name_defaults_to_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        agent = Agent(model="gpt-4o-mini")
        assert agent._model.provider == "openai"
        assert agent._model.provider_impl.model == "gpt-4o-mini"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            Agent(model="foobar:some-model")

    def test_modelspec_still_works(self):
        """Existing ModelSpec path should be unchanged."""
        provider = FakeProvider()
        agent = Agent(model=ModelSpec(provider="test", provider_impl=provider))
        assert agent._model.provider_impl is provider

    def test_bare_provider_still_works(self):
        """Passing a raw provider instance should still work."""
        provider = FakeProvider()
        agent = Agent(model=provider)
        assert agent._model is provider


# ---------------------------------------------------------------------------
# Agent name and description
# ---------------------------------------------------------------------------


class TestAgentNameDescription:
    """Test name and description on Agent."""

    def test_name_stored(self):
        agent = Agent(model=ModelSpec(provider="t", provider_impl=FakeProvider()), name="researcher")
        assert agent._name == "researcher"

    def test_description_stored(self):
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=FakeProvider()),
            description="Finds relevant papers",
        )
        assert agent._description == "Finds relevant papers"

    def test_name_on_built_agent(self):
        agent = Agent(model=ModelSpec(provider="t", provider_impl=FakeProvider()), name="analyst")
        built = agent.build()
        assert built.name == "analyst"

    def test_defaults_empty(self):
        agent = Agent(model=ModelSpec(provider="t", provider_impl=FakeProvider()))
        built = agent.build()
        assert built.name is None
        assert built.description is None


# ---------------------------------------------------------------------------
# response_model
# ---------------------------------------------------------------------------


class TestResponseModel:
    """Test response_model as default output_schema."""

    @dataclass
    class CityInfo:
        city: str
        pop: int

    async def test_response_model_used_as_default(self):
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=FakeProvider()),
            response_model=self.CityInfo,
        )
        result = await agent.arun("info about paris")
        # structured should be populated via response_model
        assert result.structured is not None
        assert result.structured.city == "Paris"
        assert result.structured.pop == 2161000

    async def test_output_schema_overrides_response_model(self):
        """Per-call output_schema should take precedence."""
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=FakeProvider()),
            response_model=dict,  # use dict as the default
        )
        result = await agent.arun("test", output_schema=self.CityInfo)
        assert isinstance(result.structured, self.CityInfo)


# ---------------------------------------------------------------------------
# Harness knobs on Agent constructor
# ---------------------------------------------------------------------------


class TestHarnessKnobs:
    """Test tool_timeout and tool_concurrency on Agent()."""

    def test_tool_timeout_on_built(self):
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=FakeProvider()),
            tool_timeout=5.0,
        )
        built = agent.build()
        assert built.tool_timeout == 5.0

    def test_tool_concurrency_on_built(self):
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=FakeProvider()),
            tool_concurrency=3,
        )
        built = agent.build()
        assert built.tool_concurrency == 3

    def test_defaults_are_none(self):
        agent = Agent(model=ModelSpec(provider="t", provider_impl=FakeProvider()))
        built = agent.build()
        assert built.tool_timeout is None
        assert built.tool_concurrency is None


# ---------------------------------------------------------------------------
# Debug mode
# ---------------------------------------------------------------------------


class TestDebugMode:
    """Test debug=True wires a tracer."""

    def test_debug_creates_tracer(self):
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=FakeProvider()),
            debug=True,
        )
        built = agent.build()
        # Should have a JSONTracer (not NoOpEvents)
        from loomable.agent.events import JSONTracer
        assert isinstance(built.events, JSONTracer)

    def test_no_debug_uses_noop(self):
        agent = Agent(model=ModelSpec(provider="t", provider_impl=FakeProvider()))
        built = agent.build()
        from loomable.agent.events import NoOpEvents
        assert isinstance(built.events, NoOpEvents)


# ---------------------------------------------------------------------------
# Lifecycle callbacks
# ---------------------------------------------------------------------------


class TestLifecycleCallbacks:
    """Test on_tool_call and on_complete callbacks."""

    async def test_on_complete_called(self):
        results_seen = []
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=FakeProvider()),
            on_complete=lambda r: results_seen.append(r.output.text()),
        )
        await agent.arun("test")
        assert len(results_seen) == 1

    def test_on_tool_call_wired_as_hook(self):
        calls_seen = []
        agent = Agent(
            model=ModelSpec(provider="t", provider_impl=FakeProvider()),
            on_tool_call=lambda name, args: calls_seen.append(name),
        )
        built = agent.build()
        assert len(built.tool_hooks) >= 1


# ---------------------------------------------------------------------------
# Context parameter
# ---------------------------------------------------------------------------


class TestRuntimeContext:
    """Test that context kwarg is accepted (plumbing for future use)."""

    async def test_context_accepted(self):
        """run/arun should accept context without error."""
        agent = Agent(model=ModelSpec(provider="t", provider_impl=FakeProvider()))
        result = await agent.arun("hi", context={"user_id": "u123"})
        assert result.output.text()  # should still produce output


# ---------------------------------------------------------------------------
# Sub-agents (no Team class)
# ---------------------------------------------------------------------------


class TestSubAgents:
    """Test sub_agents have been removed — orchestration is now via Flow."""

    def test_sub_agents_param_removed(self):
        """Verify that sub_agents is no longer a valid Agent parameter (Req 14.4)."""
        import pytest

        with pytest.raises(TypeError, match="sub_agents"):
            Agent(
                model=ModelSpec(provider="t", provider_impl=FakeProvider()),
                sub_agents=[],
                name="parent",
            )
