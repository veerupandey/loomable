"""Unit tests for the high-level Agent builder (task 3.1).

Verify that:
- A minimal config (model only) builds a runnable BuiltAgent with non-null subsystems.
- Supplied low-level overrides are used verbatim instead of constructed defaults.
- A missing/invalid model raises AgentConfigError naming the field.
- Effective capabilities resolve from arg > ModelSpec > multimodal default.
"""

from __future__ import annotations

import pytest

from loomable.agent import Agent, BuiltAgent, ModelSpec
from loomable.agent.errors import AgentConfigError
from loomable.content import Modality, ModelCapabilities
from loomable.kernel.context import ContextManager
from loomable.kernel.guardrails import GuardrailHarness
from loomable.kernel.memory import MemoryManager
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.models import ModelRequest, ModelResponse, Session
from loomable.kernel.planner import Planner
from loomable.kernel.stores import SessionStore
from loomable.kernel.tool_runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ModelProvider implementation (satisfies the structural protocol)."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


# ---------------------------------------------------------------------------
# Minimal config → runnable BuiltAgent (Req 1.1, 1.2)
# ---------------------------------------------------------------------------


class TestBuilderDefaults:
    def test_minimal_config_builds_runnable_agent(self):
        agent = Agent(model=_FakeProvider())
        built = agent.build()

        assert isinstance(built, BuiltAgent)
        assert built.loop is None
        assert isinstance(built.model_interface, ModelInterface)
        assert isinstance(built.memory, MemoryManager)
        assert isinstance(built.tool_runtime, ToolRuntime)
        assert isinstance(built.session, Session)
        assert isinstance(built.capabilities, ModelCapabilities)
        # Subsystems retained for later tasks are also constructed.
        assert isinstance(built.harness, GuardrailHarness)
        assert built.planner is None
        assert isinstance(built.session_store, SessionStore)

    def test_bare_provider_registered_under_default(self):
        provider = _FakeProvider()
        built = Agent(model=provider).build()

        assert built.model_interface.default_provider == "default"
        assert built.model_interface.providers["default"] is provider

    def test_modelspec_uses_its_provider_id_and_impl(self):
        provider = _FakeProvider()
        spec = ModelSpec(provider="acme", provider_impl=provider)
        built = Agent(model=spec).build()

        assert built.model_interface.default_provider == "acme"
        assert built.model_interface.providers["acme"] is provider

    def test_session_id_used_when_supplied(self):
        built = Agent(model=_FakeProvider(), session_id="sess-123").build()
        assert built.session.session_id == "sess-123"

    def test_new_session_created_when_no_session_id(self):
        built = Agent(model=_FakeProvider()).build()
        assert built.session.session_id.startswith("session-")


# ---------------------------------------------------------------------------
# Overrides win over defaults (Req 2.2, 2.3)
# ---------------------------------------------------------------------------


class TestBuilderOverrides:
    def test_supplied_primitives_are_used(self):
        context_manager = ContextManager(1234)
        memory = MemoryManager()
        tool_runtime = ToolRuntime({})
        harness = GuardrailHarness([])
        session_store = SessionStore()

        built = Agent(
            model=_FakeProvider(),
            context_manager=context_manager,
            kernel_memory=memory,
            tool_runtime=tool_runtime,
            harness=harness,
            session_store=session_store,
        ).build()

        assert built.memory is memory
        assert built.tool_runtime is tool_runtime
        assert built.harness is harness
        assert built.session_store is session_store

    def test_supplied_planner_is_used(self):
        provider = _FakeProvider()
        mi = ModelInterface(providers={"p": provider}, default_provider="p")
        planner = Planner(mi)

        built = Agent(model=_FakeProvider(), planner=planner).build()
        assert built.planner is planner


# ---------------------------------------------------------------------------
# Capabilities resolution (Req 6.2)
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_default_capabilities_are_multimodal(self):
        built = Agent(model=_FakeProvider()).build()
        assert built.capabilities.input == frozenset(
            {Modality.TEXT, Modality.IMAGE, Modality.VIDEO}
        )
        assert built.capabilities.output == frozenset({Modality.TEXT})
        assert built.max_tool_iterations == 12
        assert built.require_final_text is True

    def test_text_only_and_modalities_strings(self):
        text = Agent(model=_FakeProvider(), text_only=True).build()
        assert text.capabilities.input == frozenset({Modality.TEXT})
        mixed = Agent(model=_FakeProvider(), modalities="text+image").build()
        assert Modality.IMAGE in mixed.capabilities.input
        assert Modality.VIDEO not in mixed.capabilities.input

    def test_explicit_capabilities_arg_wins(self):
        caps = ModelCapabilities(
            input=frozenset({Modality.TEXT, Modality.IMAGE}),
            output=frozenset({Modality.TEXT}),
        )
        built = Agent(model=_FakeProvider(), capabilities=caps).build()
        assert built.capabilities is caps

    def test_modelspec_capabilities_used_when_no_arg(self):
        caps = ModelCapabilities(input=frozenset({Modality.VIDEO}))
        spec = ModelSpec(provider="acme", provider_impl=_FakeProvider(), capabilities=caps)
        built = Agent(model=spec).build()
        assert built.capabilities is caps


# ---------------------------------------------------------------------------
# Validation (Req 1.6)
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_model_raises_in_constructor(self):
        with pytest.raises(AgentConfigError) as exc:
            Agent(model=None)
        assert exc.value.field == "model"

    def test_invalid_token_budget_raises_on_build(self):
        agent = Agent(model=_FakeProvider(), token_budget=0)
        with pytest.raises(AgentConfigError) as exc:
            agent.build()
        assert exc.value.field == "token_budget"

    def test_invalid_checkpoint_interval_raises_on_build(self):
        agent = Agent(model=_FakeProvider(), checkpoint_interval=0)
        with pytest.raises(AgentConfigError) as exc:
            agent.build()
        assert exc.value.field == "checkpoint_interval"

    def test_empty_modelspec_provider_raises(self):
        agent = Agent(model=ModelSpec(provider="", provider_impl=_FakeProvider()))
        with pytest.raises(AgentConfigError) as exc:
            agent.build()
        assert exc.value.field == "model"


# ---------------------------------------------------------------------------
# Run flow implemented by task 3.2 (see test_built_agent_run.py for coverage)
# ---------------------------------------------------------------------------


class TestRunFlow:
    async def test_arun_returns_run_result(self):
        built = Agent(model=_FakeProvider()).build()
        result = await built.arun("hi")
        assert result.output.text() == "ok"
        assert result.session_id == built.session.session_id
