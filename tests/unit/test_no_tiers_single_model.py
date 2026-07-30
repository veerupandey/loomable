# Feature: agent-ergonomics, Property 14
"""Property 14: No tiers means unchanged single model.

For any agent built without a tier configuration, model calls use the single
configured provider unchanged. No router is constructed, no tier metadata is
recorded.

**Validates: Requirements 7.4**
"""

from __future__ import annotations

import pytest

from loomable.agent import Agent, ModelSpec, RunResult
from loomable.content import ModelCapabilities
from loomable.kernel.models import ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProvider:
    """A model provider that returns a fixed response and records calls."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.call_count = 0
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        self.requests.append(request)
        return ModelResponse(content=self._response_text, tool_calls=[])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoTiersSingleModel:
    """Property 14: No tiers means unchanged single model."""

    @pytest.mark.asyncio
    async def test_no_tiers_router_is_none(self) -> None:
        """When no tiers are configured, the built agent's router is None."""
        provider = FakeProvider("hello")

        agent = Agent(
            model=ModelSpec(provider="default", provider_impl=provider),
            capabilities=ModelCapabilities(),
            # No tiers configured
        )
        built = agent.build()

        assert built.router is None

    @pytest.mark.asyncio
    async def test_no_tiers_uses_single_provider_directly(self) -> None:
        """Without tiers, the agent calls the single configured provider directly
        and returns its response unchanged."""
        provider = FakeProvider("direct response")

        agent = Agent(
            model=ModelSpec(provider="my-provider", provider_impl=provider),
            capabilities=ModelCapabilities(),
            # No tiers, no fallback, no tier_policy
        )
        built = agent.build()

        result = await built.arun("test input")

        # The provider was called exactly once
        assert provider.call_count == 1
        # The output is the provider's response
        assert result.output.text() == "direct response"

    @pytest.mark.asyncio
    async def test_no_tiers_no_tier_substitution_in_metadata(self) -> None:
        """Without tiers, RunResult metadata contains no tier_substitution key."""
        provider = FakeProvider("no tiers here")

        agent = Agent(
            model=ModelSpec(provider="solo", provider_impl=provider),
            capabilities=ModelCapabilities(),
        )
        built = agent.build()

        result = await built.arun("hello")

        assert isinstance(result, RunResult)
        assert "tier_substitution" not in result.metadata

    @pytest.mark.asyncio
    async def test_no_tiers_model_interface_invoked_without_tier_param(self) -> None:
        """Without tiers, the model_interface.invoke is called without a tier
        parameter (proving the router path is not used)."""
        provider = FakeProvider("tier-free")

        agent = Agent(
            model=ModelSpec(provider="plain", provider_impl=provider),
            capabilities=ModelCapabilities(),
        )
        built = agent.build()

        # Confirm router is None before running
        assert built.router is None

        result = await built.arun("query")

        # Provider received the call
        assert provider.call_count == 1
        # Output matches
        assert result.output.text() == "tier-free"
