# Feature: agent-ergonomics, Property 13
"""Property 13: Tier routing selects and falls back.

For any configured tier policy, the router SHALL select a configured tier per call;
and when the selected tier is unavailable but a fallback exists, the call SHALL
route to the fallback and a tier substitution SHALL be recorded in RunResult metadata.

**Validates: Requirements 7.1, 7.2, 7.3**
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.agent import Agent, ModelSpec
from loomable.content import ModelCapabilities
from loomable.kernel.errors import ModelProviderError
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.model_router import ModelRouter, TierSubstitution
from loomable.kernel.models import ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: tier names (short identifiers)
tier_name_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N")),
    min_size=1,
    max_size=10,
)

# Strategy: number of tiers (at least 2 so fallback is meaningful)
num_tiers_st = st.integers(min_value=2, max_value=5)

# Strategy: which tier index should be the default in the policy
default_tier_idx_st = st.integers(min_value=0, max_value=100)

# Strategy: response text from the model
response_text_st = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z")),
    min_size=1,
    max_size=30,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProvider:
    """A model provider that returns a fixed response text."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.call_count = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        return ModelResponse(content=self._response_text, tool_calls=[])


class UnavailableProvider:
    """A model provider that always raises ModelProviderError."""

    def __init__(self, provider_id: str = "unavailable") -> None:
        self._provider_id = provider_id
        self.call_count = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.call_count += 1
        raise ModelProviderError(self._provider_id)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestTierSelectionAndFallback:
    """Property 13: Tier routing selects and falls back."""

    @settings(max_examples=100, deadline=None)
    @given(
        tier_names=st.lists(tier_name_st, min_size=2, max_size=5, unique=True),
        default_tier_idx=default_tier_idx_st,
        response_text=response_text_st,
    )
    @pytest.mark.asyncio
    async def test_router_selects_configured_tier(
        self,
        tier_names: list[str],
        default_tier_idx: int,
        response_text: str,
    ) -> None:
        """For any configured tier policy, the router selects a configured tier.
        (Validates Req 7.1)"""
        # Pick a valid default tier index
        default_idx = default_tier_idx % len(tier_names)
        default_tier = tier_names[default_idx]

        # Build tiers dict: each tier maps to a provider placeholder
        tiers = {name: {"provider": name} for name in tier_names}

        # Build a ModelInterface with providers for all tiers
        providers = {name: FakeProvider(f"{response_text}_{name}") for name in tier_names}
        model_interface = ModelInterface(providers=providers, default_provider=tier_names[0])

        # Build the router with a policy that selects the default tier
        tier_policy = {"default_tier": default_tier}
        router = ModelRouter(
            model_interface=model_interface,
            tiers=tiers,
            tier_policy=tier_policy,
            fallback_tiers={},
        )

        # Route a request
        request = ModelRequest(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        )
        selected = router.select_tier(request)

        # The selected tier MUST be one of the configured tiers
        assert selected in tiers, (
            f"Selected tier '{selected}' is not in configured tiers: {list(tiers.keys())}"
        )
        # When policy specifies a default_tier, it should be that one
        assert selected == default_tier

    @settings(max_examples=100, deadline=None)
    @given(
        tier_names=st.lists(tier_name_st, min_size=2, max_size=5, unique=True),
        response_text=response_text_st,
    )
    @pytest.mark.asyncio
    async def test_fallback_used_when_primary_unavailable(
        self,
        tier_names: list[str],
        response_text: str,
    ) -> None:
        """When the selected tier is unavailable but a fallback exists, the call
        routes to the fallback tier. (Validates Req 7.2)"""
        # First tier is the "primary" (will be unavailable), second is the fallback
        primary_tier = tier_names[0]
        fallback_tier = tier_names[1]

        tiers = {name: {"provider": name} for name in tier_names}

        # Primary provider raises ModelProviderError; fallback is available
        providers: dict = {}
        for name in tier_names:
            if name == primary_tier:
                providers[name] = UnavailableProvider(name)
            else:
                providers[name] = FakeProvider(f"{response_text}_{name}")

        model_interface = ModelInterface(providers=providers, default_provider=primary_tier)

        # Configure fallback: primary -> fallback_tier
        fallback_tiers = {primary_tier: fallback_tier}

        # Policy selects the primary tier (which will fail)
        tier_policy = {"default_tier": primary_tier}

        router = ModelRouter(
            model_interface=model_interface,
            tiers=tiers,
            tier_policy=tier_policy,
            fallback_tiers=fallback_tiers,
        )

        request = ModelRequest(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        )
        response, substitution = await router.route(request)

        # The response should come from the fallback provider
        assert response.content == f"{response_text}_{fallback_tier}"
        # A TierSubstitution should be recorded
        assert substitution is not None
        assert substitution.intended_tier == primary_tier
        assert substitution.fallback_tier == fallback_tier

    @settings(max_examples=100, deadline=None)
    @given(
        tier_names=st.lists(tier_name_st, min_size=2, max_size=5, unique=True),
        response_text=response_text_st,
    )
    @pytest.mark.asyncio
    async def test_tier_substitution_recorded_in_run_result_metadata(
        self,
        tier_names: list[str],
        response_text: str,
    ) -> None:
        """A tier substitution is recorded in RunResult metadata when a fallback
        is used. (Validates Req 7.3)"""
        primary_tier = tier_names[0]
        fallback_tier = tier_names[1]

        tiers = {name: {"provider": name} for name in tier_names}

        # Primary provider raises ModelProviderError; fallback is available
        providers: dict = {}
        for name in tier_names:
            if name == primary_tier:
                providers[name] = UnavailableProvider(name)
            else:
                providers[name] = FakeProvider(f"{response_text}_{name}")

        model_interface = ModelInterface(providers=providers, default_provider=primary_tier)

        # Configure fallback: primary -> fallback_tier
        fallback_tiers = {primary_tier: fallback_tier}

        # Build the agent with tiered routing where primary is unavailable
        agent = Agent(
            model=ModelSpec(provider=primary_tier, provider_impl=providers[primary_tier]),
            capabilities=ModelCapabilities(),
            tiers=tiers,
            tier_policy={"default_tier": primary_tier},
            fallback_tiers=fallback_tiers,
        )
        built = agent.build()

        # Inject the model interface with all providers so the router can
        # actually reach the fallback
        built.model_interface = model_interface
        # Reconstruct the router with the correct model_interface
        built.router = ModelRouter(
            model_interface=model_interface,
            tiers=tiers,
            tier_policy={"default_tier": primary_tier},
            fallback_tiers=fallback_tiers,
        )

        result = await built.arun("hello")

        # RunResult.metadata should contain the tier_substitution
        assert "tier_substitution" in result.metadata, (
            f"Expected 'tier_substitution' in metadata, got keys: {list(result.metadata.keys())}"
        )
        sub = result.metadata["tier_substitution"]
        assert isinstance(sub, TierSubstitution)
        assert sub.intended_tier == primary_tier
        assert sub.fallback_tier == fallback_tier

    @settings(max_examples=100, deadline=None)
    @given(
        tier_names=st.lists(tier_name_st, min_size=2, max_size=5, unique=True),
        response_text=response_text_st,
    )
    @pytest.mark.asyncio
    async def test_no_substitution_when_primary_available(
        self,
        tier_names: list[str],
        response_text: str,
    ) -> None:
        """When the primary tier is available, no substitution is recorded and
        the response comes from the primary tier. (Validates Req 7.1)"""
        primary_tier = tier_names[0]
        fallback_tier = tier_names[1]

        tiers = {name: {"provider": name} for name in tier_names}

        # All providers are available
        providers = {name: FakeProvider(f"{response_text}_{name}") for name in tier_names}
        model_interface = ModelInterface(providers=providers, default_provider=primary_tier)

        fallback_tiers = {primary_tier: fallback_tier}
        tier_policy = {"default_tier": primary_tier}

        router = ModelRouter(
            model_interface=model_interface,
            tiers=tiers,
            tier_policy=tier_policy,
            fallback_tiers=fallback_tiers,
        )

        request = ModelRequest(
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        )
        response, substitution = await router.route(request)

        # Response from primary
        assert response.content == f"{response_text}_{primary_tier}"
        # No substitution recorded
        assert substitution is None
