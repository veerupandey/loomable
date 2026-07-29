"""Unit tests for the ModelRouter class.

Validates:
- Tier selection per configured cost/latency policy (Req 17.1)
- Routing through the Model Interface (Req 17.2)
- Fallback on unavailable tier with TierSubstitution record (Req 17.3)
"""

from __future__ import annotations

import pytest

from loomable.kernel.errors import ModelProviderError
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.model_router import ModelRouter, TierSubstitution
from loomable.kernel.models import ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Fake providers for testing
# ---------------------------------------------------------------------------


class FakeProvider:
    """A fake model provider that echoes a tag."""

    def __init__(self, tag: str) -> None:
        self._tag = tag

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=f"{self._tag}: ok",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


class FailingProvider:
    """A provider that always raises ModelProviderError."""

    def __init__(self, provider_id: str) -> None:
        self._provider_id = provider_id

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise ModelProviderError(self._provider_id)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_router() -> ModelRouter:
    """Router with two tiers: 'fast' (default) and 'quality', with fallback."""
    providers = {
        "fast": FakeProvider("fast"),
        "quality": FakeProvider("quality"),
    }
    interface = ModelInterface(providers=providers, default_provider="fast")
    return ModelRouter(
        model_interface=interface,
        tiers={"fast": {"provider": "fast"}, "quality": {"provider": "quality"}},
        tier_policy={"default_tier": "fast"},
        fallback_tiers={"fast": "quality", "quality": "fast"},
    )


@pytest.fixture
def router_with_failing_primary() -> ModelRouter:
    """Router where the 'fast' tier is unavailable but 'quality' fallback works."""
    providers = {
        "fast": FailingProvider("fast"),
        "quality": FakeProvider("quality"),
    }
    interface = ModelInterface(providers=providers, default_provider="fast")
    return ModelRouter(
        model_interface=interface,
        tiers={"fast": {"provider": "fast"}, "quality": {"provider": "quality"}},
        tier_policy={"default_tier": "fast"},
        fallback_tiers={"fast": "quality"},
    )


# ---------------------------------------------------------------------------
# Tests for select_tier
# ---------------------------------------------------------------------------


class TestSelectTier:
    """Tests for ModelRouter.select_tier()."""

    def test_selects_default_tier_from_policy(self, basic_router: ModelRouter) -> None:
        """select_tier returns the policy default_tier when configured."""
        request = ModelRequest(messages=[{"role": "user", "content": "hi"}])
        tier = basic_router.select_tier(request)
        assert tier == "fast"

    def test_selects_tier_from_request_metadata(self) -> None:
        """select_tier respects a tier hint in request metadata."""
        providers = {"fast": FakeProvider("fast"), "quality": FakeProvider("quality")}
        interface = ModelInterface(providers=providers, default_provider="fast")
        router = ModelRouter(
            model_interface=interface,
            tiers={"fast": {}, "quality": {}},
            tier_policy=None,  # no default_tier in policy
            fallback_tiers={},
        )
        request = ModelRequest(
            messages=[{"role": "user", "content": "hi"}],
            metadata={"tier": "quality"},
        )
        tier = router.select_tier(request)
        assert tier == "quality"

    def test_falls_back_to_first_tier_when_no_policy(self) -> None:
        """select_tier returns first configured tier when no policy default."""
        providers = {"alpha": FakeProvider("alpha"), "beta": FakeProvider("beta")}
        interface = ModelInterface(providers=providers, default_provider="alpha")
        router = ModelRouter(
            model_interface=interface,
            tiers={"alpha": {}, "beta": {}},
            tier_policy=None,
            fallback_tiers={},
        )
        request = ModelRequest(messages=[{"role": "user", "content": "hi"}])
        tier = router.select_tier(request)
        assert tier == "alpha"

    def test_raises_value_error_when_no_tiers(self) -> None:
        """select_tier raises ValueError when no tiers are configured."""
        interface = ModelInterface(providers={}, default_provider="none")
        router = ModelRouter(
            model_interface=interface,
            tiers={},
            tier_policy=None,
            fallback_tiers={},
        )
        request = ModelRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(ValueError, match="No tiers configured"):
            router.select_tier(request)

    def test_ignores_invalid_default_tier_in_policy(self) -> None:
        """select_tier ignores a policy default_tier not in configured tiers."""
        providers = {"real": FakeProvider("real")}
        interface = ModelInterface(providers=providers, default_provider="real")
        router = ModelRouter(
            model_interface=interface,
            tiers={"real": {}},
            tier_policy={"default_tier": "nonexistent"},
            fallback_tiers={},
        )
        request = ModelRequest(messages=[{"role": "user", "content": "hi"}])
        tier = router.select_tier(request)
        assert tier == "real"


# ---------------------------------------------------------------------------
# Tests for fallback
# ---------------------------------------------------------------------------


class TestFallback:
    """Tests for ModelRouter.fallback()."""

    def test_returns_configured_fallback(self, basic_router: ModelRouter) -> None:
        """fallback() returns the configured fallback tier."""
        assert basic_router.fallback("fast") == "quality"
        assert basic_router.fallback("quality") == "fast"

    def test_returns_none_when_no_fallback(self) -> None:
        """fallback() returns None when no fallback is configured for the tier."""
        providers = {"only": FakeProvider("only")}
        interface = ModelInterface(providers=providers, default_provider="only")
        router = ModelRouter(
            model_interface=interface,
            tiers={"only": {}},
            tier_policy=None,
            fallback_tiers={},
        )
        assert router.fallback("only") is None


# ---------------------------------------------------------------------------
# Tests for route
# ---------------------------------------------------------------------------


class TestRoute:
    """Tests for ModelRouter.route()."""

    @pytest.mark.asyncio
    async def test_route_succeeds_on_primary_tier(self, basic_router: ModelRouter) -> None:
        """route() returns response with no substitution when primary succeeds."""
        request = ModelRequest(messages=[{"role": "user", "content": "test"}])
        response, substitution = await basic_router.route(request)

        assert isinstance(response, ModelResponse)
        assert "fast" in response.content
        assert substitution is None

    @pytest.mark.asyncio
    async def test_route_falls_back_and_produces_substitution(
        self, router_with_failing_primary: ModelRouter
    ) -> None:
        """route() falls back to configured tier and returns TierSubstitution."""
        request = ModelRequest(messages=[{"role": "user", "content": "test"}])
        response, substitution = await router_with_failing_primary.route(request)

        assert isinstance(response, ModelResponse)
        assert "quality" in response.content
        assert substitution is not None
        assert isinstance(substitution, TierSubstitution)
        assert substitution.intended_tier == "fast"
        assert substitution.fallback_tier == "quality"

    @pytest.mark.asyncio
    async def test_route_raises_when_no_fallback_available(self) -> None:
        """route() raises ModelProviderError when tier fails and no fallback configured."""
        providers = {"broken": FailingProvider("broken")}
        interface = ModelInterface(providers=providers, default_provider="broken")
        router = ModelRouter(
            model_interface=interface,
            tiers={"broken": {}},
            tier_policy={"default_tier": "broken"},
            fallback_tiers={},  # no fallback
        )
        request = ModelRequest(messages=[{"role": "user", "content": "test"}])

        with pytest.raises(ModelProviderError) as exc_info:
            await router.route(request)
        assert exc_info.value.provider_id == "broken"

    @pytest.mark.asyncio
    async def test_route_raises_when_fallback_also_fails(self) -> None:
        """route() raises ModelProviderError when both primary and fallback fail."""
        providers = {
            "primary": FailingProvider("primary"),
            "backup": FailingProvider("backup"),
        }
        interface = ModelInterface(providers=providers, default_provider="primary")
        router = ModelRouter(
            model_interface=interface,
            tiers={"primary": {}, "backup": {}},
            tier_policy={"default_tier": "primary"},
            fallback_tiers={"primary": "backup"},
        )
        request = ModelRequest(messages=[{"role": "user", "content": "test"}])

        with pytest.raises(ModelProviderError) as exc_info:
            await router.route(request)
        # The error should be from the fallback tier that also failed
        assert exc_info.value.provider_id == "backup"

    @pytest.mark.asyncio
    async def test_route_uses_tier_from_request_metadata(self) -> None:
        """route() respects tier hint in request metadata."""
        providers = {
            "fast": FakeProvider("fast"),
            "quality": FakeProvider("quality"),
        }
        interface = ModelInterface(providers=providers, default_provider="fast")
        router = ModelRouter(
            model_interface=interface,
            tiers={"fast": {}, "quality": {}},
            tier_policy=None,  # no default
            fallback_tiers={},
        )
        request = ModelRequest(
            messages=[{"role": "user", "content": "test"}],
            metadata={"tier": "quality"},
        )
        response, substitution = await router.route(request)

        assert "quality" in response.content
        assert substitution is None


# ---------------------------------------------------------------------------
# Tests for TierSubstitution dataclass
# ---------------------------------------------------------------------------


class TestTierSubstitution:
    """Tests for the TierSubstitution dataclass."""

    def test_frozen_dataclass(self) -> None:
        """TierSubstitution is immutable."""
        sub = TierSubstitution(intended_tier="fast", fallback_tier="quality")
        assert sub.intended_tier == "fast"
        assert sub.fallback_tier == "quality"

        with pytest.raises(Exception):  # FrozenInstanceError
            sub.intended_tier = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two TierSubstitution with same fields are equal."""
        a = TierSubstitution(intended_tier="x", fallback_tier="y")
        b = TierSubstitution(intended_tier="x", fallback_tier="y")
        assert a == b
