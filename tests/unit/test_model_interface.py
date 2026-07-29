"""Unit tests for the ModelInterface class.

Validates:
- Provider-agnostic invocation via ModelRequest/ModelResponse (Req 2.1, 2.2)
- Routing to the configured provider without provider-specific code (Req 2.3)
- ModelProviderError raised for unavailable providers (Req 2.4)
"""

from __future__ import annotations

import pytest

from loomable.kernel import ModelInterface, ModelProviderError, ModelRequest, ModelResponse


class FakeProviderA:
    """A fake model provider that echoes back content tagged with its id."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=f"provider_a: {request.messages[0]['content'] if request.messages else ''}",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


class FakeProviderB:
    """A second fake model provider, demonstrating provider swapping."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content=f"provider_b: {request.messages[0]['content'] if request.messages else ''}",
            usage={"input_tokens": 8, "output_tokens": 3},
        )


@pytest.fixture
def model_interface() -> ModelInterface:
    """ModelInterface with two registered providers, 'a' as default."""
    return ModelInterface(
        providers={"a": FakeProviderA(), "b": FakeProviderB()},
        default_provider="a",
    )


@pytest.mark.asyncio
async def test_invoke_uses_default_provider(model_interface: ModelInterface) -> None:
    """invoke() without tier routes to the default provider."""
    request = ModelRequest(messages=[{"role": "user", "content": "hello"}])
    response = await model_interface.invoke(request)

    assert isinstance(response, ModelResponse)
    assert response.content == "provider_a: hello"


@pytest.mark.asyncio
async def test_invoke_routes_to_specified_tier(model_interface: ModelInterface) -> None:
    """invoke() with tier routes to the named provider."""
    request = ModelRequest(messages=[{"role": "user", "content": "world"}])
    response = await model_interface.invoke(request, tier="b")

    assert isinstance(response, ModelResponse)
    assert response.content == "provider_b: world"


@pytest.mark.asyncio
async def test_invoke_unavailable_provider_raises_error(model_interface: ModelInterface) -> None:
    """invoke() raises ModelProviderError naming the unavailable provider."""
    request = ModelRequest(messages=[{"role": "user", "content": "test"}])

    with pytest.raises(ModelProviderError) as exc_info:
        await model_interface.invoke(request, tier="nonexistent")

    assert exc_info.value.provider_id == "nonexistent"
    assert "nonexistent" in str(exc_info.value)


@pytest.mark.asyncio
async def test_invoke_unavailable_default_raises_error() -> None:
    """invoke() raises ModelProviderError when default provider is missing."""
    interface = ModelInterface(providers={}, default_provider="missing")
    request = ModelRequest(messages=[{"role": "user", "content": "test"}])

    with pytest.raises(ModelProviderError) as exc_info:
        await interface.invoke(request)

    assert exc_info.value.provider_id == "missing"


@pytest.mark.asyncio
async def test_same_request_shape_across_providers(model_interface: ModelInterface) -> None:
    """The same ModelRequest shape works across different providers (Req 2.1)."""
    request = ModelRequest(
        messages=[{"role": "user", "content": "same request"}],
        temperature=0.7,
        max_tokens=100,
    )

    response_a = await model_interface.invoke(request, tier="a")
    response_b = await model_interface.invoke(request, tier="b")

    # Both return well-formed ModelResponse regardless of provider
    assert isinstance(response_a, ModelResponse)
    assert isinstance(response_b, ModelResponse)
    assert "provider_a" in response_a.content
    assert "provider_b" in response_b.content


@pytest.mark.asyncio
async def test_swapping_provider_requires_no_agent_code_change() -> None:
    """Swapping the configured provider requires no changes to agent code (Req 2.3).

    Demonstrates that an agent-style function using ModelInterface works
    identically regardless of which provider is configured as default.
    """

    async def agent_logic(interface: ModelInterface) -> str:
        """Simulated agent logic that uses the model interface."""
        request = ModelRequest(messages=[{"role": "user", "content": "plan"}])
        response = await interface.invoke(request)
        return response.content

    # Same agent logic, different providers configured
    interface_a = ModelInterface(providers={"main": FakeProviderA()}, default_provider="main")
    interface_b = ModelInterface(providers={"main": FakeProviderB()}, default_provider="main")

    result_a = await agent_logic(interface_a)
    result_b = await agent_logic(interface_b)

    assert "provider_a" in result_a
    assert "provider_b" in result_b


def test_providers_property_returns_copy(model_interface: ModelInterface) -> None:
    """The providers property returns a copy, not the internal dict."""
    providers = model_interface.providers
    providers["c"] = FakeProviderA()
    assert "c" not in model_interface.providers


def test_default_provider_property(model_interface: ModelInterface) -> None:
    """The default_provider property returns the configured default."""
    assert model_interface.default_provider == "a"
