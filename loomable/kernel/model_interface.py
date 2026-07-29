"""loomable.kernel.model_interface - Provider-agnostic model routing.

The ModelInterface wraps one or more configured ModelProviders and routes
invocations without any provider-specific agent code. Agents interact solely
with ModelRequest/ModelResponse; swapping providers requires only a
configuration change.
"""

from __future__ import annotations

from loomable.kernel.contracts import ModelProvider
from loomable.kernel.errors import ModelProviderError
from loomable.kernel.models import ModelRequest, ModelResponse


class ModelInterface:
    """Routes model invocations to configured providers.

    The ModelInterface holds a mapping of provider identifiers to
    ``ModelProvider`` instances and a default provider id. Calling
    ``invoke()`` routes the request to the appropriate provider. If a
    requested provider is not registered (unavailable), a
    ``ModelProviderError`` naming it is raised.

    Parameters
    ----------
    providers:
        Mapping of provider identifiers to ``ModelProvider`` instances.
    default_provider:
        The provider id used when no tier/provider is specified.
    """

    def __init__(
        self,
        providers: dict[str, ModelProvider],
        default_provider: str,
    ) -> None:
        self._providers = providers
        self._default_provider = default_provider

    @property
    def providers(self) -> dict[str, ModelProvider]:
        """Return the registered provider mapping (read-only view)."""
        return dict(self._providers)

    @property
    def default_provider(self) -> str:
        """Return the default provider identifier."""
        return self._default_provider

    async def invoke(
        self,
        request: ModelRequest,
        tier: str | None = None,
    ) -> ModelResponse:
        """Invoke a model provider with the given request.

        Parameters
        ----------
        request:
            A provider-agnostic ``ModelRequest``.
        tier:
            Optional provider/tier identifier. When ``None``, the
            default provider is used.

        Returns
        -------
        ModelResponse
            The provider-agnostic response from the model.

        Raises
        ------
        ModelProviderError
            If the resolved provider id is not registered (unavailable).
        """
        provider_id = tier if tier is not None else self._default_provider

        provider = self._providers.get(provider_id)
        if provider is None:
            raise ModelProviderError(provider_id)

        return await provider.complete(request)
