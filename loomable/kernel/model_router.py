"""loomable.kernel.model_router - Tiered model routing with fallback.

The ModelRouter selects a model tier for each model call according to a
configured cost/latency policy, routes through the ModelInterface, and falls
back to a configured alternative tier when the selected tier is unavailable.
A TierSubstitution record is produced whenever a fallback is used.
"""

from __future__ import annotations

from dataclasses import dataclass

from loomable.kernel.errors import ModelProviderError
from loomable.kernel.model_interface import ModelInterface
from loomable.kernel.models import (
    AgentConfig,
    ModelRequest,
    ModelResponse,
    TierPolicy,
)


@dataclass(frozen=True)
class TierSubstitution:
    """Record of a tier substitution event.

    Produced when the intended tier is unavailable and the router falls
    back to an alternative tier.
    """

    intended_tier: str
    fallback_tier: str


class ModelRouter:
    """Selects among tiered models per cost/latency policy and handles fallback.

    Parameters
    ----------
    model_interface:
        The ModelInterface used to invoke the selected provider/tier.
    tiers:
        Mapping of tier names to their provider specs (used to validate
        that a selected tier exists in configuration).
    tier_policy:
        Optional policy dict controlling tier selection logic. When None,
        the first tier in the configured tiers is used as default.
    fallback_tiers:
        Mapping of tier name -> fallback tier name. Used when the selected
        tier is unavailable.
    """

    def __init__(
        self,
        model_interface: ModelInterface,
        tiers: dict[str, object],
        tier_policy: TierPolicy | None = None,
        fallback_tiers: dict[str, str] | None = None,
    ) -> None:
        self._model_interface = model_interface
        self._tiers = tiers
        self._tier_policy = tier_policy or {}
        self._fallback_tiers = fallback_tiers or {}

    def select_tier(self, request: ModelRequest) -> str:
        """Select a model tier for the given request based on cost/latency policy.

        The policy may specify a ``default_tier`` key. If it does, that tier
        is returned (provided it exists in the configured tiers). Otherwise
        the first configured tier is used.

        A more sophisticated implementation could inspect request metadata
        (e.g. estimated token count, urgency flags) to pick tiers dynamically;
        for now the policy-driven default is sufficient.

        Parameters
        ----------
        request:
            The model request being routed.

        Returns
        -------
        str
            The name of the selected tier.

        Raises
        ------
        ValueError
            If no tiers are configured.
        """
        if not self._tiers:
            raise ValueError("No tiers configured for model routing")

        # Check policy for a default_tier directive
        default_tier = self._tier_policy.get("default_tier")
        if default_tier and default_tier in self._tiers:
            return default_tier

        # Check request metadata for a tier hint
        tier_hint = request.metadata.get("tier") if request.metadata else None
        if tier_hint and tier_hint in self._tiers:
            return tier_hint

        # Fall back to the first configured tier
        return next(iter(self._tiers))

    def fallback(self, tier: str) -> str | None:
        """Return the configured fallback tier for the given tier, or None.

        Parameters
        ----------
        tier:
            The tier whose fallback is requested.

        Returns
        -------
        str | None
            The fallback tier name if configured, otherwise None.
        """
        return self._fallback_tiers.get(tier)

    async def route(
        self, request: ModelRequest
    ) -> tuple[ModelResponse, TierSubstitution | None]:
        """Route a model request through tiered selection with fallback.

        Selects a tier via ``select_tier()``, invokes through the
        ModelInterface. If the selected tier is unavailable
        (``ModelProviderError``), attempts the configured fallback tier.
        Produces a ``TierSubstitution`` record when a fallback is used.

        Parameters
        ----------
        request:
            The provider-agnostic model request.

        Returns
        -------
        tuple[ModelResponse, TierSubstitution | None]
            The model response and an optional substitution record (present
            only when the fallback tier was used).

        Raises
        ------
        ModelProviderError
            If neither the selected tier nor its fallback (if any) is
            available.
        """
        selected_tier = self.select_tier(request)

        try:
            response = await self._model_interface.invoke(request, tier=selected_tier)
            return response, None
        except ModelProviderError:
            # Attempt fallback
            fallback_tier = self.fallback(selected_tier)
            if fallback_tier is None:
                raise

            response = await self._model_interface.invoke(request, tier=fallback_tier)
            substitution = TierSubstitution(
                intended_tier=selected_tier,
                fallback_tier=fallback_tier,
            )
            return response, substitution
