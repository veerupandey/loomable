"""loomable.providers.errors - Transient and permanent provider error classification.

These edge errors subclass the kernel ``ModelProviderError`` so existing
``except ModelProviderError`` sites keep working while giving the resilience
layer (``ResilientModel``) enough information to decide retry-vs-fail-fast.
"""

from __future__ import annotations

__all__ = ["TransientProviderError", "PermanentProviderError"]

from loomable.kernel.errors import ModelProviderError


class TransientProviderError(ModelProviderError):
    """A provider failure that MAY succeed on retry (429, 5xx, timeout, conn reset).

    Subclasses the kernel error so existing ``except ModelProviderError`` sites keep
    working; adds ``status_code`` (None for timeouts/connection errors) and
    ``retry_after``.
    """

    def __init__(
        self,
        provider_id: str,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.status_code = status_code
        self.retry_after = retry_after  # parsed from Retry-After header when present
        super().__init__(provider_id)


class PermanentProviderError(ModelProviderError):
    """A provider failure that will NOT succeed on retry (4xx, auth, content policy)."""

    def __init__(
        self,
        provider_id: str,
        *,
        status_code: int | None = None,
    ) -> None:
        self.status_code = status_code
        super().__init__(provider_id)
