"""Offline hashing embedder for local code indexes (no API key)."""

from __future__ import annotations

import hashlib
import re


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,64}|[0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-tokens embedder into a fixed-dim vector.

    Good enough for local code search demos and tests. Swap for
    :class:`~loomable.providers.embedders.OpenAIEmbedder` (or any Embedder)
    in production.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = int(dim)

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _TOKEN.findall(text or "")
        if not tokens:
            return vec
        for tok in tokens:
            digest = hashlib.sha1(tok.lower().encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        # L2 normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec
