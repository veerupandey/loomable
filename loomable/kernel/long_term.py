"""loomable.kernel.long_term - Long-Term Store with pluggable vector backend.

The LongTermStore indexes items in a pluggable VectorBackend (defaulting to zvec)
and returns similarity-ranked results on query. Alternative backends can be
substituted without agent changes — only the backend instance needs to change.

An unavailable backend raises MemoryBackendError naming the backend.
"""

from __future__ import annotations

import math
from typing import Any

from loomable.kernel.contracts import VectorBackend
from loomable.kernel.errors import MemoryBackendError


# ---------------------------------------------------------------------------
# Default backend: ZvecVectorBackend (in-memory cosine-similarity store)
# ---------------------------------------------------------------------------


class ZvecVectorBackend:
    """A lightweight in-memory vector backend using cosine similarity.

    This serves as the default 'zvec' backend. It stores vectors in a simple
    dictionary and computes cosine similarity for queries. Suitable for
    development, testing, and low-volume usage.
    """

    def __init__(self) -> None:
        # Stores {id: {"vector": [...], "metadata": {...}}}
        self._store: dict[str, dict[str, Any]] = {}

    async def index(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """Index a vector with associated metadata."""
        self._store[id] = {"vector": vector, "metadata": metadata}

    async def query(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        """Query for top-k most similar vectors by cosine similarity.

        Returns results ranked by non-increasing similarity. Each result
        dict contains 'id', 'score', and all metadata fields.
        """
        if not self._store:
            return []

        scored: list[tuple[float, str, dict[str, Any]]] = []
        for item_id, item in self._store.items():
            score = _cosine_similarity(vector, item["vector"])
            scored.append((score, item_id, item["metadata"]))

        # Sort by score descending (non-increasing similarity)
        scored.sort(key=lambda x: x[0], reverse=True)

        results: list[dict[str, Any]] = []
        for score, item_id, metadata in scored[:k]:
            result = {"id": item_id, "score": score, **metadata}
            results.append(result)

        return results

    async def delete(self, id: str) -> None:
        """Remove an indexed vector by id."""
        self._store.pop(id, None)


# ---------------------------------------------------------------------------
# Long-Term Store
# ---------------------------------------------------------------------------


class LongTermStore:
    """Long-term memory store backed by a pluggable VectorBackend.

    Indexes items as vectors and retrieves them ranked by similarity.
    Defaults to ZvecVectorBackend (in-memory cosine similarity) when no
    backend is provided.

    Alternative backends (FAISS, Pinecone, etc.) can be substituted by
    passing any object satisfying the VectorBackend protocol — no agent
    changes required.

    If the backend is unavailable or raises, MemoryBackendError is raised
    naming the backend.
    """

    def __init__(
        self,
        backend: VectorBackend | None = None,
        backend_name: str = "zvec",
    ) -> None:
        self.backend: VectorBackend = backend or ZvecVectorBackend()
        self.backend_name = backend_name

    async def index(self, id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """Index an item in the long-term store.

        Args:
            id: Unique identifier for the item.
            vector: Embedding vector for similarity search.
            metadata: Arbitrary metadata associated with the item.

        Raises:
            MemoryBackendError: If the backend is unavailable or fails.
        """
        try:
            await self.backend.index(id, vector, metadata)
        except MemoryBackendError:
            raise
        except Exception as exc:
            raise MemoryBackendError(self.backend_name) from exc

    async def query(self, vector: list[float], k: int) -> list[dict[str, Any]]:
        """Query the long-term store for the top-k most similar items.

        Args:
            vector: Query embedding vector.
            k: Number of results to return.

        Returns:
            List of result dicts ordered by non-increasing similarity.
            Each dict contains 'id', 'score', and all indexed metadata.

        Raises:
            MemoryBackendError: If the backend is unavailable or fails.
        """
        try:
            return await self.backend.query(vector, k)
        except MemoryBackendError:
            raise
        except Exception as exc:
            raise MemoryBackendError(self.backend_name) from exc

    async def delete(self, id: str) -> None:
        """Remove an item from the long-term store.

        Args:
            id: Identifier of the item to remove.

        Raises:
            MemoryBackendError: If the backend is unavailable or fails.
        """
        try:
            await self.backend.delete(id)
        except MemoryBackendError:
            raise
        except Exception as exc:
            raise MemoryBackendError(self.backend_name) from exc


# ---------------------------------------------------------------------------
# Utility: cosine similarity
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in [-1, 1]. Returns 0.0 for zero-magnitude vectors.
    """
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)
