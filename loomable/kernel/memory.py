"""loomable.kernel.memory - Memory Manager coordinating the three memory tiers.

The MemoryManager maintains:
- L1: raw recent conversation turns (list[Turn])
- L2: compressed summaries and entity representations (list[StructuredSummary])
- L3: vector-indexed episodic memory via LongTermStore

Requirements covered: 11.1, 11.2, 11.3, 11.4
"""

from __future__ import annotations

from typing import Any

from loomable.kernel.long_term import LongTermStore
from loomable.kernel.models import StructuredSummary, Turn


class MemoryManager:
    """Coordinates multi-tier memory: L1 raw turns, L2 summaries, L3 episodic.

    L1 stores raw recent conversation turns for immediate context.
    L2 stores compressed summaries and entity representations derived from
    conversation history (produced by checkpoint summarization).
    L3 is a vector-indexed episodic memory store for long-term recall.
    Default backend is Alibaba zvec (``.loomable/memory_zvec``); pass an
    explicit ``LongTermStore`` for FAISS, Postgres, or in-memory.

    Key methods:
    - record_turn(): appends a turn to L1.
    - add_summary(): appends a structured summary to L2.
    - recall(): retrieves similarity-ranked items from L3 via the LongTermStore.
    """

    def __init__(self, long_term_store: LongTermStore | None = None) -> None:
        """Initialize the Memory Manager.

        Args:
            long_term_store: Optional LongTermStore instance for L3 episodic
                memory. If not provided, a default LongTermStore (using the
                zvec in-memory backend) is created.
        """
        self.l1: list[Turn] = []
        self.l2: list[StructuredSummary] = []
        self.l3: LongTermStore = long_term_store or LongTermStore()

    def record_turn(self, turn: Turn) -> None:
        """Record a conversation turn in L1 (raw recent turns).

        Args:
            turn: The Turn to append to L1 memory.
        """
        self.l1.append(turn)

    def add_summary(self, summary: StructuredSummary) -> None:
        """Add a structured summary to L2 (compressed summaries/entities).

        Args:
            summary: The StructuredSummary to append to L2 memory.
        """
        self.l2.append(summary)

    async def recall(self, query_vector: list[float], k: int) -> list[dict[str, Any]]:
        """Retrieve similarity-ranked episodic items from L3 memory.

        Delegates to the LongTermStore's query method which returns items
        ordered by non-increasing similarity to the query vector.

        Args:
            query_vector: The embedding vector to query against L3.
            k: The number of top results to return.

        Returns:
            A list of result dicts ordered by non-increasing similarity.
            Each dict contains 'id', 'score', and all indexed metadata.

        Raises:
            MemoryBackendError: If the vector backend is unavailable.
        """
        return await self.l3.query(query_vector, k)
