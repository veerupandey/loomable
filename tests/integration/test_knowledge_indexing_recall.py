# Feature: agent-ergonomics, Property 16

"""Property 16: Attached knowledge is indexed and recalled.

For any set of knowledge documents and a query relevant to one of them, building
indexes each document and a run SHALL recall the relevant document and include it
in the model context.

**Validates: Requirements 8.2, 8.3, 8.5**
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from loomable.agent.builder import Agent
from loomable.kernel.long_term import LongTermStore
from loomable.kernel.models import ModelResponse


# ---------------------------------------------------------------------------
# Helpers: Fake Embedder and Mock Model Provider
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """A deterministic embedder that produces vectors based on keyword overlap.

    Each document/query is embedded into a fixed-dimension vector where each
    dimension corresponds to a word from the vocabulary (all unique words across
    all documents). The vector is a bag-of-words indicator (1.0 if word is present,
    0.0 otherwise). This ensures that cosine similarity between a query and a
    document correlates with shared word content, making recall deterministic.
    """

    def __init__(self, vocabulary: list[str] | None = None) -> None:
        self._vocabulary: list[str] = vocabulary or []
        self._embed_calls: list[str] = []

    def _ensure_vocabulary(self, texts: list[str]) -> None:
        """Build vocabulary from a set of texts if not already built."""
        if not self._vocabulary:
            words: set[str] = set()
            for text in texts:
                words.update(text.lower().split())
            self._vocabulary = sorted(words)

    async def embed(self, text: str) -> list[float]:
        """Embed text into a bag-of-words vector over the vocabulary."""
        self._embed_calls.append(text)
        words = set(text.lower().split())
        return [1.0 if w in words else 0.0 for w in self._vocabulary]


class _CapturingProvider:
    """A model provider that captures the messages it receives and returns a canned response."""

    def __init__(self) -> None:
        self.captured_requests: list[Any] = []

    async def complete(self, request: Any) -> ModelResponse:
        self.captured_requests.append(request)
        return ModelResponse(
            content="I acknowledge the knowledge context.",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestKnowledgeIndexingAndRecall:
    """Integration tests for Property 16: Attached knowledge is indexed and recalled."""

    def test_single_document_indexed_and_recalled(self) -> None:
        """A single knowledge document is embedded at build time, and at run time
        the input is embedded and the document is recalled into model context."""
        doc = "Python is a programming language used for data science"
        query = "Tell me about Python programming"

        # Build vocabulary from both doc and query for deterministic embeddings
        all_texts = [doc, query]
        embedder = _FakeEmbedder()
        embedder._ensure_vocabulary(all_texts)

        provider = _CapturingProvider()
        built = Agent(
            model=provider,
            knowledge=[doc],
            embedder=embedder,
        ).build()

        # Verify the document was indexed (embedder was called at build time)
        assert doc in embedder._embed_calls

        # Verify the LongTermStore was created and populated
        assert built.long_term is not None

        # Run the agent with a query relevant to the document
        import asyncio
        result = asyncio.run(built.arun(query))

        # The embedder should have been called with the query at run time
        assert query in embedder._embed_calls

        # The model should have received the knowledge document in its context
        assert len(provider.captured_requests) == 1
        request = provider.captured_requests[0]
        # Find the recalled knowledge in the messages (as a system message)
        knowledge_found = False
        for msg in request.messages:
            if msg.get("role") == "system":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and doc in part.get("text", ""):
                            knowledge_found = True
                            break
        assert knowledge_found, (
            f"Expected document '{doc}' to appear in model context messages. "
            f"Messages: {request.messages}"
        )

    def test_multiple_documents_recalls_most_relevant(self) -> None:
        """With multiple knowledge docs, the most relevant one to the query is recalled."""
        docs = [
            "Rust is a systems programming language focused on safety",
            "Machine learning uses neural networks for pattern recognition",
            "Python is widely used for web development and data analysis",
        ]
        query = "How do neural networks work in machine learning"

        # Build vocabulary from all texts
        all_texts = docs + [query]
        embedder = _FakeEmbedder()
        embedder._ensure_vocabulary(all_texts)

        provider = _CapturingProvider()
        built = Agent(
            model=provider,
            knowledge=docs,
            embedder=embedder,
        ).build()

        # All docs should have been embedded at build time
        for doc in docs:
            assert doc in embedder._embed_calls

        # Run with a query most similar to the ML document
        import asyncio
        result = asyncio.run(built.arun(query))

        # The query should have been embedded at run time
        assert query in embedder._embed_calls

        # The model context should include the ML document (most relevant)
        request = provider.captured_requests[0]
        ml_doc = docs[1]
        recalled_texts = []
        for msg in request.messages:
            if msg.get("role") == "system":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text", ""):
                            recalled_texts.append(part["text"])

        assert ml_doc in recalled_texts, (
            f"Expected the ML document to be recalled. Got system texts: {recalled_texts}"
        )

    def test_embedder_called_for_all_docs_at_build_and_query_at_run(self) -> None:
        """The embedder is called once per document at build time, and once for
        the input query at run time."""
        docs = [
            "Document about cats and their behavior",
            "Document about dogs and training techniques",
            "Document about birds and migration patterns",
        ]
        query = "Tell me about cat behavior"

        all_texts = docs + [query]
        embedder = _FakeEmbedder()
        embedder._ensure_vocabulary(all_texts)

        provider = _CapturingProvider()
        built = Agent(
            model=provider,
            knowledge=docs,
            embedder=embedder,
        ).build()

        # After build: embedder should have been called once per document
        build_calls = list(embedder._embed_calls)
        assert len(build_calls) == len(docs)
        for doc in docs:
            assert doc in build_calls

        # After run: embedder should also have been called with the query
        import asyncio
        asyncio.run(built.arun(query))

        # Total calls = docs (build) + query (run)
        assert len(embedder._embed_calls) == len(docs) + 1
        assert embedder._embed_calls[-1] == query

    def test_knowledge_uses_long_term_store(self) -> None:
        """Building with knowledge= creates a LongTermStore with indexed documents
        (reuses the kernel LongTermStore — Req 8.5)."""
        docs = ["Alpha document content", "Beta document content"]

        embedder = _FakeEmbedder()
        embedder._ensure_vocabulary(docs)

        provider = _CapturingProvider()
        built = Agent(
            model=provider,
            knowledge=docs,
            embedder=embedder,
        ).build()

        # LongTermStore should be set on the built agent
        assert built.long_term is not None
        assert isinstance(built.long_term, LongTermStore)
        # The embedder should be stored on the built agent
        assert built.embedder is embedder

    def test_no_knowledge_no_recall(self) -> None:
        """Without knowledge configured, no recall happens and no LongTermStore is created."""
        provider = _CapturingProvider()
        built = Agent(model=provider).build()

        assert built.long_term is None
        assert built.embedder is None

        import asyncio
        result = asyncio.run(built.arun("Hello"))

        # No knowledge-related system messages should be present
        request = provider.captured_requests[0]
        # Should only have the user message (no system messages from knowledge)
        system_msgs = [
            msg for msg in request.messages
            if msg.get("role") == "system"
        ]
        # No knowledge system messages expected
        knowledge_msgs = [
            msg for msg in system_msgs
            if any(
                isinstance(p, dict) and "document" in p.get("text", "").lower()
                for p in msg.get("content", [])
                if isinstance(p, dict)
            )
        ]
        assert knowledge_msgs == []

    def test_knowledge_top_k_limits_recalled_documents(self) -> None:
        """The knowledge_top_k parameter limits how many documents are recalled."""
        # Create many documents but set top_k=1
        docs = [
            "Apple is a fruit that grows on trees",
            "Banana is a tropical fruit rich in potassium",
            "Cherry is a small red fruit often used in desserts",
            "Date is a sweet fruit from palm trees",
        ]
        query = "Tell me about fruits"

        all_texts = docs + [query]
        embedder = _FakeEmbedder()
        embedder._ensure_vocabulary(all_texts)

        provider = _CapturingProvider()
        built = Agent(
            model=provider,
            knowledge=docs,
            embedder=embedder,
            knowledge_top_k=1,
        ).build()

        import asyncio
        asyncio.run(built.arun(query))

        # Only 1 document should be recalled (top_k=1)
        request = provider.captured_requests[0]
        knowledge_system_msgs = []
        for msg in request.messages:
            if msg.get("role") == "system":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text", "")
                            # Check if this system message is one of our docs
                            if any(doc == text for doc in docs):
                                knowledge_system_msgs.append(text)

        assert len(knowledge_system_msgs) == 1, (
            f"Expected exactly 1 recalled document (top_k=1), "
            f"got {len(knowledge_system_msgs)}: {knowledge_system_msgs}"
        )
