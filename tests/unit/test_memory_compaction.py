"""Unit tests for automatic memory compaction (Req 6.1–6.5).

When retained turns exceed `compaction_threshold`, the oldest overflow turns
are summarized via the kernel Summarizer into session.l2 and dropped from
session.l1, preserving only the most recent window. `_memory_prefix` prepends
L2 summaries ahead of the retained recent turns.
"""

from __future__ import annotations

from loomable.agent import Agent, ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse


class RecordingProvider:
    """Captures each request for assertions."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(content=self.reply, usage={"output_tokens": 1})

    def texts_in(self, index: int) -> list[str]:
        """All text parts across all messages of the request at `index`."""
        out: list[str] = []
        for message in self.requests[index].messages:
            content = message.get("content", "")
            if isinstance(content, list):
                out.extend(p.get("text", "") for p in content if p.get("type") == "text")
            elif isinstance(content, str):
                out.append(content)
        return out

    def roles_in(self, index: int) -> list[str]:
        return [m["role"] for m in self.requests[index].messages]


class TestMemoryCompaction:
    """Tests for the automatic memory compaction feature (Req 6.1–6.5)."""

    async def test_compaction_triggers_when_threshold_exceeded(self):
        """When l1 exceeds compaction_threshold, oldest turns are summarized (Req 6.1, 6.2)."""
        provider = RecordingProvider()
        # memory_window=4, compaction_threshold=6 — so after 4 turns (2 exchanges)
        # plus 2 more turns (3rd exchange = 6 turns total), crossing 6 triggers compaction.
        agent = Agent(
            model=ModelSpec(provider="rec", provider_impl=provider),
            session_id="compact-1",
            memory_window=4,
            compaction_threshold=6,
        )
        built = agent.build()

        # 3 exchanges produce 6 turns in l1; the 4th pushes to 8 which exceeds 6.
        await built.arun("msg-1")
        await built.arun("msg-2")
        await built.arun("msg-3")

        # After 3 exchanges: 6 turns. Not yet exceeding (threshold is 6, we need > 6).
        assert len(built.session.l1) == 6
        assert len(built.session.l2) == 0

        # 4th exchange pushes to 8 turns, then compaction fires.
        await built.arun("msg-4")

        # After compaction: l1 should have at most memory_window (4) turns.
        assert len(built.session.l1) <= 4
        # A summary should be stored in l2.
        assert len(built.session.l2) == 1
        # The summary covers the compacted turns.
        assert built.session.l2[0].text  # non-empty summary text

    async def test_compaction_preserves_recent_window(self):
        """After compaction, the most recent turns (up to memory_window) remain (Req 6.4)."""
        provider = RecordingProvider()
        agent = Agent(
            model=ModelSpec(provider="rec", provider_impl=provider),
            session_id="compact-2",
            memory_window=4,
            compaction_threshold=6,
        )
        built = agent.build()

        await built.arun("old-1")
        await built.arun("old-2")
        await built.arun("old-3")
        await built.arun("recent")  # triggers compaction

        # The most recent turns should remain (the window).
        remaining_contents = [t.content for t in built.session.l1]
        assert "recent" in remaining_contents[-2]  # user turn "recent"
        assert "ok" in remaining_contents[-1]  # assistant reply

    async def test_l2_summary_prepended_in_context(self):
        """After compaction, _memory_prefix prepends L2 summaries (Req 6.3)."""
        provider = RecordingProvider()
        agent = Agent(
            model=ModelSpec(provider="rec", provider_impl=provider),
            session_id="compact-3",
            memory_window=4,
            compaction_threshold=6,
        )
        built = agent.build()

        await built.arun("old-1")
        await built.arun("old-2")
        await built.arun("old-3")
        await built.arun("recent")  # triggers compaction

        # Now the next request should have the L2 summary prepended as system.
        await built.arun("after-compact")

        # In the last request (index 4), roles should start with "system" (the summary).
        last_request_roles = provider.roles_in(4)
        assert last_request_roles[0] == "system"

        # And the summary text should appear in the request.
        last_texts = provider.texts_in(4)
        assert any("Summary" in t for t in last_texts)

    async def test_no_compaction_below_threshold(self):
        """When l1 is at or below compaction_threshold, no compaction occurs."""
        provider = RecordingProvider()
        agent = Agent(
            model=ModelSpec(provider="rec", provider_impl=provider),
            session_id="compact-4",
            memory_window=4,
            compaction_threshold=10,  # high threshold
        )
        built = agent.build()

        await built.arun("msg-1")
        await built.arun("msg-2")

        # 4 turns, well below threshold of 10.
        assert len(built.session.l1) == 4
        assert len(built.session.l2) == 0

    async def test_compaction_reuses_kernel_summarizer(self):
        """Compaction uses the kernel Summarizer (Req 6.5), not a custom implementation."""
        provider = RecordingProvider()
        agent = Agent(
            model=ModelSpec(provider="rec", provider_impl=provider),
            session_id="compact-5",
            memory_window=4,
            compaction_threshold=6,
        )
        built = agent.build()

        # The BuiltAgent should have a Summarizer from the kernel.
        from loomable.kernel.summarizer import Summarizer
        assert isinstance(built.summarizer, Summarizer)

    async def test_multiple_compactions_accumulate_l2(self):
        """Repeated threshold crossings add multiple summaries to L2."""
        provider = RecordingProvider()
        agent = Agent(
            model=ModelSpec(provider="rec", provider_impl=provider),
            session_id="compact-6",
            memory_window=2,
            compaction_threshold=4,
        )
        built = agent.build()

        # 1st+2nd exchange: 4 turns. 3rd exchange pushes to 6 => compaction.
        await built.arun("batch-1-a")
        await built.arun("batch-1-b")
        await built.arun("batch-1-c")  # 6 turns > 4 threshold => compact

        assert len(built.session.l2) == 1

        # Continue: after compaction l1 has memory_window=2 turns. Two more exchanges
        # push l1 to 6 again, triggering another compaction.
        await built.arun("batch-2-a")
        await built.arun("batch-2-b")  # l1 goes to 6 again => compact

        assert len(built.session.l2) == 2

    async def test_stateless_agent_no_compaction(self):
        """Without a session_id (stateless), compaction never triggers."""
        provider = RecordingProvider()
        agent = Agent(
            model=ModelSpec(provider="rec", provider_impl=provider),
            memory_window=4,
            compaction_threshold=4,
        )
        built = agent.build()

        await built.arun("a")
        await built.arun("b")
        await built.arun("c")

        # No session persistence means l1 stays empty (turns aren't recorded).
        assert len(built.session.l2) == 0
