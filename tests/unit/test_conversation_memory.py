"""Unit tests for conversational memory injection (Req 15).

When an agent has a session, its recent turns are replayed into each request so it
remembers across calls. Stateless agents (no session_id) are unaffected.
"""

from __future__ import annotations

from loomable.agent import Agent, ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse


class RecordingProvider:
    """Captures each request's flattened user/assistant text for assertions."""

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


class TestConversationMemory:
    async def test_second_call_includes_prior_turns(self):
        """With a session, the 2nd request replays the 1st exchange (Req 15)."""
        provider = RecordingProvider(reply="Nice to meet you!")
        agent = Agent(
            model=ModelSpec(provider="rec", provider_impl=provider),
            session_id="chat-1",
        )

        await agent.arun("My name is Ada.")
        await agent.arun("What is my name?")

        # First request only has the current input (no prior history yet).
        assert any("My name is Ada." in t for t in provider.texts_in(0))
        assert not any("What is my name?" in t for t in provider.texts_in(0))

        # Second request replays the first user turn AND the first assistant reply,
        # then the new question — so the model can answer "Ada".
        second = provider.texts_in(1)
        assert any("My name is Ada." in t for t in second)
        assert any("Nice to meet you!" in t for t in second)
        assert any("What is my name?" in t for t in second)

    async def test_history_precedes_current_input(self):
        provider = RecordingProvider()
        agent = Agent(
            model=ModelSpec(provider="rec", provider_impl=provider),
            session_id="chat-2",
        )
        await agent.arun("first")
        await agent.arun("second")

        texts = provider.texts_in(1)
        # Prior turns appear before the current input.
        assert texts.index("first") < texts.index("second")

    async def test_no_session_means_no_memory(self):
        """Without a session_id, requests stay stateless (transport-safe)."""
        provider = RecordingProvider()
        agent = Agent(model=ModelSpec(provider="rec", provider_impl=provider))

        await agent.arun("alpha")
        await agent.arun("beta")

        # The second request must NOT contain the first input.
        assert not any("alpha" in t for t in provider.texts_in(1))

    async def test_use_memory_false_disables_injection(self):
        provider = RecordingProvider()
        agent = Agent(
            model=ModelSpec(provider="rec", provider_impl=provider),
            session_id="chat-3",
            use_memory=False,
        )
        await agent.arun("remember me")
        await agent.arun("do you?")
        assert not any("remember me" in t for t in provider.texts_in(1))

    async def test_memory_window_caps_replayed_turns(self):
        """Only the last `memory_window` turns are replayed."""
        provider = RecordingProvider()
        agent = Agent(
            model=ModelSpec(provider="rec", provider_impl=provider),
            session_id="chat-4",
            memory_window=2,  # keep only the last 2 turns (one exchange)
        )
        await agent.arun("turn-one")   # after: l1 = [u:turn-one, a:ok]
        await agent.arun("turn-two")   # sees last 2 turns (turn-one, ok)
        await agent.arun("turn-three")  # window=2 => only [u:turn-two, a:ok] replayed

        third = provider.texts_in(2)
        assert any("turn-two" in t for t in third)
        assert not any("turn-one" in t for t in third)
