"""Phase C gate — ContextPolicy auto-compaction under long shift handoff.

Forces many L1 turns past compaction_threshold and asserts L2 summaries appear
without losing pinned facts.
"""

from __future__ import annotations

import asyncio

from loomable import Agent, ContextPolicy
from loomable.agent import ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse, Session, Turn
from loomable.kernel.stores import SessionStore
from loomable.kernel.summarizer import Summarizer


class _Echo:
    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        return ModelResponse(
            content=f"ack-{self.n}",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


async def main() -> None:
    provider = _Echo()
    store = SessionStore(":memory:")
    agent = Agent(
        model=ModelSpec(provider="echo", provider_impl=provider),
        session_id="shift-handoff-1",
        memory_window=4,
        compaction_threshold=6,
        token_budget=2048,
    )
    built = agent.build()
    built.session_store = store
    built.persist_session = True
    built.session = Session(session_id="shift-handoff-1", agent_config_ref="echo")
    built.summarizer = Summarizer(1)
    built.context_policy = ContextPolicy(
        memory_window=4,
        compaction_threshold=6,
        token_budget=2048,
        soft_limit_ratio=0.5,
        hard_limit_ratio=0.7,
    )

    # Pin a critical fact
    built.session.l1.append(
        Turn(role="system", content="PIN:INC-88421 is SEV-1 BharatNova", tokens=0, step=0)
    )
    built.pinned_steps.add(0)

    # Drive many turns to force compaction
    for i in range(10):
        await built.arun(f"shift update #{i}: queue depth rising")

    assert len(built.session.l2) >= 1, "expected L2 summaries after compaction"
    pinned_ok = any("PIN:INC-88421" in (t.content or "") for t in built.session.l1)
    assert pinned_ok, "pinned incident fact was compacted away"
    assert len(built.session.l1) <= built.compaction_threshold + 2

    bulky = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "tool", "content": "X" * 5000, "tool_call_id": "1"},
    ]
    spilled = built.context_policy.spill_bulky_tool_messages(
        bulky, max_tool_chars=200, force=True
    )
    assert len(spilled[1]["content"]) < 5000
    assert "truncated" in spilled[1]["content"]

    print("[ok] Phase C memory/compaction gate")
    print(f"    l1={len(built.session.l1)} l2={len(built.session.l2)} pinned_ok={pinned_ok}")


if __name__ == "__main__":
    asyncio.run(main())
