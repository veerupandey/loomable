"""Build pluggable retrievers for docs / markdown / code and attach to Agent.

Run::

    python examples/advanced/02_build_retriever.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loomable.agent import Agent, ModelSpec
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall
from loomable.retrieval import build_retriever, list_strategies

ROOT = Path(__file__).resolve().parent / ".retrieval_demo"
ROOT.mkdir(parents=True, exist_ok=True)


def _seed() -> Path:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "auth.md").write_text(
        "# Authentication\n\nUse OAuth2 bearer tokens for API login.\n",
        encoding="utf-8",
    )
    (docs / "pricing.md").write_text(
        "# Pricing\n\nEnterprise plans include SSO and audit logs.\n",
        encoding="utf-8",
    )
    return docs


class _Scripted:
    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        if self.n == 1:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="1",
                        tool_name="docs",
                        args={"query": "OAuth2 login", "k": 2},
                    )
                ],
            )
        return ModelResponse(content="Retriever demo complete.")


async def main() -> None:
    docs = _seed()
    print("strategies", list_strategies())
    retriever = await build_retriever(
        [docs, {"id": "tip", "text": "Prefer hybrid mode for mixed corpora."}],
        name="docs",
        mode="hybrid",
        strategy="auto",
        persist_path=ROOT / "docs_zvec",  # Alibaba zvec (pip install loomable[zvec])
    )
    agent = Agent(
        model=ModelSpec(provider="scripted", provider_impl=_Scripted()),
        retrievers=[retriever],
        use_llm_summarizer=False,
    )
    result = await agent.arun("How do we authenticate?")
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
