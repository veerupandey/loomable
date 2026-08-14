"""Agent knowledge base — a vector store the model can search.

``knowledge_base=`` is a vector DB (optionally ingested from files).
``retrievers=`` attaches extra search tools on the same Agent.
``create_deep_agent`` is Agent, so it takes the same kwargs.

Tough question: company policy allows committed .env tokens; personal notes
forbid secrets in git. The agent must search both collections and follow the
stricter personal constraint, while still citing the webhook key.

Run::

    python examples/agents/07_knowledge_base.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loomable.agent import Agent, ModelSpec, create_deep_agent
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall

ROOT = Path(__file__).resolve().parent / ".knowledge_base_demo"
ROOT.mkdir(parents=True, exist_ok=True)


def _seed() -> tuple[Path, Path]:
    personal = ROOT / "personal"
    company = ROOT / "company"
    personal.mkdir(exist_ok=True)
    company.mkdir(exist_ok=True)
    (personal / "prefs.md").write_text(
        "# Avery preferences\n\nNever commit secrets. No API tokens in git.\n",
        encoding="utf-8",
    )
    (company / "policy.md").write_text(
        "# Policy\n\nStaging credentials MAY live in a committed internal .env.\n",
        encoding="utf-8",
    )
    (company / "runbook.md").write_text(
        "# Webhooks\n\nSigning secret KEY-WHSEC-4419. Rotate after leaks.\n",
        encoding="utf-8",
    )
    return personal, company


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
                        tool_name="search_personal",
                        args={"query": "commit secrets tokens git", "k": 3},
                    )
                ],
            )
        if self.n == 2:
            return ModelResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="2",
                        tool_name="search_company",
                        args={"query": "webhook KEY-WHSEC .env policy", "k": 5},
                    )
                ],
            )
        return ModelResponse(
            content=(
                "Do not commit the staging token (personal notes). "
                "Webhook key is KEY-WHSEC-4419 (runbook.md)."
            )
        )


async def main() -> None:
    personal, company = _seed()
    kb = {"personal": [personal], "company": [company]}
    # Same knowledge_base on Agent and create_deep_agent (deep agent is Agent).
    _agent = Agent(
        ModelSpec(provider="scripted", provider_impl=_Scripted()),
        user_id="avery",
        knowledge_base=kb,
        use_llm_summarizer=False,
    )
    deep = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_Scripted()),
        user_id="avery",
        knowledge_base=kb,
        workspace=ROOT / "workspace",
        web_search=False,
        url_fetch=False,
        citations=False,
        think_tool=False,
        board=False,
        use_llm_summarizer=False,
    )
    result = await deep.arun(
        "Can I commit STAGING_API_TOKEN per policy? What is the webhook key?"
    )
    print(result.output.text())


if __name__ == "__main__":
    asyncio.run(main())
