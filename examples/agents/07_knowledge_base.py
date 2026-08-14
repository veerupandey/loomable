"""Searchable knowledge base — a vector store the agent queries as tools.

``knowledge_base=`` is a vector DB (optionally ingested from files/dirs).
``retrievers=`` attaches extra search tools on the same Agent.
``create_deep_agent`` is Agent, so it takes the same kwargs.

This script is offline (scripted model). For a live model, swap in Gemini /
Azure OpenAI and keep the same ``knowledge_base=``.

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


def _seed() -> dict[str, list[Path]]:
    personal = ROOT / "personal"
    company = ROOT / "company"
    personal.mkdir(exist_ok=True)
    company.mkdir(exist_ok=True)
    (personal / "prefs.md").write_text(
        "# Avery preferences\n\n"
        "Never commit secrets. No API tokens in git, including .env files.\n",
        encoding="utf-8",
    )
    (company / "policy.md").write_text(
        "# Credential policy\n\n"
        "Staging credentials MAY be stored in a committed internal .env file.\n",
        encoding="utf-8",
    )
    (company / "runbook.md").write_text(
        "# Webhooks\n\n"
        "Current signing secret is DEMO-WH-4419. Rotate after any leak.\n",
        encoding="utf-8",
    )
    return {"personal": [personal], "company": [company]}


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
                        args={"query": "webhook DEMO-WH .env policy", "k": 5},
                    )
                ],
            )
        return ModelResponse(
            content=(
                "Do not commit the staging token (personal notes are stricter "
                "than company .env policy). Webhook key is DEMO-WH-4419 (runbook.md)."
            )
        )


async def _ask(agent: Agent, label: str) -> None:
    result = await agent.arun(
        "Can I commit STAGING_TOKEN=demo-not-a-secret per policy? "
        "What is the webhook signing secret? Cite sources."
    )
    print(f"[{label}] {(result.output.text() or '').strip()}")


async def main() -> None:
    kb = _seed()
    model = ModelSpec(provider="scripted", provider_impl=_Scripted())
    agent = Agent(
        model,
        user_id="avery",
        knowledge_base=kb,
        use_llm_summarizer=False,
        max_tool_iterations=8,
    )
    await _ask(agent, "Agent")

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
        max_tool_iterations=8,
    )
    await _ask(deep, "create_deep_agent")


if __name__ == "__main__":
    asyncio.run(main())
