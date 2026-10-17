"""Opt-in smoke test against Gemini, OpenAI, or Cloudflare Workers AI."""
import os

import pytest
from pydantic import BaseModel

from loomable import Agent, Workflow, tool
from loomable.agent import AgentPaused, DurabilityOptions
from loomable.persist import JsonFileCheckpointer, SQLiteCheckpointer
from loomable.providers.gemini import GeminiProvider
from loomable.providers.openai import OpenAIProvider

pytestmark = pytest.mark.live


class Answer(BaseModel):
    value: int


@tool
def lookup() -> str:
    """Return the fixture code."""
    return "7243"


def live_provider():
    backend = os.environ.get("LOOMABLE_LIVE_PROVIDER", "gemini")
    if backend == "gemini":
        token = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if token:
            model = os.environ.get("GEMINI_MODEL") or "gemini-flash-latest"
            return GeminiProvider(model, api_key=token, timeout=20)
    elif backend == "openai":
        token = os.environ.get("OPENAI_API_KEY")
        if token:
            model = os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
            return OpenAIProvider(model, api_key=token, timeout=20)
    elif backend == "cloudflare":
        account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        token = os.environ.get("CLOUDFLARE_API_TOKEN")
        model = os.environ.get("LOOMABLE_CLOUDFLARE_MODEL")
        if all((account, token, model)):
            return OpenAIProvider(
                model, api_key=token, timeout=20,
                base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
            )
    else:
        pytest.fail("LOOMABLE_LIVE_PROVIDER must be gemini, openai, or cloudflare")
    pytest.skip(f"Configure credentials for {backend} before running live tests")


async def test_live_model_chat_completions():
    async with live_provider() as provider:
        chat = Agent(provider, session_id="workers-ai-smoke")
        result = await chat.arun("Remember the test code 7243. Reply with just the code.")
        assert "7243" in result.text
        chunks = [chunk async for chunk in chat.astream("What test code did I give you? Reply with digits only.")]
        assert chunks[-1].done and "7243" in chunks[-1].result.text
        typed = await Agent(provider, response_model=Answer).arun("Return JSON with value set to 7.")
        assert typed.structured.value == 7
        worker = Agent(provider, tools=[lookup], require_tools=["lookup"], strict_require_tools=True)
        flow = Workflow("smoke").step("lookup", worker)
        run = await flow.arun("Call lookup to retrieve the fixture code, then reply with the code.")
        assert "7243" in run.output.text()


@pytest.mark.parametrize("store_class", [JsonFileCheckpointer, SQLiteCheckpointer])
async def test_live_idempotent_run_reuses_completed_result(tmp_path, store_class):
    checkpoint_path = str(tmp_path / "journal")
    prompt = "Return JSON with value set to 7243."
    async with live_provider() as provider:
        def build():
            return Agent(
                provider, session_id="idempotent-7243", response_model=Answer,
                durability=DurabilityOptions(store_class(checkpoint_path)),
            )

        first = await build().arun_once(prompt)
        replayed = await build().arun_once(prompt)
    assert first.structured.value == replayed.structured.value == 7243
    assert first.text == replayed.text
    assert first.usage == replayed.usage
    assert replayed.metadata == {"run_id": first.metadata["run_id"], "replayed": True}


async def test_live_idempotent_run_preserves_approval_boundary(tmp_path):
    calls = []

    @tool
    def read_code() -> str:
        """Read the staging verification code."""
        calls.append("read")
        return "7243"

    prompt = "Call read_code and reply with the exact returned code."
    async with live_provider() as provider:
        def build():
            return Agent(
                provider, session_id="approval-7243", tools=[read_code],
                require_tools=["read_code"], strict_require_tools=True,
                require_confirmation=["read_code"],
                durability=DurabilityOptions(SQLiteCheckpointer(str(tmp_path / "approval.sqlite"))),
            )

        with pytest.raises(AgentPaused) as first:
            await build().arun_once(prompt)
        with pytest.raises(AgentPaused) as retried:
            await build().arun_once(prompt)
        call_id = first.value.pending[0].id
        assert retried.value.pending[0].id == call_id
        assert retried.value.run_id == first.value.run_id
        assert calls == []
        result = await build().acontinue_run(first.value.run_id, approvals={call_id: True})
        replayed = await build().arun_once(prompt)
    assert "7243" in result.text and replayed.text == result.text
    assert calls == ["read"]
