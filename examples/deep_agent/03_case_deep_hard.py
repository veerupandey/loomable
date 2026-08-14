"""Deep agent + Case accept gate — hard-task spine.

Shows create_deep_agent(mode=\"case\") so planning, workspace tools, and an
accept verifier run through the Case plan→dispatch→synthesize loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loomable.agent import ModelSpec, create_deep_agent
from loomable.flow.loop import VerdictResult
from loomable.kernel.models import ModelRequest, ModelResponse, ToolCall

ROOT = Path(__file__).resolve().parent / ".workspace_case"
ROOT.mkdir(parents=True, exist_ok=True)


class _CaseDeepScript:
    def __init__(self) -> None:
        self.n = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.n += 1
        blob = str(request.messages).lower()
        # Planner
        if "json array" in blob or "break the user task" in blob:
            return ModelResponse(content='["Draft findings into workspace", "Synthesize brief"]')
        # Worker / specialist steps may call tools
        if "draft findings" in blob or "workspace" in blob and self.n < 8:
            if "write_file" not in blob and "findings" not in blob[-200:]:
                return ModelResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=f"w{self.n}",
                            tool_name="write_file",
                            args={
                                "path": "notes/findings.md",
                                "content": "# Findings\nCase+deep agent collaboration works.\n",
                            },
                        )
                    ],
                )
        if "integrate" in blob or "merge" in blob or "synthesizer" in blob:
            return ModelResponse(
                content=(
                    "FINAL BRIEF: Deep agents on loomable combine todos, workspace "
                    "files, and Case accept gates. SEV-ready."
                )
            )
        return ModelResponse(content=f"step-done-{self.n}")


def accept(output, context) -> VerdictResult:  # noqa: ANN001
    text = output.text() or ""
    ok = "FINAL BRIEF" in text and "SEV" in text
    return VerdictResult(ok=ok, detail="" if ok else "need FINAL BRIEF + SEV")


async def main() -> None:
    agent = create_deep_agent(
        ModelSpec(provider="scripted", provider_impl=_CaseDeepScript()),
        workspace=ROOT,
        web_search=False,
        enable_task_tool=False,
        think_tool=False,
        mode="case",
        accept=accept,
        max_rounds=2,
        max_plan_steps=2,
        board=True,
        modalities="text",
        instructions="Hard task: produce a FINAL BRIEF mentioning SEV.",
    )
    result = await agent.arun("Ship a deep-agent architecture brief with SEV label")
    print(result.output.text())
    print("metadata.case =", (result.metadata or {}).get("case"))


if __name__ == "__main__":
    asyncio.run(main())
