"""Experiment harness: learn when PLAN beats SINGLE.

Runs the same tasks three ways against an OpenAI-compatible provider (Z.AI):
  A) force SINGLE
  B) force PLAN
  C) ComplexityRouter heuristic

Logs strategy, worker count, latency, model calls, and a simple coverage score.
Secrets must come from the environment — never commit keys.

  export ZAI_API_KEY=...
  python experiments/routing_ab.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loomable.agent import Agent
from loomable.agent.routing import ComplexityRouter, RunStrategy
from loomable.content import AgentInput
from loomable.providers.openai import OpenAIProvider


TASKS: list[dict[str, Any]] = [
    {
        "id": "simple_launch",
        "text": (
            "Help me launch AI software that helps factories plan shop-floor work. "
            "Cover who to sell to, who we compete with, simple pricing, "
            "and a 90-day plan. Keep it plain English."
        ),
        "must_cover": ["sell", "compet", "pric", "90", "day"],
    },
    {
        "id": "cue_rich_launch",
        "text": (
            "Compare and analyze how to launch AI software that helps factories "
            "plan shop-floor work. Break down the work step by step. "
            "For each area cover: who to sell to, who we compete with, simple pricing, "
            "and a 90-day plan. Decompose into multiple steps, then synthesize "
            "one clear CEO answer in plain English."
        ),
        "must_cover": ["sell", "compet", "pric", "90", "day"],
    },
    {
        "id": "short_faq",
        "text": "What is shop-floor scheduling in one short paragraph?",
        "must_cover": ["schedul"],
    },
    {
        "id": "multi_compare",
        "text": (
            "Compare Python, Rust, and Go for factory control APIs. "
            "Analyze and break down the work step by step. For each language "
            "cover speed, safety, and hiring. Decompose into multiple steps, "
            "then synthesize one recommendation."
        ),
        "must_cover": ["python", "rust", "go", "recommend"],
    },
]


class AlwaysStrategy:
    def __init__(self, strategy: RunStrategy) -> None:
        self.strategy = strategy

    def classify(self, agent_input, *, has_tools: bool) -> RunStrategy:
        return self.strategy


class CountingProvider(OpenAIProvider):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.roles: list[str] = []
        self.calls = 0

    async def complete(self, request):  # type: ignore[override]
        self.calls += 1
        text = "\n".join(str(getattr(m, "content", m)) for m in (request.messages or []))
        lower = text.lower()
        if "you are a planner" in lower and "json array" in lower:
            self.roles.append("planner")
        elif "results from the planned steps" in lower:
            self.roles.append("synthesizer")
        elif "complete only this step" in lower:
            self.roles.append("worker")
        else:
            self.roles.append("other")
        return await super().complete(request)


@dataclass
class TrialResult:
    task_id: str
    mode: str  # single | plan | heuristic
    chosen: str
    model_calls: int
    workers: int
    latency_s: float
    coverage: float
    answer_chars: int
    answer_preview: str
    notes: list[str] = field(default_factory=list)


def coverage_score(answer: str, needles: list[str]) -> float:
    lower = answer.lower()
    hits = sum(1 for n in needles if n.lower() in lower)
    return hits / max(len(needles), 1)


def make_provider() -> CountingProvider:
    api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set ZAI_API_KEY first")
    return CountingProvider(
        model=os.environ.get("ZAI_MODEL", "glm-5.2"),
        api_key=api_key,
        base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
        timeout=180.0,
    )


async def run_trial(task: dict[str, Any], mode: str) -> TrialResult:
    provider = make_provider()
    if mode == "single":
        router = ComplexityRouter(model_classifier=AlwaysStrategy(RunStrategy.SINGLE))
    elif mode == "plan":
        router = ComplexityRouter(model_classifier=AlwaysStrategy(RunStrategy.PLAN))
    else:
        router = ComplexityRouter()

    chosen = router.classify(AgentInput.from_text(task["text"]), has_tools=False)
    agent = Agent(
        model=provider,
        instructions="Plain English. Short sentences. Be concrete.",
        complexity_router=router,
    )

    t0 = time.perf_counter()
    result = await agent.arun(task["text"])
    latency = time.perf_counter() - t0
    answer = result.output.text()
    cov = coverage_score(answer, task["must_cover"])
    notes: list[str] = []
    if mode == "heuristic" and chosen == RunStrategy.SINGLE and cov < 1.0:
        notes.append("heuristic stayed SINGLE and missed coverage")
    if mode == "plan" and provider.roles.count("worker") == 0:
        notes.append("forced PLAN but no workers ran")
    if latency > 90:
        notes.append("slow")

    return TrialResult(
        task_id=task["id"],
        mode=mode,
        chosen=chosen.value,
        model_calls=provider.calls,
        workers=provider.roles.count("worker"),
        latency_s=round(latency, 1),
        coverage=round(cov, 2),
        answer_chars=len(answer),
        answer_preview=re.sub(r"\s+", " ", answer)[:160],
        notes=notes,
    )


def learn(results: list[TrialResult]) -> dict[str, Any]:
    """Derive simple routing lessons from A/B outcomes."""
    by_task: dict[str, list[TrialResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)

    lessons: list[str] = []
    prefer_plan_when: list[str] = []
    prefer_single_when: list[str] = []

    for task_id, trials in by_task.items():
        single = next(t for t in trials if t.mode == "single")
        plan = next(t for t in trials if t.mode == "plan")
        heur = next(t for t in trials if t.mode == "heuristic")

        # Prefer PLAN if better coverage, or same coverage with richer answer,
        # unless PLAN is much slower for a short FAQ-like task.
        plan_better = (
            plan.coverage > single.coverage
            or (plan.coverage == single.coverage and plan.answer_chars > single.answer_chars * 1.2
                and plan.workers >= 2)
        )
        single_enough = single.coverage >= 1.0 and single.latency_s < plan.latency_s * 0.5

        if plan_better and not single_enough:
            prefer_plan_when.append(task_id)
            if heur.chosen != "plan":
                lessons.append(
                    f"{task_id}: PLAN won (cov {plan.coverage} vs {single.coverage}, "
                    f"{plan.workers} workers) but heuristic chose {heur.chosen}"
                )
        if single_enough:
            prefer_single_when.append(task_id)
            if heur.chosen == "plan":
                lessons.append(
                    f"{task_id}: SINGLE was enough/faster but heuristic chose PLAN"
                )
        if heur.chosen == "plan" and plan.workers == 0:
            lessons.append(f"{task_id}: heuristic chose PLAN but fan-out produced 0 workers")

    return {
        "prefer_plan_for": prefer_plan_when,
        "prefer_single_for": prefer_single_when,
        "lessons": lessons,
        "router_tweaks": [
            "Count multi-part lists (1) 2) 3) / Cover A, B, C) as plan cues",
            "Treat 'cover X and Y and Z' as multi-step even without 'step by step'",
            "Lower PLAN score threshold from 3 → 2 when >=3 topic sections are requested",
            "Keep SHORT FAQ / one-paragraph asks on SINGLE",
            "Log chosen strategy + workers on every run for continued learning",
        ],
    }


async def main() -> None:
    modes = ["single", "plan", "heuristic"]
    results: list[TrialResult] = []
    print("Running routing A/B experiments on Z.AI...\n")

    for task in TASKS:
        print(f"## Task: {task['id']}")
        for mode in modes:
            print(f"  → mode={mode} ...", flush=True)
            trial = await run_trial(task, mode)
            results.append(trial)
            print(
                f"     chose={trial.chosen} calls={trial.model_calls} "
                f"workers={trial.workers} cov={trial.coverage} "
                f"t={trial.latency_s}s notes={trial.notes}"
            )
        print()

    summary = learn(results)
    out_dir = Path("/tmp/loomable_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": os.environ.get("ZAI_MODEL", "glm-5.2"),
        "results": [asdict(r) for r in results],
        "learning": summary,
    }
    out_path = out_dir / "routing_ab.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print("=" * 64)
    print("LEARNINGS")
    print("=" * 64)
    for lesson in summary["lessons"] or ["No heuristic mistakes on this batch."]:
        print(f"- {lesson}")
    print("\nPrefer PLAN for:", ", ".join(summary["prefer_plan_for"]) or "(none)")
    print("Prefer SINGLE for:", ", ".join(summary["prefer_single_for"]) or "(none)")
    print("\nSuggested router tweaks:")
    for t in summary["router_tweaks"]:
        print(f"  • {t}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
