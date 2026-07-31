"""Experiment harness: learn when PLAN beats SINGLE (comparative LLM judge).

For each task:
  1) run forced SINGLE and forced PLAN
  2) ask the model which answer is better (head-to-head)
  3) run the heuristic router and see if it matched the winner

Secrets from the environment only — never commit keys.

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
from loomable.kernel.models import ModelRequest
from loomable.providers.openai import OpenAIProvider


TASKS: list[dict[str, Any]] = [
    {
        "id": "simple_launch",
        "text": (
            "Help me launch AI software that helps factories plan shop-floor work. "
            "Cover who to sell to, who we compete with, simple pricing, "
            "and a 90-day plan. Keep it plain English."
        ),
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
    },
    {
        "id": "short_faq",
        "text": "What is shop-floor scheduling in one short paragraph?",
    },
    {
        "id": "multi_compare",
        "text": (
            "Compare Python, Rust, and Go for factory control APIs. "
            "Analyze and break down the work step by step. For each language "
            "cover speed, safety, and hiring. Decompose into multiple steps, "
            "then synthesize one recommendation."
        ),
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
class RunOutcome:
    mode: str
    chosen: str
    model_calls: int
    workers: int
    latency_s: float
    answer: str
    answer_chars: int


@dataclass
class CompareVerdict:
    winner: str  # single | plan | tie
    margin: float  # 0-2 how much better
    reason: str


@dataclass
class TaskResult:
    task_id: str
    single: dict[str, Any]
    plan: dict[str, Any]
    heuristic: dict[str, Any]
    compare: dict[str, Any]
    preferred: str
    heuristic_match: bool
    notes: list[str] = field(default_factory=list)


def provider_kwargs() -> dict[str, Any]:
    api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set ZAI_API_KEY first")
    return {
        "model": os.environ.get("ZAI_MODEL", "glm-5.2"),
        "api_key": api_key,
        "base_url": os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
        "timeout": 180.0,
    }


async def run_mode(task_text: str, mode: str) -> RunOutcome:
    provider = CountingProvider(**provider_kwargs())
    if mode == "single":
        router = ComplexityRouter(model_classifier=AlwaysStrategy(RunStrategy.SINGLE))
    elif mode == "plan":
        router = ComplexityRouter(model_classifier=AlwaysStrategy(RunStrategy.PLAN))
    else:
        router = ComplexityRouter()

    chosen = router.classify(AgentInput.from_text(task_text), has_tools=False)
    agent = Agent(
        model=provider,
        instructions="Plain English. Short sentences. Be concrete.",
        complexity_router=router,
    )
    t0 = time.perf_counter()
    result = await agent.arun(task_text)
    latency = time.perf_counter() - t0
    answer = result.output.text()
    return RunOutcome(
        mode=mode,
        chosen=chosen.value,
        model_calls=provider.calls,
        workers=provider.roles.count("worker"),
        latency_s=round(latency, 1),
        answer=answer,
        answer_chars=len(answer),
    )


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()
    # Recover JSON object if model added prose.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


async def compare_answers(task: str, single_answer: str, plan_answer: str) -> CompareVerdict:
    """Head-to-head judge: which answer is better for the task?"""
    import random

    judge = OpenAIProvider(**provider_kwargs())
    # Randomize presentation order to reduce position bias.
    if random.random() < 0.5:
        order = ("single", "plan")
        a_text, b_text = single_answer, plan_answer
    else:
        order = ("plan", "single")
        a_text, b_text = plan_answer, single_answer

    prompt = (
        "Compare Answer A and Answer B for the TASK.\n"
        "Pick the better answer for a busy CEO.\n"
        "Criteria (priority order):\n"
        "1) specificity (concrete roles, numbers, steps)\n"
        "2) usefulness (actionable)\n"
        "3) structure (easy to scan)\n"
        "4) fidelity to the ask (including length constraints)\n"
        "If nearly equal, choose tie.\n"
        "Return ONLY JSON:\n"
        '{"winner":"A"|"B"|"tie","margin":0|1|2,"reason":"short"}\n'
        "margin 0=tie/negligible, 1=clearly better, 2=much better.\n\n"
        f"TASK:\n{task}\n\n"
        f"ANSWER A:\n{a_text[:5000]}\n\n"
        f"ANSWER B:\n{b_text[:5000]}\n"
    )
    resp = await judge.complete(
        ModelRequest(messages=[{"role": "user", "content": prompt}], temperature=0)
    )
    try:
        data = _parse_json_object(resp.content or "")
        raw = str(data.get("winner", "tie")).upper()
        if raw == "A":
            winner = order[0]
        elif raw == "B":
            winner = order[1]
        else:
            winner = "tie"
        margin = float(data.get("margin", 0))
        return CompareVerdict(winner=winner, margin=margin, reason=str(data.get("reason", ""))[:300])
    except (json.JSONDecodeError, TypeError, ValueError):
        if len(plan_answer) > len(single_answer) * 1.4:
            return CompareVerdict("plan", 1, "fallback-length")
        if len(single_answer) > len(plan_answer) * 1.4:
            return CompareVerdict("single", 1, "fallback-length")
        return CompareVerdict("tie", 0, "fallback-tie")


def learn(results: list[TaskResult]) -> dict[str, Any]:
    prefer_plan: list[str] = []
    prefer_single: list[str] = []
    lessons: list[str] = []
    mismatches = 0

    for r in results:
        if r.preferred == "plan":
            prefer_plan.append(r.task_id)
        elif r.preferred == "single":
            prefer_single.append(r.task_id)

        if not r.heuristic_match:
            mismatches += 1
            lessons.append(
                f"{r.task_id}: judge preferred {r.preferred} "
                f"(margin {r.compare.get('margin')}) but heuristic chose {r.heuristic['chosen']}"
            )
        else:
            lessons.append(
                f"{r.task_id}: heuristic matched preference ({r.preferred})"
            )

    tweaks: list[str] = []
    if "simple_launch" in prefer_plan:
        tweaks.append("Escalate multi-topic launch asks ('cover A, B, and C') to PLAN.")
    if "simple_launch" in prefer_single:
        tweaks.append("Keep multi-topic asks on SINGLE when judge finds no quality lift.")
    if "short_faq" in prefer_single:
        tweaks.append("Keep short FAQ / one-paragraph asks on SINGLE.")
    if "short_faq" in prefer_plan:
        tweaks.append("Even short FAQs sometimes gain from PLAN — inspect fidelity-to-length.")
    if "cue_rich_launch" in prefer_plan or "multi_compare" in prefer_plan:
        tweaks.append("Keep strong compare/step-by-step/decompose cues on PLAN.")
    if mismatches == 0:
        tweaks.append("No router mismatches this batch — hold thresholds, keep logging.")
    tweaks.append("Continue logging run_strategy + plan_workers and re-run this harness.")

    return {
        "prefer_plan_for": prefer_plan,
        "prefer_single_for": prefer_single,
        "lessons": lessons,
        "router_tweaks": tweaks,
        "mismatch_count": mismatches,
    }


async def run_task(task: dict[str, Any]) -> TaskResult:
    print(f"## Task: {task['id']}")
    print("  → single ...", flush=True)
    single = await run_mode(task["text"], "single")
    print(f"     calls={single.model_calls} t={single.latency_s}s chars={single.answer_chars}")

    print("  → plan ...", flush=True)
    plan = await run_mode(task["text"], "plan")
    print(
        f"     calls={plan.model_calls} workers={plan.workers} "
        f"t={plan.latency_s}s chars={plan.answer_chars}"
    )

    print("  → comparative judge ...", flush=True)
    verdict = await compare_answers(task["text"], single.answer, plan.answer)
    print(f"     winner={verdict.winner} margin={verdict.margin} ({verdict.reason})")

    print("  → heuristic ...", flush=True)
    heur = await run_mode(task["text"], "heuristic")
    print(
        f"     chose={heur.chosen} calls={heur.model_calls} "
        f"workers={heur.workers} t={heur.latency_s}s"
    )

    # Preference: judge winner, but if tie prefer cheaper SINGLE.
    if verdict.winner == "tie":
        preferred = "single"
    else:
        preferred = verdict.winner

    # Cost override: if PLAN barely wins (margin 0/1) but is >2.5x slower, prefer single.
    notes: list[str] = []
    if (
        preferred == "plan"
        and verdict.margin <= 1
        and plan.latency_s > single.latency_s * 2.5
        and task["id"] == "short_faq"
    ):
        preferred = "single"
        notes.append("cost-override: short FAQ, weak PLAN margin")

    heuristic_match = heur.chosen == preferred or (
        preferred == "single" and heur.chosen == "single"
    )
    # If preferred plan and heuristic chose plan — match.
    # If preferred single and heuristic chose single — match.
    heuristic_match = heur.chosen == preferred

    print(f"  preferred={preferred} heuristic_match={heuristic_match}\n")

    return TaskResult(
        task_id=task["id"],
        single={
            "chosen": single.chosen,
            "model_calls": single.model_calls,
            "workers": single.workers,
            "latency_s": single.latency_s,
            "answer_chars": single.answer_chars,
            "preview": re.sub(r"\s+", " ", single.answer)[:160],
        },
        plan={
            "chosen": plan.chosen,
            "model_calls": plan.model_calls,
            "workers": plan.workers,
            "latency_s": plan.latency_s,
            "answer_chars": plan.answer_chars,
            "preview": re.sub(r"\s+", " ", plan.answer)[:160],
        },
        heuristic={
            "chosen": heur.chosen,
            "model_calls": heur.model_calls,
            "workers": heur.workers,
            "latency_s": heur.latency_s,
            "answer_chars": heur.answer_chars,
        },
        compare={
            "winner": verdict.winner,
            "margin": verdict.margin,
            "reason": verdict.reason,
        },
        preferred=preferred,
        heuristic_match=heuristic_match,
        notes=notes,
    )


async def main() -> None:
    print("Running comparative routing experiments on Z.AI...\n")
    results: list[TaskResult] = []
    for task in TASKS:
        results.append(await run_task(task))

    summary = learn(results)
    out_dir = Path("/tmp/loomable_experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "provider": os.environ.get("ZAI_MODEL", "glm-5.2"),
        "results": [asdict(r) for r in results],
        "learning": summary,
    }
    out_path = out_dir / "routing_ab_compare.json"
    out_path.write_text(json.dumps(payload, indent=2))

    summary_path = Path(__file__).resolve().parent / "last_learning_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "provider": payload["provider"],
                "learning": summary,
                "scoreboard": [
                    {
                        "task_id": r.task_id,
                        "preferred": r.preferred,
                        "judge_winner": r.compare["winner"],
                        "margin": r.compare["margin"],
                        "heuristic": r.heuristic["chosen"],
                        "match": r.heuristic_match,
                        "single_s": r.single["latency_s"],
                        "plan_s": r.plan["latency_s"],
                        "plan_workers": r.plan["workers"],
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )

    print("=" * 64)
    print("LEARNINGS")
    print("=" * 64)
    for lesson in summary["lessons"]:
        print(f"- {lesson}")
    print("\nPrefer PLAN for:", ", ".join(summary["prefer_plan_for"]) or "(none)")
    print("Prefer SINGLE for:", ", ".join(summary["prefer_single_for"]) or "(none)")
    print(f"Heuristic mismatches: {summary['mismatch_count']}")
    print("\nSuggested router tweaks:")
    for t in summary["router_tweaks"]:
        print(f"  • {t}")
    print(f"\nWrote {out_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
