# Routing experiments

Run real A/B tests to learn when `ComplexityRouter` should choose
`single` vs `plan`.

## How to run

```bash
export ZAI_API_KEY="your-key"
export ZAI_BASE_URL="https://api.z.ai/api/coding/paas/v4"
export ZAI_MODEL="glm-5.2"

python experiments/routing_ab.py
```

For each task it runs three modes:
- forced `single`
- forced `plan`
- heuristic router

Then writes learnings to `/tmp/loomable_experiments/routing_ab.json`.

## Improve loop

1. Add / edit tasks in `routing_ab.py`
2. Run A/B on a real model
3. Read lessons (coverage, latency, worker count, heuristic mistakes)
4. Change router cues / thresholds OR plug in a model classifier
5. Re-run and keep only changes that win on quality *and* cost

## What we learned (Z.AI glm-5.2 batch)

| Task | Heuristic | Forced PLAN workers | Takeaway |
|------|-----------|---------------------|----------|
| simple launch | single | 5 | SINGLE already covered topics; PLAN ~2× slower |
| cue-rich launch | plan | 5 | Heuristic correct; fan-out works |
| short FAQ | single | 4 | Keep FAQ on SINGLE |
| multi-compare | plan | 5 | PLAN works but keyword coverage same as SINGLE |

Keyword coverage alone is too weak to judge quality.
Next experiments should add an LLM-as-judge for depth/structure,
and keep logging `result.metadata["run_strategy"]` + `plan_workers`.
