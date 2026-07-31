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

For each task it:
1. Runs forced `single` and forced `plan`
2. Asks an LLM judge which answer is better (A/B order randomized)
3. Runs the heuristic router and checks whether it matched

Writes:
- `/tmp/loomable_experiments/routing_ab_compare.json` (full)
- `experiments/last_learning_summary.json` (scoreboard + lessons)

## Improve loop

1. Edit tasks in `routing_ab.py`
2. Run on a real model
3. Read lessons / mismatches
4. Change router thresholds or inject a model classifier
5. Re-run — keep only changes that win on quality *and* cost

## Latest Z.AI glm-5.2 learnings (comparative judge)

| Task | Judge preferred | Heuristic | Match? |
|------|-----------------|-----------|--------|
| simple_launch | single | single | yes |
| cue_rich_launch | single (margin 2) | plan | no |
| short_faq | tie → single | single | yes |
| multi_compare | single (margin 1) | plan | no |

Bias check (swapped A/B) still preferred SINGLE for cue-rich — not just position bias.

**Takeaway:** On this model/batch, PLAN fan-out worked but often lost on quality/cost vs one strong SINGLE pass.

**Router changes applied:**
- PLAN score threshold raised `3 → 4` (fewer false PLAN escalations)
- Short-answer constraints (`one short paragraph`, etc.) force SINGLE/TOOL_LOOP
- Runs now log `run_strategy` + `plan_workers` for continued learning

Force PLAN when you explicitly want fan-out:

```python
class AlwaysPlan:
    def classify(self, agent_input, *, has_tools): 
        return RunStrategy.PLAN

Agent(..., complexity_router=ComplexityRouter(model_classifier=AlwaysPlan()))
```
