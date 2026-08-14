# Loomable beta research plan

**Goal:** Graduate loomable from **alpha** (`0.1.0`, Development Status :: 3) to a credible **public beta** — feature-complete enough for external users, with a stability contract, without claiming GA/production polish.

**Industry bar for beta** (Google AIP-181 / AndroidX / peer agent SDKs):
- Feature-complete for the declared scope (not experimental stubs)
- Publicly installable; breaking changes rare and documented
- Reliability primitives that production agent stacks expect: durable state, cancel, budgets, HITL, observability hooks
- CI that runs more than the happy unit path
- Security baseline for any network-facing surface

**Loomable today:** strong alpha. Deep agent + discovery + Case/Workflow/Memory/Postgres/AG-UI are real and Gemini-proven. Gaps are packaging, cancel/serve hardening, CI breadth, API contract, and a few incomplete edges.

---

## Beta definition (what we promise)

| In beta scope | Explicitly out of beta (document as limits) |
|---------------|-----------------------------------------------|
| Agent / Team / Workflow / Case | Hosted provider-native `tool_search` |
| `create_deep_agent` + research skill + discovery | Remote object-store workspace FS (unless shipped) |
| Memory.compose + session stores (memory/file/sqlite/postgres) | Merging `ExtensionRegistry` into Discovery without design |
| AG-UI SSE (`mount_agent` / `mount_case`) with auth baseline | Multi-region managed runtime (LangSmith-style) |
| Postgres checkpointer + Case resume | Guaranteed ≥50% schema reduction (optional profile) |
| Public PyPI (or tagged) install + CHANGELOG + stability policy | Perfect ecosystem parity with Agno/Crew marketing |

---

## Exit criteria (measurable — ship beta when all green)

1. **Packaging:** version `0.2.0bN` (or `0.2.0`), classifier `4 - Beta`, README badge beta, `__version__` matches pyproject, installable from PyPI **or** GitHub Release wheel with documented `pip install`.
2. **Stability policy:** `docs/STABILITY.md` — stable vs experimental APIs; deprecation window for breaks.
3. **CI:** 3.11 + 3.12 green on `tests/unit` + `tests/properties` + `tests/toolkits` + non-live `tests/integration`; optional gated jobs for `POSTGRES_URL` and `DEEP_AGENT_LIVE`.
4. **Cancel contract:** public cancel API + SSE disconnect → `RunContext.cancel`; tests assert `stop_reason=cancelled` and no further model calls.
5. **Serve baseline:** reference auth middleware (API key or bearer) on `mount_*`; locked-down example + test that anonymous `/run` is rejected when auth enabled.
6. **Public surface freeze:** top-level `__all__` covered by a unit test; either wire or clearly mark `ExtensionRegistry` as kernel-internal/experimental.
7. **Durability E2E:** docker-compose Postgres path green for checkpointer + Memory + Case resume (documented command).
8. **Deep gate:** `05_live_gemini_gate.py` remains PASS on a supported model; publish measured schema budget (revise ≥50% target or ship `discovery_core="slim"` profile).
9. **Docs ops:** CHANGELOG for beta cut; README install matches packaging; examples index includes `deep_agent/`; SECURITY.md with report path.
10. **Quality bar:** no open P0 security issues in toolkits (SSRF/path); `require_tools` / deliverable-complete behavior covered by tests.

---

## Research findings → workstreams

### W1 — Packaging & API contract (beta identity)

**Why:** Alpha cannot be “public beta” without an install story and stability promise.

| Work item | Detail |
|-----------|--------|
| Version + classifiers | `0.2.0b0` → beta classifier; badge update |
| `__version__` | Single source of truth |
| STABILITY.md | Stable: Agent, Team, Workflow, Case, Memory, create_deep_agent, mount_*; Experimental: ExtensionRegistry, Flow optimizer, slim discovery profile |
| CHANGELOG.md | Beta cut notes + migration from 0.1 |
| Public surface test | Fail CI if top-level exports drift undocumented |
| Research profile | `create_deep_agent(..., profile="research")` |

### W2 — Reliability (production agent bar)

Peers treat **durable state + cancel + budgets** as the demo→prod divide.

| Work item | Detail |
|-----------|--------|
| Cancel API | `BuiltAgent.cancel(run_id|ctx)` / cooperative cancel from serve |
| SSE disconnect | FastAPI disconnect → cancel in-flight run |
| Cancel tests | Unit + serve integration: stop_reason cancelled |
| Budget defaults | Document token/step/tool iteration defaults for deep vs simple Agent |
| Fail-closed option | `Agent(strict_require_tools=True)` / `Workflow(strict_require_tools=True)` (WR-021) |
| Team coordinate | Soft `require_tools` on delegates + deterministic fallback for uncalled members (WR-020) |

### W3 — Serve / security baseline

| Work item | Detail |
|-----------|--------|
| Auth hook | `mount_agent(..., api_key=)` or dependency injection |
| Example | Locked AG-UI mount with key; README snippet |
| Threat model note | SSRF/path/python sandboxes already exist — document trust boundary (serve is edge) |
| SECURITY.md | How to report vulnerabilities |

### W4 — CI / quality gates

| Work item | Detail |
|-----------|--------|
| Expand CI | properties + toolkits + integration (non-live) |
| Optional live jobs | `workflow_dispatch` or secret-gated Postgres + Gemini |
| Lint/type | ruff (+ optional mypy on public `__all__`) — start advisory then gate |
| Coverage | Report on CI; target floor for public modules (no vanity 100%) |

### W5 — Deep agent / discovery (competitive beta)

Already P0–P2 + Gemini PASS. Remaining for beta polish:

| Work item | Detail |
|-----------|--------|
| Slim core profile | `discovery_core="research-slim"` to approach ≥50% schema cut **or** revise COMPETITIVE metric to “correctness + deferred non-core” |
| Remote FS decision | Document local-only as beta limit **or** ship S3/GCS-backed WorkspaceStore with round-trip tests |
| Specialist discovery | Already wired — add one integration test that specialist inherits discovery |
| Eval harness | Scripted golden + optional live rubric (accept, sources≥1, report path) in CI-friendly form |

### W6 — Docs & DX

| Work item | Detail |
|-----------|--------|
| README honesty | Replace “production” marketing with “beta — durable primitives, expect polish gaps” |
| examples/README | Include `deep_agent/` tree |
| API.md | Stability markers; cancel; auth mount |
| Beta cookbook | Single path: Agent → deep research → Case → mount SSE |
| Postgres recipe | compose up → checkpointer + memory + Case resume |

### W7 — Explicit non-goals for beta

Do **not** block beta on:
- Hosted tool_search / LangSmith-managed runtime
- Merging ExtensionRegistry into DiscoveryRuntime
- Full Agno AgentOS RBAC/audit parity
- Perfect schema-budget ≥50% (ship slim profile or revise metric)

---

## Priority order (technical dependency)

```text
W1 Packaging/stability  ─┐
W4 CI expansion         ─┼─→ public beta identity
W2 Cancel + budgets     ─┤
W3 Serve auth baseline  ─┘
         │
         ├─→ W5 Deep polish (slim profile / FS decision / eval)
         └─→ W6 Docs/DX + CHANGELOG → cut 0.2.0bN
```

**Cut beta when exit criteria 1–10 are green.** Anything unfinished moves to “beta known limits” in STABILITY.md / README — not silent alpha debt.

---

## Already beta-worthy (do not rebuild)

- Agent · Team · Workflow · Case on one Runnable contract
- AG-UI SSE mounts + war-room examples
- Postgres checkpointer + Memory.compose + session stores
- Deep agent research skill, discovery P0–P2, live Gemini gate PASS
- Toolkit SSRF / path safety / citation accept gates

---

## Validation plan for the beta cut

| Gate | Command / artifact |
|------|--------------------|
| Unit+property+toolkit | CI matrix green |
| Postgres durability | `docker compose up` + documented E2E script |
| Cancel | New unit/serve tests |
| Auth mount | Example test anonymous rejected |
| Deep | `DEEP_AGENT_LIVE=1 python examples/deep_agent/05_live_gemini_gate.py` PASS |
| Install | Fresh venv `pip install` from beta artifact; `import loomable; loomable.__version__` |

---

## Success definition

External developer can:
1. `pip install loomable` (beta)
2. Run Agent / create_deep_agent research / Case / mount SSE with auth
3. Rely on Postgres resume + cancel + documented stability policy
4. File issues against a CHANGELOG’d beta without surprise silent API rewrites

That is **beta**, not GA: bugs OK; missing managed hosting OK; breaking changes only with deprecation notes.
