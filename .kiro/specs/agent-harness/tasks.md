# Implementation Plan: agent-harness

## Overview

This plan hardens the high-level `loomable.agent` run path for production. All work is additive and lives in the edge layers (`loomable.agent`, `loomable.providers`, plus small new edge modules); `loomable.kernel` is never modified. The feature is unified by one new seam — `RunContext` — threaded through `_run_single` / `_run_tool_loop`, which every workstream reads from.

The build order introduces the standalone edge modules first (events, context seam, errors, resilient model, summarizer, notes, reasoning tools, router), then threads them into the `BuiltAgent` run path in `loomable/agent/builder.py` one concern at a time (loop control, tool gating, context bounding, pinned facts, loop retirement, full wiring), and finishes by attaching the trace to `RunResult` and re-asserting kernel independence. Language: Python, tested with `uv run pytest` (property tests via `hypothesis`, min. 100 examples). Providers / HTTP / MCP are mocked.

## Tasks

- [x] 1. Observability event foundation
  - [x] 1.1 Implement the `AgentEvents` protocol and event types
    - Create `loomable/agent/events.py` with `Event` (kind, `t`, `duration_ms`, `tokens_in`, `tokens_out`, `attributes`), the `AgentEvents` Protocol, `NoOpEvents` (zero-overhead default), and `JSONTracer` (appends one JSON line per event, accumulates a `trace` list, exposes `trace` property)
    - Use OpenTelemetry GenAI semantic-convention names for `Event.attributes` keys
    - _Requirements: 11.5, 11.6_
  - [ ]* 1.2 Write unit tests for the tracers
    - Assert `NoOpEvents.emit` records nothing and `JSONTracer` accumulates events in emission order and serializes valid JSON lines
    - _Requirements: 11.5, 11.6_

- [x] 2. RunContext seam
  - [x] 2.1 Implement `RunContext` and `StopReason`
    - Create `loomable/agent/context.py` (edge module, distinct from kernel `ContextManager`) with `StopReason` (kind + detail + `STOP_*` constants) and `RunContext` (events emitter defaulting to `NoOpEvents`, `max_steps`, `token_budget`, `loop_repeat_threshold`, cooperative `cancel`/`cancelled`, `tick_step`, `add_tokens`/`token_budget_exceeded`, `record_call`/`is_looping`, `elapsed`)
    - Implement `_signature(tool_name, args)` canonicalizing args via `json.dumps(args, sort_keys=True, default=str)` and hashing `f"{tool_name}:{canonical}"` with `hashlib.sha1`
    - _Requirements: 3.1, 4.2, 4.5_
  - [ ]* 2.2 Write property/unit tests for `RunContext`
    - **Property 6 (partial): signature determinism** — identical `(tool_name, args)` produce the same signature and monotonically increasing counts; `is_looping` becomes true exactly at `loop_repeat_threshold`
    - Test `tick_step` reports "may continue" while under `max_steps` and `add_tokens`/`token_budget_exceeded` track cumulative usage; fresh context defaults to a no-op emitter
    - **Validates: Requirements 3.1, 4.2, 4.5**

- [x] 3. Provider error classification (resilience workstream A.1)
  - [x] 3.1 Implement transient/permanent provider errors
    - Create `loomable/providers/errors.py` with `TransientProviderError` (carries `status_code`, `retry_after`) and `PermanentProviderError` (carries `status_code`), both subclassing the kernel `ModelProviderError`
    - _Requirements: 1.1, 1.2, 1.9, 15.2_
  - [ ]* 3.2 Write unit tests for error subclassing
    - Assert both new errors are instances of the kernel `ModelProviderError` and preserve `provider_id`
    - _Requirements: 1.9, 15.2_
  - [x] 3.3 Classify HTTP errors in provider `complete()`
    - Add `_classify_http_error` and `_parse_retry_after` and apply them in the OpenAI/Azure and Anthropic providers so timeouts/connect/read/remote-protocol and 429/5xx map to `TransientProviderError` (parsing `Retry-After`) and other 4xx/auth/policy map to `PermanentProviderError`, keeping `ModelProviderError` as the wrapped base
    - _Requirements: 1.1, 1.2, 1.5_
  - [ ]* 3.4 Write unit tests for HTTP classification
    - Use `httpx.MockTransport` to drive 429/500/503/timeout/connection-reset and 400/401/403 responses and assert the correct transient vs permanent classification and `retry_after` parsing
    - _Requirements: 1.1, 1.2, 1.5_

- [x] 4. ResilientModel wrapper (resilience workstream A.2)
  - [x] 4.1 Implement `RetryPolicy`, `_backoff_delay`, and `ResilientModel`
    - Create `loomable/providers/resilient.py` implementing the kernel `ModelProvider` protocol; retry only transient errors up to `max_attempts`, fail fast on permanent errors after one attempt, bound each attempt with `asyncio.wait_for(per_call_timeout)`, and honor `Retry-After` when larger than the computed backoff
    - Implement full-jitter `_backoff_delay(attempt, policy, retry_after)` returning `random.uniform(0, min(max_delay, base*multiplier**attempt))`, never below `retry_after`
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_
  - [ ]* 4.2 Write property test for backoff bounds
    - **Property 2: Backoff is bounded and jittered** — for any attempt index and policy, `_backoff_delay` returns a value in `[0, min(max_delay, base*multiplier**attempt)]` and never below a provided `retry_after`
    - **Validates: Requirements 1.5, 1.6**
  - [ ]* 4.3 Write property test for retry vs fail-fast
    - **Property 1: Transient errors are retried, permanent errors fail fast** — `k < max_attempts` transient failures then success returns after exactly `k` retries; a permanent error raises after exactly one attempt
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
  - [ ]* 4.4 Write property test for transparency
    - **Property 20: Resilient wrapper is a transparent ModelProvider** — on a non-failing provider, `complete` returns exactly the inner `ModelResponse` with one underlying call
    - **Validates: Requirements 1.8**

- [x] 5. Checkpoint - resilience layer
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Tool idempotency declaration
  - [x] 6.1 Add an `idempotent` flag to tools
    - Add `idempotent: bool = True` to `FunctionTool` and the `@tool(...)` decorator in `loomable/agent/tools.py` so side-effecting tools can declare `idempotent=False`
    - _Requirements: 3.5_
  - [ ]* 6.2 Write unit test for the idempotency flag
    - Assert `@tool(idempotent=False)` and the `FunctionTool` field propagate correctly and default to `True`
    - _Requirements: 3.5_

- [x] 7. Loop detection, stop reasons, and RunContext threading (workstream A.4, control flow)
  - [x] 7.1 Thread `RunContext` through the run path and add loop control
    - In `loomable/agent/builder.py`, give `_run_single` / `_run_tool_loop` an optional `ctx: RunContext | None = None` (defaulting to a fresh no-op context); before dispatching each proposed call, `record_call` its signature and stop with `StopReason.LOOP_DETECTED` (without dispatching) when the count reaches `loop_repeat_threshold`
    - On reaching `max_tool_iterations`, stop with `StopReason.MAX_ITERATIONS` and re-invoke the model once with a "you must answer now, no tools" system nudge; check the cancel flag and step/token budgets at each loop boundary, stopping with `CANCELLED` / `STEP_BUDGET` / `TOKEN_BUDGET`; record the chosen `StopReason` in `RunResult.metadata["stop_reason"]` and emit a `loop_stop` event; exclude `idempotent=False` tools from any auto-retry
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4_
  - [ ]* 7.2 Write property test for loop detection
    - **Property 6: Loop detection short-circuits with an explicit reason** — a scripted model repeating the same `(tool_name, args)` `loop_repeat_threshold` times stops with `LOOP_DETECTED` and never dispatches the repeated call
    - **Validates: Requirements 3.1, 3.2, 3.4**
  - [ ]* 7.3 Write property test for the iteration cap
    - **Property 7: Iteration cap produces a final answer with a stop reason** — a model that always requests tools terminates at `max_tool_iterations` with `MAX_ITERATIONS` recorded and a non-tool final response
    - **Validates: Requirements 3.3, 3.4, 4.3**
  - [ ]* 7.4 Write property test for cooperative cancellation
    - **Property 8: Cancellation is cooperative and prompt** — a cancelled `RunContext` stops at the next loop boundary with `CANCELLED` and issues no further model or tool calls
    - **Validates: Requirements 4.1, 4.2**

- [x] 8. Bounded tool dispatch (workstream A.3)
  - [x] 8.1 Add per-tool timeout and concurrency cap to gated dispatch
    - In `loomable/agent/builder.py`, add `_dispatch_with_limits` and route `dispatch_tools_gated` through it: bound each kernel `ToolRuntime` call with `asyncio.wait_for(per_tool_timeout)` and cap parallelism with an `asyncio.Semaphore`, turning a timeout into a `ToolOutcome` carrying a `ToolError` naming the tool (fed back to the model, never blind-retried) while sibling calls still complete
    - _Requirements: 2.1, 2.2, 2.3, 2.4_
  - [ ]* 8.2 Write property test for per-tool timeout
    - **Property 4: Per-tool timeout yields a fed-back error, not a hang** — a tool exceeding `tool_timeout` produces a `ToolOutcome` error naming the tool while sibling calls still complete
    - **Validates: Requirements 2.1, 2.2**
  - [ ]* 8.3 Write property test for the concurrency cap
    - **Property 5: Concurrency cap is respected** — for a batch of N calls with `tool_concurrency = c`, in-flight invocations never exceed `c` (verified with an instrumented semaphore counter); also covers **Property 3: Retry never touches tools** by asserting each tool is invoked exactly once
    - **Validates: Requirements 2.3, 2.4**

- [x] 9. Checkpoint - control flow and dispatch
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Model-based summarizer (workstream B.1)
  - [x] 10.1 Implement `LLMSummarizer`
    - Create `loomable/agent/summarize.py` implementing the kernel `Summarizer.summarize(turns) -> StructuredSummary` contract via a model call (rendering turns to a prompt and parsing into `StructuredSummary`), with a synchronous bridge for the model call and a kernel-style regex fallback when the model call fails
    - _Requirements: 5.1, 5.2, 5.3_
  - [ ]* 10.2 Write property test for the summarizer contract
    - **Property 9: LLMSummarizer honors the kernel contract** — for any non-empty turn list, `summarize` returns a `StructuredSummary` whose `covers_steps` spans the turns' step range with positive `tokens`, and a mocked model failure still yields a valid fallback summary
    - **Validates: Requirements 5.1, 5.2, 5.3**

- [x] 11. Pinned facts and rolling window (workstream B.2)
  - [x] 11.1 Implement pinned facts in the harness
    - In `loomable/agent/builder.py`, add `pinned_steps: set[int]` and `pin_fact(text)` to `BuiltAgent`; in `_persist_session`, exclude any `Turn` whose `step in pinned_steps` from the compaction overflow slice (rolling window + pinned facts) and always replay pinned turns in `_memory_prefix`
    - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - [ ]* 11.2 Write property test for compaction with pinned facts
    - **Property 10: Compaction preserves pinned facts and recent window** — after a run on a conversation exceeding the threshold, retained raw turns are at most the window size plus all pinned turns, every pinned turn remains in working memory, and a summary covers the compacted non-pinned turns
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 13.3**

- [x] 12. Durable structured notes (workstream B.3)
  - [x] 12.1 Implement `NoteStore` and the `memory` tool
    - Create `loomable/agent/notes.py` with `Note`, `NoteStore` (over the kernel `LongTermStore`: `write` upsert-by-id, `read`, `list`, `delete`, `recall` vector search), and `make_memory_tool(store)` exposing a single `memory` tool with `action ∈ {write, read, list, delete, recall}`
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  - [ ]* 12.2 Write property test for note upsert/delete
    - **Property 11: NoteStore upserts, never duplicates** — for any sequence of `write(note_id, text)` calls over a fake `LongTermStore`, `list()` contains exactly one note per distinct `note_id` with the latest text, and `delete(note_id)` removes it
    - **Validates: Requirements 7.1, 7.2**

- [x] 13. Reasoning tools (workstream C.1, C.2)
  - [x] 13.1 Implement the `think` and `plan` tools
    - Create `loomable/agent/reasoning.py` with `make_think_tool()` (no-side-effect scratchpad returning its `thought`, `idempotent=True`) and `make_plan_tool(agent)` (invokes `AutoPlan(agent, max_steps).run(...)` — plan → parallel subagents → synthesize — returning the synthesized string)
    - _Requirements: 8.1, 8.2, 9.1, 9.2_
  - [ ]* 13.2 Write property test for the `think` tool
    - **Property 12: `think` tool is an identity scratchpad** — for any string, the tool returns it as the result content and causes no change to memory, notes, or control flow
    - **Validates: Requirements 8.1, 8.2**
  - [ ]* 13.3 Write integration test for the `plan` tool
    - **Property 13: `plan` tool escalates to fan-out and synthesizes** — with a scripted provider and fake stores, invoking the tool runs AutoPlan (plan → subagents → synthesize) and returns a single synthesized string
    - **Validates: Requirements 9.1, 9.2**

- [x] 14. Complexity routing (workstream C.3)
  - [x] 14.1 Implement `ComplexityRouter`
    - Create `loomable/agent/routing.py` with `RunStrategy` (SINGLE / TOOL_LOOP / PLAN) and a heuristic (stdlib-only) `ComplexityRouter.classify(agent_input, has_tools)` using token length, question count, conjunction/step cues, and tool presence, with an optional injected model-based classifier
    - _Requirements: 10.1_
  - [ ]* 14.2 Write property test for router selection
    - **Property 14: Complexity router selects a valid strategy and defaults safely** — `classify` always returns a `RunStrategy`; assert the default (router unset) chooses the loop iff tools exist else single-shot
    - **Validates: Requirements 10.1, 10.3**

- [x] 15. Checkpoint - memory and reasoning
  - Ensure all tests pass, ask the user if questions arise.

- [x] 16. Token-bounded context assembly (workstream E.1)
  - [x] 16.1 Implement `_bound_messages` and apply it before each model call
    - In `loomable/agent/builder.py`, add `_bound_messages(messages, budget)` that feeds messages as `ContextItem`s into the kernel `ContextManager` (pinning system/instructions/tool-schemas/pinned facts), runs evict-then-admit against the token budget with a cheap estimator, and reassembles kept messages in order; call it immediately before each `ResilientModel.complete` / `router.route` in `_run_single` / `_run_tool_loop` and add reported usage via `ctx.add_tokens`
    - _Requirements: 13.1, 13.2, 13.3, 13.4_
  - [ ]* 16.2 Write property test for context bounding
    - **Property 17: Context bounding never exceeds the token budget and keeps pinned items** — for any messages and budget, returned messages have estimated total tokens ≤ budget (when satisfiable by evicting non-pinned items) and every pinned message is retained
    - **Validates: Requirements 13.1, 13.2, 13.3, 13.4**

- [x] 17. Two-loop resolution (workstream E.2)
  - [x] 17.1 Stop implicitly constructing the kernel `AgentLoop`
    - In `loomable/agent/builder.py`, make `build()` no longer construct an `AgentLoop`; change `BuiltAgent.loop` to `AgentLoop | None` defaulting to `None`, and document the kernel `AgentLoop` as the autonomous/batch loop distinct from the interactive high-level harness (kernel source untouched)
    - _Requirements: 14.1, 14.2_
  - [ ]* 17.2 Write property test for the absent implicit loop
    - **Property 18: No implicit AgentLoop on the high-level path** — for any agent built via the high-level builder, `BuiltAgent.loop` is `None` and a full `arun` completes without constructing or invoking a kernel `AgentLoop`
    - **Validates: Requirements 14.1, 14.2**

- [x] 18. Run-result trace and full builder wiring
  - [x] 18.1 Extend `RunResult` with a trace field
    - In `loomable/agent/run.py`, add `trace: list[Event] = field(default_factory=list)` to `RunResult` (additive, defaulted) so existing callers are unaffected
    - _Requirements: 12.3_
  - [x] 18.2 Wire the harness features into the builder and run path
    - In `loomable/agent/builder.py`, add the new `BuiltAgent` / builder fields (`resilience: RetryPolicy | None`, `tool_timeout`, `tool_concurrency`, `events`, `complexity_router`, `note_store`, `loop_repeat_threshold`, `use_llm_summarizer`/`summarizer`); wrap each provider impl in `ResilientModel` when `resilience` is set before constructing the `ModelInterface`; build a `RunContext` per run and consult the `complexity_router` before mode selection (SINGLE→`_run_single`, TOOL_LOOP→`_run_tool_loop`, PLAN→`AutoPlan`); register the `think` / `plan` / `memory` tools when configured; emit `run_start` / `model_call` / `tool_call` / `compaction` / `tier_substitution` / `run_end` events at their points; and copy a recording tracer's accumulated events onto `RunResult.trace`
    - _Requirements: 4.5, 10.2, 10.3, 11.1, 11.2, 11.3, 11.4, 12.1_
  - [ ]* 18.3 Write property test for event ordering
    - **Property 15: Events are emitted in a well-formed order** — for any run, the sequence starts with `run_start`, ends with `run_end`, every `model_call` end follows its start, and durations/token counts are non-negative
    - **Validates: Requirements 11.1, 11.2**
  - [ ]* 18.4 Write property test for trace fidelity
    - **Property 16: Trace faithfully records model and tool calls** — for any run with a recording tracer, the count of `model_call` events equals model invocations and `tool_call` events equals gated dispatch batches, and `RunResult.trace` contains them all
    - **Validates: Requirements 11.3, 12.1, 12.2**

- [ ] 19. Kernel independence guarantee
  - [ ]* 19.1 Extend the kernel independence test
    - **Property 19: Kernel remains independent** — extend `tests/unit/test_kernel_independence.py` to assert the `loomable.kernel` tree imports no module from `loomable.agent` / `content` / `serve` / `providers`, and that every new error is an instance of the kernel `LoomableError`; confirm no new mandatory runtime dependency beyond the stdlib and existing deps
    - **Validates: Requirements 15.1, 15.2, 15.3**

- [x] 20. Final checkpoint - full suite green
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test sub-tasks and can be skipped for a faster MVP; core implementation sub-tasks are never optional.
- Each task references specific requirement clauses for traceability, and every property test task references its property number from the design's Correctness Properties section.
- All new modules live under `loomable.agent` / `loomable.providers`; `loomable.kernel` is never modified and its independence test must stay green after every task.
- The six `builder.py` integration tasks (7.1, 8.1, 11.1, 16.1, 17.1, 18.2) are intentionally sequenced across separate waves to avoid write conflicts on the same file.
- Providers / HTTP / MCP are mocked; property tests use `hypothesis` (min. 100 examples) and the suite runs with `uv run pytest`.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "3.1", "6.1", "10.1", "12.1", "14.1", "18.1"] },
    { "id": 1, "tasks": ["2.1", "3.3", "4.1", "13.1", "1.2", "3.2", "6.2", "10.2", "12.2", "14.2"] },
    { "id": 2, "tasks": ["7.1", "2.2", "3.4", "4.2", "4.3", "4.4", "13.2", "13.3"] },
    { "id": 3, "tasks": ["8.1", "7.2", "7.3", "7.4"] },
    { "id": 4, "tasks": ["16.1", "8.2", "8.3"] },
    { "id": 5, "tasks": ["11.1", "16.2"] },
    { "id": 6, "tasks": ["17.1", "11.2"] },
    { "id": 7, "tasks": ["18.2", "17.2"] },
    { "id": 8, "tasks": ["18.3", "18.4", "19.1"] }
  ]
}
```
