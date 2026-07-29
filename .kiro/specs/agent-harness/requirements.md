# Requirements Document

## Introduction

The `agent-harness` feature hardens the high-level `loomable.agent` run path for production without modifying `loomable.kernel`. Today `BuiltAgent.arun` runs a clean ReAct loop but lacks transport resilience, model-based summarization, runtime plan escalation, structured observability, and it constructs a kernel `AgentLoop` in `build()` that the high-level path never runs. This feature closes those five gaps.

All behavior is additive and lives in the edge layers (`loomable.agent`, `loomable.providers`, and small new edge modules). The kernel is never modified, so the existing import-independence guarantee must continue to hold. The work is unified by one new seam, `RunContext`, threaded through the run path to carry an event emitter, step and token budgets, a cooperative cancel flag, and a tool-call-signature history. The framework stays lean: retries, backoff, and jitter use only the standard library, and observability ships a no-op default with an optional recording tracer.

## Glossary

- **Harness**: The high-level run path (`BuiltAgent.arun` and the `_run_single` / `_run_tool_loop` functions) that this feature hardens.
- **Run_Context**: The per-run edge object threaded through the Harness carrying the event emitter, step budget, token budget, cooperative cancel flag, and tool-call-signature history.
- **Stop_Reason**: A value recording why the Harness loop terminated (one of: final, max_iterations, loop_detected, cancelled, step_budget, token_budget, error).
- **Model_Provider**: A component implementing the kernel `ModelProvider` protocol (an `async complete` method).
- **Resilient_Model**: An edge wrapper around a Model_Provider that adds per-call timeout and backoff-with-jitter retry for transient transport failures.
- **Retry_Policy**: The configuration controlling Resilient_Model retry behavior (max attempts, base delay, max delay, multiplier, jitter fraction, per-call timeout).
- **Transient_Provider_Error**: An edge error (subclass of the kernel `ModelProviderError`) representing a provider failure that may succeed on retry (429, 5xx, timeout, connection reset).
- **Permanent_Provider_Error**: An edge error (subclass of the kernel `ModelProviderError`) representing a provider failure that will not succeed on retry (4xx, auth, content policy).
- **Tool_Dispatcher**: The edge gated-dispatch layer that wraps kernel `ToolRuntime` calls with per-tool timeout and a concurrency cap.
- **Tool_Outcome**: The result object surfaced for a tool call, which may carry a `ToolError`.
- **LLM_Summarizer**: An edge summarizer implementing the kernel `Summarizer.summarize(turns) -> StructuredSummary` contract using a model call, with a regex fallback.
- **Note_Store**: An edge store of structured, deduplicated notes backed by the kernel `LongTermStore`.
- **Memory_Tool**: A single tool exposing Note_Store actions (write, read, list, delete, recall) to the model.
- **Think_Tool**: A no-side-effect scratchpad tool that returns its input thought as its result.
- **Plan_Tool**: A tool that invokes `AutoPlan` (plan → parallel subagents → synthesize) and returns a synthesized answer.
- **Complexity_Router**: An opt-in pre-flight classifier that selects a run strategy (single-shot, tool-loop, or plan).
- **Run_Strategy**: The classification produced by the Complexity_Router (SINGLE, TOOL_LOOP, or PLAN).
- **Agent_Events**: The observability protocol whose implementations receive typed `Event` objects.
- **Tracer**: An Agent_Events implementation that records events (e.g., the in-box JSON/console tracer).
- **Event**: A typed observability record with a kind, timestamp, optional duration and token counts, and an attributes map.
- **Run_Result**: The object returned by a run, extended with a `trace` list and a `stop_reason` entry in its metadata.
- **Context_Bounder**: The edge function that feeds messages through the kernel `ContextManager` to evict/admit against a token budget before each model call.
- **Builder**: The high-level agent builder (`build()`) that assembles a `BuiltAgent`.
- **Kernel**: The `loomable.kernel` package tree, which this feature must not modify.

## Requirements

### Requirement 1: Transport resilience for model calls

**User Story:** As an operator running an agent in production, I want transient model-provider failures to be retried automatically, so that temporary network or rate-limit issues do not fail an entire run.

#### Acceptance Criteria

1. WHEN a Model_Provider call fails with a 429 status, a 5xx status, a timeout, or a connection reset, THE Resilient_Model SHALL classify the failure as a Transient_Provider_Error.
2. WHEN a Model_Provider call fails with a 4xx status other than 429, an authentication error, or a content-policy error, THE Resilient_Model SHALL classify the failure as a Permanent_Provider_Error.
3. WHILE the number of attempts is below the Retry_Policy maximum, WHEN a Model_Provider call raises a Transient_Provider_Error, THE Resilient_Model SHALL retry the call.
4. WHEN a Model_Provider call raises a Permanent_Provider_Error, THE Resilient_Model SHALL raise the error after exactly one attempt without retrying.
5. WHEN a Transient_Provider_Error carries a retry-after value, THE Resilient_Model SHALL wait at least the retry-after duration before the next attempt.
6. WHEN retrying between transient failures, THE Resilient_Model SHALL wait a backoff delay in the range from 0 to the smaller of the maximum delay and the base delay multiplied by the multiplier raised to the attempt index.
7. WHEN each attempt is made, THE Resilient_Model SHALL bound that attempt with the Retry_Policy per-call timeout.
8. WHEN a Model_Provider call succeeds on the first attempt, THE Resilient_Model SHALL return the inner provider's response unchanged after exactly one underlying call.
9. THE Transient_Provider_Error and Permanent_Provider_Error SHALL each be an instance of the kernel `ModelProviderError`.

### Requirement 2: Bounded tool dispatch

**User Story:** As an operator, I want tool execution to be time-bounded and concurrency-limited, so that a single slow or hanging tool cannot stall a run or overwhelm downstream systems.

#### Acceptance Criteria

1. IF a tool execution exceeds the configured tool timeout, THEN THE Tool_Dispatcher SHALL produce a Tool_Outcome carrying a `ToolError` that names the tool.
2. WHEN one tool call in a batch times out, THE Tool_Dispatcher SHALL allow the remaining tool calls in that batch to complete.
3. WHILE dispatching a batch with a configured concurrency cap, THE Tool_Dispatcher SHALL limit the number of simultaneously in-flight tool invocations to at most the configured cap.
4. WHEN a tool returns an error or times out, THE Harness SHALL feed the error back into the model conversation and SHALL invoke that tool exactly once for that call.

### Requirement 3: Loop detection and explicit stop reasons

**User Story:** As an operator, I want the agent to detect no-progress loops and iteration limits and stop with a clear reason, so that runs terminate predictably and return usable results.

#### Acceptance Criteria

1. WHEN a proposed tool call is about to be dispatched, THE Run_Context SHALL record a canonical signature of the tool name and arguments and return the updated repeat count for that signature.
2. WHEN the repeat count for a tool-call signature reaches the loop-repeat threshold, THE Harness SHALL stop with Stop_Reason loop_detected and SHALL NOT dispatch the repeated call.
3. WHEN the number of iterations reaches the maximum tool iterations, THE Harness SHALL stop with Stop_Reason max_iterations and SHALL re-invoke the model once with a no-tools instruction so the output is a non-tool final response.
4. WHEN the Harness loop terminates, THE Harness SHALL record the selected Stop_Reason in the Run_Result metadata and SHALL emit a loop_stop event carrying that Stop_Reason.
5. WHERE a tool is declared non-idempotent, THE Harness SHALL exclude that tool from any automatic retry.

### Requirement 4: Cooperative cancellation and budgets

**User Story:** As a developer embedding the agent, I want to cancel a run and cap its steps and tokens, so that I can bound cost and stop work that is no longer needed.

#### Acceptance Criteria

1. WHEN the Run_Context cancel flag is set, THE Harness SHALL stop at the next loop boundary with Stop_Reason cancelled and SHALL issue no further model or tool calls.
2. WHILE the number of used steps is below the step budget, THE Run_Context SHALL report that the run may continue.
3. WHEN the step budget is exhausted, THE Harness SHALL stop with Stop_Reason step_budget.
4. WHEN the accumulated token usage exceeds the token budget, THE Harness SHALL stop with Stop_Reason token_budget.
5. THE Run_Context SHALL be created fresh for each run and SHALL default to a no-op event emitter so existing callers are unaffected.

### Requirement 5: Model-based summarization

**User Story:** As a developer running long-horizon agents, I want conversation history compacted by a model-based summarizer, so that context stays within budget while preserving meaning better than a regex summary.

#### Acceptance Criteria

1. WHEN given a non-empty list of turns, THE LLM_Summarizer SHALL return a `StructuredSummary` whose covered-steps range spans the turns' step range and whose token count is positive.
2. IF the model call for summarization fails, THEN THE LLM_Summarizer SHALL return a valid fallback `StructuredSummary` so compaction does not break the run.
3. THE LLM_Summarizer SHALL implement the same `summarize(turns) -> StructuredSummary` contract as the kernel `Summarizer` so it drops into the existing compaction path.

### Requirement 6: Pinned facts and rolling window

**User Story:** As a developer, I want to pin important facts so they are never summarized away, so that precise values survive compaction.

#### Acceptance Criteria

1. WHEN a fact is pinned, THE Harness SHALL append a pinned turn and record its step as never eligible for compaction.
2. WHEN compaction runs on a conversation exceeding the compaction threshold, THE Harness SHALL retain at most the rolling window size of raw turns plus all pinned turns.
3. WHEN compaction runs, THE Harness SHALL retain every pinned turn in the working memory and SHALL produce a summary covering the compacted non-pinned turns.
4. WHEN assembling the memory prefix for a run, THE Harness SHALL replay every pinned turn.

### Requirement 7: Durable structured notes

**User Story:** As a developer building cross-session agents, I want the agent to take durable, deduplicated notes, so that lessons persist and are not duplicated.

#### Acceptance Criteria

1. WHEN a note is written with a given note identifier, THE Note_Store SHALL upsert the note so that listing contains exactly one note per distinct identifier with the latest text.
2. WHEN a note is deleted by identifier, THE Note_Store SHALL remove that note.
3. WHEN a recall query is issued, THE Note_Store SHALL return the notes most relevant to the query.
4. THE Memory_Tool SHALL expose the write, read, list, delete, and recall actions of the Note_Store to the model as a single tool.
5. THE Note_Store SHALL persist notes through the kernel `LongTermStore` without modifying the kernel.

### Requirement 8: Scratchpad reasoning tool

**User Story:** As a developer, I want the model to have a scratchpad tool for intermediate reasoning, so that policy adherence improves over long tool chains without side effects.

#### Acceptance Criteria

1. WHEN the Think_Tool is invoked with a thought string, THE Think_Tool SHALL return that string as the tool result content.
2. WHEN the Think_Tool is invoked, THE Think_Tool SHALL cause no change to memory, notes, or control flow.

### Requirement 9: Runtime escalation to fan-out

**User Story:** As a developer, I want the model to escalate a complex task into a plan-and-fan-out at runtime, so that hard tasks can be decomposed on demand without a separate graph engine.

#### Acceptance Criteria

1. WHEN the Plan_Tool is invoked with a task, THE Plan_Tool SHALL run `AutoPlan` to plan, execute parallel subagents, and synthesize the results.
2. WHEN `AutoPlan` completes, THE Plan_Tool SHALL return a single synthesized string result as the tool result.

### Requirement 10: Complexity routing

**User Story:** As a developer, I want an opt-in classifier that chooses between single-shot, tool-loop, and plan strategies, so that simple inputs skip unnecessary machinery and complex inputs escalate.

#### Acceptance Criteria

1. WHEN the Complexity_Router classifies any input, THE Complexity_Router SHALL return a Run_Strategy of SINGLE, TOOL_LOOP, or PLAN.
2. WHERE a Complexity_Router is configured, WHEN a run starts, THE Harness SHALL select the run path corresponding to the returned Run_Strategy before mode selection.
3. WHERE no Complexity_Router is configured, WHEN a run starts, THE Harness SHALL choose the tool-loop path if tools exist and the single-shot path otherwise.

### Requirement 11: Structured observability

**User Story:** As an operator, I want structured events emitted throughout a run, so that I can trace and diagnose agent behavior in production.

#### Acceptance Criteria

1. WHEN a run executes, THE Harness SHALL emit a run_start event first and a run_end event last.
2. WHEN a model call is made, THE Harness SHALL emit a model_call event whose end follows its start and SHALL include non-negative timing and token counts.
3. WHEN a gated dispatch batch executes, THE Harness SHALL emit a tool_call event.
4. WHEN a summary is produced during compaction, THE Harness SHALL emit a compaction event.
5. WHERE no Tracer is configured, THE Harness SHALL emit events to a no-op emitter that incurs no recording overhead.
6. THE Event attributes SHALL use OpenTelemetry GenAI semantic-convention names.

### Requirement 12: Trace on run result

**User Story:** As a developer, I want the full event trace attached to the run result, so that I can inspect what happened after a run completes.

#### Acceptance Criteria

1. WHERE a recording Tracer is active, WHEN a run completes, THE Harness SHALL copy the accumulated events onto the Run_Result trace.
2. WHEN a run completes with a recording Tracer, THE number of model_call events in the Run_Result trace SHALL equal the number of model invocations and the number of tool_call events SHALL equal the number of gated dispatch batches.
3. THE Run_Result SHALL provide a trace list that defaults to empty so the addition does not break existing callers.

### Requirement 13: Token-bounded context assembly

**User Story:** As a developer, I want the high-level loop to bound each model request against a token budget, so that requests stay within limits while preserving essential context.

#### Acceptance Criteria

1. WHEN assembling messages for a model call, THE Context_Bounder SHALL feed the messages through the kernel `ContextManager` and apply evict-then-admit against the token budget.
2. WHEN the token budget is satisfiable by evicting non-pinned items, THE Context_Bounder SHALL return messages whose estimated total token count is at most the budget.
3. WHEN bounding messages, THE Context_Bounder SHALL retain every pinned message including system instructions, tool schemas, and pinned facts.
4. WHEN a model response reports token usage, THE Harness SHALL add that usage to the Run_Context cumulative token count.

### Requirement 14: Single high-level harness loop

**User Story:** As a maintainer, I want the high-level path to be the single production harness without an implicitly constructed kernel loop, so that there is no ambiguity about which loop runs.

#### Acceptance Criteria

1. WHEN an agent is built via the high-level Builder, THE Builder SHALL leave the built agent's kernel loop reference as none.
2. WHEN a run executes on an agent built via the high-level Builder, THE Harness SHALL complete without constructing or invoking a kernel `AgentLoop`.

### Requirement 15: Kernel independence

**User Story:** As a maintainer, I want the kernel to remain independent of the edge layers, so that the kernel stays reusable and the existing independence guarantee holds.

#### Acceptance Criteria

1. THE Kernel package tree SHALL import no module from `loomable.agent`, `loomable.content`, `loomable.serve`, or `loomable.providers`.
2. THE new errors introduced by this feature SHALL each be an instance of the kernel `LoomableError`.
3. THE feature SHALL add no mandatory runtime dependency beyond the standard library and existing dependencies.
