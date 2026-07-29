# Implementation Plan: loomable

## Overview

This plan builds `loomable` incrementally in Python (managed with `uv`), following the kernel + capabilities architecture. Work begins with project scaffolding and the stable Kernel contracts (abstract interfaces, data models, error taxonomy), then layers in each subsystem: model interface/router, memory/context/summarizer, stores, tool runtime, extension edge (Skills, MCP, API tools, retrievers), planner, subagents, and finally the agent loop with guardrails that wires everything together. Each subsystem lands with its property-based tests (via `hypothesis`) mapped to the 23 correctness properties, so correctness is validated close to implementation. The final tasks integrate the loop end-to-end and validate the extension model with an example Domain_Skill.

Property tests use `hypothesis` with a minimum of 100 examples per property and are tagged `# Feature: loomable, Property {n}`. Tests run via `uv run pytest` (single run, not watch mode).

## Tasks

- [x] 1. Set up project structure and Kernel contracts
  - [x] 1.1 Initialize the uv-managed Python project and test scaffolding
    - Create the `uv`-managed project (`pyproject.toml`) with dependencies: `hypothesis`, `pytest`, `pytest-asyncio`, an HTTP client (e.g. `httpx`), and MCP client support
    - Create the package layout `loomable/kernel/` and test layout `tests/properties/`, `tests/unit/`, `tests/integration/`, `tests/benchmarks/`
    - Configure `pytest` (asyncio mode) so `uv run pytest` runs the suite once
    - _Requirements: 20.1, 20.2_

  - [x] 1.2 Define core data models
    - Implement `ToolCall`, `ToolOutcome`, `ToolResult`, `Turn`, `StructuredSummary`, `ContextItem`, `ContextWindow`, `Session`, `LoopState`, `AgentConfig`, `OnboardingRequest`, and the `ExtensionMechanism` enum as described in the design Data Models section
    - Encode invariants in the types (e.g. `ToolOutcome` carries exactly one of result/error; pinned flag for system/schema items)
    - _Requirements: 1.1_

  - [x] 1.3 Define abstract Kernel contracts
    - Declare the stable abstract interfaces `Tool`, `ModelProvider`, `MemoryBackend`, `VectorBackend`, `Retriever`, `Skill` in `loomable/kernel/` with no imports of any concrete/example module
    - _Requirements: 1.1, 1.3_

  - [x] 1.4 Implement the error taxonomy
    - Implement all error types from the design Error Taxonomy (`UnsupportedExtensionError`, `ModelProviderError`, `MCPConnectionError`, `MCPToolError`, `SkillLoadError`, `ScriptToolError`, `APIToolError`, `APIToolTimeoutError`, `MemoryBackendError`, `SessionNotFoundError`, `PlanningModelError`, `SubagentError`, `GuardrailViolation`), each carrying the identifying field(s) noted in the taxonomy
    - _Requirements: 2.4, 4.4, 4.5, 5.4, 5.5, 6.3, 6.4, 7.6, 8.6, 12.4, 14.4, 15.4, 16.5, 18.3_

- [x] 2. Implement the Extension Registry and lazy loading
  - [x] 2.1 Implement ExtensionRegistry onboarding and lazy resolution
    - Implement `ExtensionRegistry.onboard()` accepting only `SKILL`, `MCP_SERVER`, `API_TOOL`; reject any other mechanism (including `kernel_modification`) with `UnsupportedExtensionError` naming the supported mechanisms
    - Implement `enabled_extensions()` and `resolve_tool()` so expensive resources materialize lazily on first use, and only `enabled` extensions are eligible
    - Share immutable Kernel data (tool schemas, static system prompt, parsed config) by reference across agent instances
    - _Requirements: 1.2, 1.4, 1.5, 3.3, 19.3_

  - [x]* 2.2 Write property test for onboarding mechanism acceptance/rejection
    - **Property 1: Onboarding accepts supported mechanisms and rejects everything else**
    - **Validates: Requirements 1.2, 1.4, 19.3**

  - [x]* 2.3 Write property test for disabled-extension non-materialization
    - **Property 2: Disabled extensions are never materialized**
    - **Validates: Requirements 3.3, 1.5**

  - [x]* 2.4 Write unit test for the two advertised extension points
    - Assert the registry advertises exactly two extension points (Skills and MCP servers)
    - _Requirements: 1.1_

- [x] 3. Implement the Model Interface and Router
  - [x] 3.1 Implement provider-agnostic Model Interface
    - Implement `ModelProvider` protocol, `ModelRequest`/`ModelResponse` shapes, and `ModelInterface.invoke()` routing to the configured provider with no provider-specific agent code; unavailable provider returns `ModelProviderError` naming the provider
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x]* 3.2 Write property test for provider-agnostic invocation
    - **Property 3: Provider-agnostic invocation**
    - **Validates: Requirements 2.1, 2.2, 2.3**

  - [x] 3.3 Implement the tiered Model Router with fallback
    - Implement `ModelRouter.select_tier()` (cost/latency policy) and `fallback()`; route through the Model Interface; on unavailable tier, select a configured fallback and produce a `TierSubstitution` record naming intended and fallback tiers
    - _Requirements: 17.1, 17.2, 17.3_

  - [x]* 3.4 Write property test for tier selection and fallback
    - **Property 19: Tier selection and fallback**
    - **Validates: Requirements 17.1, 17.2, 17.3**

- [x] 4. Implement Context Manager, Summarizer, and Memory Manager
  - [x] 4.1 Implement the Context Manager with token budget and pinning
    - Implement `assemble()` (system prompt + tool schemas pinned at the head), `admit()` with evict-then-admit (evict lowest-priority non-pinned items until at/below budget before admitting; never evict pinned; refuse admission rather than evict pinned), and `current_tokens()` tracking equal to retained-item token sum
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 11.1_

  - [x]* 4.2 Write property test for token budget and pinning invariant
    - **Property 11: Context token budget and pinning invariant**
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 11.1**

  - [x]* 4.3 Write property test for static-content head placement
    - **Property 12: Static content is placed at the head of the window**
    - **Validates: Requirements 9.5**

  - [x] 4.4 Implement the Summarizer and checkpoint summarization
    - Implement `Summarizer.summarize()` producing a `StructuredSummary` preserving objectives/decisions; trigger summarization exactly when step count is a positive multiple of `Checkpoint_Interval`; store as L2; replace covered raw turns with the summary in the context window
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 11.2_

  - [x]* 4.5 Write property test for checkpoint summarization
    - **Property 13: Checkpoint summarization triggers on interval and compresses covered turns**
    - **Validates: Requirements 10.1, 10.2, 10.3, 11.2**

  - [x]* 4.6 Write unit test for summary content preservation
    - Assert a produced summary preserves scripted task objectives and decisions
    - _Requirements: 10.4_

  - [x] 4.7 Implement the Memory Manager tiers and recall
    - Implement `MemoryManager` maintaining L1 raw turns, L2 summaries/entities, L3 vector episodic; implement `record_turn()` and `recall()` returning similarity-ranked L3 items
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Short-Term, Long-Term, and Session stores
  - [x] 6.1 Implement the Short-Term Store with pluggable RDBMS backend (SQLite default)
    - Implement `ShortTermStore` over a pluggable `MemoryBackend` defaulting to SQLite; writes persist and reads return persisted state; alternative backends require no agent changes; unavailable backend returns `MemoryBackendError` naming the backend
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

  - [x]* 6.2 Write property test for short-term write/read round-trip
    - **Property 9: Short-term memory write/read round-trip**
    - **Validates: Requirements 7.1, 7.3, 7.4, 7.5**

  - [x]* 6.3 Write unit test for the default short-term backend
    - Assert SQLite is the default short-term Memory_Backend
    - _Requirements: 7.2_

  - [x] 6.4 Implement the Long-Term Store with pluggable vector backend (zvec default)
    - Implement `LongTermStore` over a pluggable `VectorBackend` defaulting to zvec; store indexes items; queries return similarity-ranked items; alternative backends require no agent changes; unavailable backend returns `MemoryBackendError` naming the backend
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [x]* 6.5 Write property test for similarity-ranked recall
    - **Property 10: Long-term recall is ranked by similarity**
    - **Validates: Requirements 8.1, 8.3, 8.4, 8.5, 11.3, 11.4**

  - [x]* 6.6 Write unit test for the default long-term backend
    - Assert zvec is the default long-term Memory_Backend
    - _Requirements: 8.2_

  - [x] 6.7 Implement the Session Store with SQLite persistence and resume
    - Implement `SessionStore.save()` and `resume()` on SQLite out of the box (no extra config); restore by id; unknown id returns `SessionNotFoundError` naming the id
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [x]* 6.8 Write property test for session persistence round-trip
    - **Property 14: Session persistence round-trip**
    - **Validates: Requirements 12.1, 12.2, 12.3**

- [x] 7. Implement the Tool Runtime and parallel dispatch
  - [x] 7.1 Implement concurrent tool dispatch with isolation and matching
    - Implement `ToolRuntime.dispatch()` using `asyncio.gather(..., return_exceptions=True)`; each `ToolOutcome` carries its originating `tool_call_id` with either a result or an isolated error; one failure does not cancel siblings
    - _Requirements: 13.1, 13.2, 13.3_

  - [x]* 7.2 Write property test for concurrent execution with overlap
    - **Property 15: Concurrent execution completes all units with overlap**
    - **Validates: Requirements 13.1, 14.3**

  - [x]* 7.3 Write property test for result-to-call matching
    - **Property 16: Results are matched to their originating calls**
    - **Validates: Requirements 13.2**

- [x] 8. Implement the extension edge: Skills, MCP, API tools, retrievers
  - [x] 8.1 Implement the Skills subsystem with progressive disclosure and script tools
    - Implement `SkillLoader.discover()`/`load()` over Anthropic-style `SKILL.md` (YAML frontmatter name/description + Markdown body); register bundled script tools as invocable `Tool`s executed in subprocess; a failing Skill yields `SkillLoadError` naming it while others continue; script execution failure yields `ScriptToolError` naming the tool
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x]* 8.2 Write property test for script-tool loading and invocation
    - **Property 8: Script tools are loaded and invocable**
    - **Validates: Requirements 5.1, 5.2, 5.3**

  - [x] 8.3 Implement the MCP client
    - Implement `MCPClient.connect()`/`list_capabilities()`/`call_tool()` implementing MCP at the tools boundary; on connect enumerate tools and data resources; failed connection yields `MCPConnectionError` naming the server while others keep operating; tool invocation error yields `MCPToolError` naming the tool
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 8.4 Implement the API tool runtime
    - Implement `APITool.invoke()` sending the configured HTTP request and returning the response; non-success status yields `APIToolError` including the HTTP status code; exceeding the configured timeout cancels the request and returns `APIToolTimeoutError`
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 8.5 Wire retrievers as MCP/API tools
    - Expose Retrievers to an Agent as MCP or HTTP API tools with no Kernel change; retriever invocation returns retrieved content; failure yields an error naming the retriever
    - _Requirements: 16.1, 16.2, 16.4, 16.5_

  - [x]* 8.6 Write property test for tool payload round-trip
    - **Property 7: Tool invocation round-trips the payload**
    - **Validates: Requirements 4.3, 6.2, 16.2**

  - [x]* 8.7 Write property test for fault isolation across independent units
    - **Property 6: Fault isolation across independent units**
    - **Validates: Requirements 4.4, 5.4, 13.3, 14.4**

  - [x]* 8.8 Write property test for error component identification
    - **Property 4: Errors identify the responsible component**
    - **Validates: Requirements 2.4, 4.5, 5.5, 7.6, 8.6, 12.4, 15.4, 16.5**

  - [x]* 8.9 Write property test for HTTP status in API errors
    - **Property 5: HTTP API errors include the status code**
    - **Validates: Requirements 6.3**

  - [x]* 8.10 Write edge-case test for API tool timeout
    - Test the timeout path against a controlled server delay, asserting `APIToolTimeoutError`
    - _Requirements: 6.4_

  - [x]* 8.11 Write integration test for the MCP boundary
    - Connect to a reference/mock MCP server, enumerate capabilities, and invoke a tool
    - _Requirements: 4.1, 4.2_

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement the Planner and Subagents
  - [x] 10.1 Implement the Planner with optional separate model
    - Implement `Planner.plan()` producing an execution plan; invoke the configured `Planning_Model` when present, otherwise the primary model; unavailable planning model yields `PlanningModelError` naming it
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

  - [x]* 10.2 Write property test for planner model selection
    - **Property 18: Planner model selection**
    - **Validates: Requirements 15.1, 15.2, 15.3**

  - [x] 10.3 Implement the Subagent Manager
    - Implement `SubagentManager.spawn()`/`run_all()`; a delegated task runs as a concurrent agent loop and returns its result to the parent keyed to the originating task; a subagent failure returns `SubagentError` naming the failed subagent while others still return results
    - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [x]* 10.4 Write property test for subagent delegation results
    - **Property 17: Subagent delegation returns each result to the parent**
    - **Validates: Requirements 14.1, 14.2**

- [x] 11. Implement the Agent Loop and guardrail harness (integration)
  - [x] 11.1 Implement the guardrail harness and verification gates
    - Implement the harness that evaluates configured guardrail rules before any tool dispatch: a violating action is never dispatched, is recorded in the blocked-actions record (`GuardrailViolation`), while non-violating actions are dispatched; implement configurable per-step verification gates that block advancement unless they pass
    - _Requirements: 18.2, 18.3, 18.4_

  - [x]* 11.2 Write property test for guardrail blocking and recording
    - **Property 21: Guardrails block and record violating actions**
    - **Validates: Requirements 18.2, 18.3**

  - [x]* 11.3 Write property test for verification gate advancement
    - **Property 22: Verification gate blocks advancement on failure**
    - **Validates: Requirements 18.4**

  - [x] 11.4 Implement the agent loop and state persistence, wiring all subsystems
    - Implement the `perceive → plan → act → observe` loop wiring Context Manager, Planner, harness, Tool Runtime, Memory Manager, and Session/loop-state persistence; persist a `LoopState` snapshot after each step so an interrupted loop resumes from the last completed step and phase
    - _Requirements: 18.1, 18.5_

  - [x]* 11.5 Write property test for loop phase ordering
    - **Property 20: Loop phase ordering**
    - **Validates: Requirements 18.1**

  - [x]* 11.6 Write property test for loop-state resumption round-trip
    - **Property 23: Loop state round-trip enables resumption**
    - **Validates: Requirements 18.5**

- [x] 12. Validate the extension model with an example Domain_Skill
  - [x] 12.1 Provide an example Domain_Skill enabled purely via configuration
    - Add a representative `Domain_Skill` under `examples/skills/`, enabled through configuration only; report unsupported capability rather than adding to the Kernel when a capability is unavailable through Skills/MCP/API tools
    - _Requirements: 19.1, 19.2, 19.3_

  - [x]* 12.2 Write test asserting Kernel package is unchanged and imports no example
    - Assert the Domain_Skill is exercised by an agent while the `loomable.kernel` package tree imports no example module
    - _Requirements: 19.1, 19.2_

- [x] 13. Performance and stack smoke checks
  - [x]* 13.1 Write benchmark smoke tests for instantiation ceilings
    - Assert agent instantiation completes within 50 ms and allocates no more than 15 MB resident for the reference configuration
    - _Requirements: 3.1, 3.2_

  - [x]* 13.2 Write test for the technical stack facts
    - Assert the framework is Python and managed with uv
    - _Requirements: 20.1, 20.2_

- [x] 14. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP.
- Each task references specific requirements (granular sub-clauses) for traceability.
- Property tests use `hypothesis` (min. 100 examples each), tagged `# Feature: loomable, Property {n}`, with providers/MCP/HTTP/vector backends mocked to keep iterations cheap.
- All 23 correctness properties from the design are covered by exactly one property-test sub-task.
- Checkpoints ensure incremental validation; run the suite once with `uv run pytest`.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "1.4"] },
    { "id": 2, "tasks": ["2.1", "3.1", "4.1", "6.1", "6.4", "6.7", "7.1", "8.4"] },
    { "id": 3, "tasks": ["2.2", "2.3", "2.4", "3.2", "3.3", "4.2", "4.3", "4.4", "4.7", "6.2", "6.3", "6.5", "6.6", "6.8", "7.2", "7.3", "8.1", "8.3", "8.9", "8.10"] },
    { "id": 4, "tasks": ["3.4", "4.5", "4.6", "8.2", "8.5", "8.11", "10.1", "10.3"] },
    { "id": 5, "tasks": ["8.6", "8.7", "8.8", "10.2", "10.4", "11.1"] },
    { "id": 6, "tasks": ["11.2", "11.3", "11.4"] },
    { "id": 7, "tasks": ["11.5", "11.6", "12.1"] },
    { "id": 8, "tasks": ["12.2", "13.1", "13.2"] }
  ]
}
```
