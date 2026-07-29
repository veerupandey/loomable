# Implementation Plan: agent-api

## Overview

This plan builds the `agent-api` feature incrementally on top of the existing `loomable` kernel, in Python (managed with `uv`), without modifying `loomable.kernel`. Work begins with the low-level multimodal content model and kernel bridging, then the high-level `Agent` builder with capability gating, and finally the two edge transports (FastAPI and MCP) plus transport-parity validation. Each subsystem lands with its property-based tests (via `hypothesis`) mapped to the 16 correctness properties, so correctness is validated close to implementation.

Property tests use `hypothesis` with a minimum of 100 examples per property and are tagged `# Feature: agent-api, Property {n}`. Tests run via `uv run pytest` (single run, not watch mode). Providers, HTTP client, and MCP client are mocked to keep iterations cheap.

## Tasks

- [x] 1. Set up packages and dependencies
  - [x] 1.1 Create the new package layout and add dependencies
    - Create packages `loomable/content/`, `loomable/agent/`, `loomable/serve/` with `__init__.py` files
    - Add dependencies via uv: `fastapi`, an ASGI server (`uvicorn`), and an HTTP test client (`httpx` already present); ensure MCP server support is available (the `mcp` package is already a dependency)
    - Create test layout `tests/unit/` (exists) and `tests/integration/` (exists) placeholders for the new modules
    - _Requirements: 10.1, 10.2_

- [x] 2. Implement the low-level multimodal content model
  - [x] 2.1 Implement MediaPart, Modality, and content constructors
    - Implement `Modality` enum and `MediaPart` (frozen) with `__post_init__` enforcing exactly one of `data`/`uri` and modality/media-type consistency; add `Text`, `Image`, `Video` constructors; raise `MediaPartError` on invalid construction
    - _Requirements: 3.1, 3.2, 3.5, 3.6_

  - [x] 2.2 Implement Message, AgentInput, and AgentOutput
    - Implement `Message`, `AgentInput` (ordered non-empty messages, `from_text`, `modalities()`), and `AgentOutput` (ordered non-empty parts, `text()`, `modalities()`)
    - _Requirements: 3.3, 3.4_

  - [x] 2.3 Implement ModelCapabilities and kernel bridging
    - Implement `ModelCapabilities` (default text-only input/output); implement `to_model_request()` mapping multimodal parts to the provider-agnostic content-array shape and `from_model_response()` rebuilding `AgentOutput` (text from content, media from `metadata["media"]`)
    - _Requirements: 4.3, 4.5, 5.2, 5.3, 5.5, 6.1, 6.2_

  - [x]* 2.4 Write property test for media part exclusivity
    - **Property 1: Media part exclusivity** — **Validates: Req 3.5**

  - [x]* 2.5 Write property test for modality/media-type consistency
    - **Property 2: Modality / media-type consistency** — **Validates: Req 3.6**

  - [x]* 2.6 Write property test for input round-trip through the model request
    - **Property 3: Input round-trip through the model request** — **Validates: Req 3.3, 4.3, 4.5**

  - [x]* 2.7 Write property test for output round-trip through the model response
    - **Property 4: Output round-trip through the model response** — **Validates: Req 5.2, 5.3, 5.5**

  - [x]* 2.8 Write property test for default text-only capabilities
    - **Property 9: Default capabilities are text-only** — **Validates: Req 6.2**

- [x] 3. Implement the high-level Agent builder and run flow
  - [x] 3.1 Implement the Agent builder with defaults and overrides
    - Implement `Agent.__init__` accepting model plus optional config and low-level overrides; implement `build()` constructing default kernel subsystems (ContextManager, MemoryManager, ToolRuntime, GuardrailHarness, Planner, SessionStore, AgentLoop) for any not supplied, using supplied primitives when present; raise `AgentConfigError(field)` on missing/invalid required fields
    - _Requirements: 1.1, 1.2, 1.3, 1.6, 1.7, 2.1, 2.2, 2.3_

  - [x] 3.2 Implement BuiltAgent run/stream with capability gating
    - Implement `BuiltAgent.arun()`/`astream()` and `Agent.run/arun/astream` wrappers; wrap bare-string input via `AgentInput.from_text`; validate input modalities against `capabilities.input` before invoking the provider and output modalities against `capabilities.output`, raising `UnsupportedModalityError(modality, model)`; return `RunResult(output, session_id, usage, tool_activity)`
    - _Requirements: 1.4, 1.5, 4.1, 4.2, 4.4, 5.1, 5.4, 6.3, 6.4_

  - [x] 3.3 Implement high-level media helpers
    - Implement `image()`/`video()` helpers constructing input parts from a file path (inferring media type from extension), raw bytes, or a URI
    - _Requirements: 4.2_

  - [x]* 3.4 Write property test for builder defaults producing a runnable agent
    - **Property 5: Builder defaults produce a runnable agent** — **Validates: Req 1.1, 1.2**

  - [x]* 3.5 Write property test for overrides winning over defaults
    - **Property 6: Overrides win over defaults** — **Validates: Req 2.2, 2.3**

  - [x]* 3.6 Write property test for input capability gating
    - **Property 7: Capability gating on input** — **Validates: Req 4.4, 6.3, 6.4**

  - [x]* 3.7 Write property test for output capability gating
    - **Property 8: Capability gating on output** — **Validates: Req 5.4**

  - [x]* 3.8 Write property test for missing-field validation
    - **Property 10: Missing-field validation** — **Validates: Req 1.6**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement the FastAPI transport adapter
  - [x] 5.1 Implement FastAPIAdapter with run, stream, and health endpoints
    - Implement `FastAPIAdapter(agent).app()` exposing `POST /run` (JSON AgentInput → RunResult), `POST /run/stream` (streamed RunChunks), and `GET /health`; support `session_id` routing so state persists across calls; map `UnsupportedModalityError`/validation errors to 4xx with descriptive messages; hold no agent logic beyond translation/routing
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 9.3_

  - [x]* 5.2 Write integration test for FastAPI run mapping
    - **Property 11: FastAPI run maps to a Run_Result** — **Validates: Req 7.2, 7.6**

  - [x]* 5.3 Write integration test for FastAPI session routing
    - **Property 12: FastAPI session routing** — **Validates: Req 7.5**

- [x] 6. Implement the MCP server transport adapter
  - [x] 6.1 Implement MCPServerAdapter exposing the agent as a tool
    - Implement `MCPServerAdapter(agent, tool_name)` that advertises a run tool accepting an AgentInput, runs `BuiltAgent.arun`, and maps `AgentOutput` parts to MCP content (text as text; image/video per MCP media/embedded-resource conventions); return an MCP error result on failure; hold no agent logic beyond translation
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 9.3_

  - [x]* 6.2 Write integration test for MCP tool advertise and run
    - **Property 13: MCP tool advertises and runs the agent** — **Validates: Req 8.2, 8.3, 8.4**

  - [x]* 6.3 Write integration test for MCP failure error result
    - **Property 14: MCP failure surfaces an error result** — **Validates: Req 8.5**

- [x] 7. Validate transport parity and kernel independence
  - [x]* 7.1 Write integration test for transport parity
    - **Property 15: Transport parity** — drive one BuiltAgent in-process, via FastAPI TestClient, and via a direct MCP tool call; assert equivalent AgentOutput — **Validates: Req 9.1, 9.2**

  - [x]* 7.2 Write unit test asserting kernel imports no new module
    - **Property 16: Kernel remains independent** — assert `loomable.kernel` tree imports nothing from `loomable.agent`/`content`/`serve` — **Validates: Req 1.7, 2.4, 7.7, 8.6, 10.3**

- [x] 8. Implement parallel multi-agent orchestration
  - [x] 8.1 Implement the Orchestrator with parallel, route, and coordinate modes
    - Implement `OrchestrationMode` and `Orchestrator`; add `sub_agents` and `mode` to the builder; PARALLEL wraps each sub-agent's `arun` as a `DelegatedTask` and delegates to the kernel `SubagentManager.run_all()` (concurrent, isolated, keyed); ROUTE selects exactly one sub-agent; COORDINATE delegates then synthesizes; aggregate child results into `RunResult.sub_results`
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8_

  - [x] 8.2 Surface parallel tool calling through the Built_Agent
    - Ensure multi-tool steps dispatch through the kernel `ToolRuntime.dispatch()` concurrently with per-call matching and fault isolation, exposed via the high-level run path
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [x]* 8.3 Write property test for parallel sub-agent concurrency
    - **Property 17: Parallel sub-agents execute concurrently** — **Validates: Req 11.2, 11.3**

  - [x]* 8.4 Write property test for sub-agent result keying and fault isolation
    - **Property 18: Sub-agent results keyed with fault isolation** — **Validates: Req 11.4, 11.5**

  - [x]* 8.5 Write property test for route mode single-agent execution
    - **Property 19: Route mode runs exactly one sub-agent** — **Validates: Req 11.6**

  - [x]* 8.6 Write property test for parallel tool call matching and isolation
    - **Property 20: Parallel tool calls complete with matching and isolation** — **Validates: Req 12.1, 12.2, 12.3**

- [x] 9. Implement structured output, hooks/HITL, knowledge, and sessions
  - [x] 9.1 Implement structured output
    - Add `output_schema` to `arun`/`run`; format the request for structured output, parse/validate against the schema into `RunResult.structured`; raise `StructuredOutputError` on failure; pass through unchanged when no schema
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [x] 9.2 Implement tool hooks and human-in-the-loop confirmation
    - Add `tool_hooks` (pre/post) and `require_confirmation`; express rejections as guardrail rules so the kernel `GuardrailHarness` blocks and records without executing; install a confirmation gate (injectable approver, default deny headless) for tools needing approval
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [x] 9.3 Implement high-level knowledge/retriever attachment
    - Add `retrievers` to the builder; wrap each with the kernel `RetrieverTool` and register in the `ToolRuntime` so retrieval is invocable as a tool (Agentic RAG at the edge)
    - _Requirements: 16.1, 16.2, 16.3, 16.4_

  - [x] 9.4 Implement high-level persistent memory and session resume
    - Add `session_id` create/resume via `SessionStore`; persist state after each run; raise the kernel `SessionNotFoundError` for unknown ids on resume
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

  - [x]* 9.5 Write property test for structured output validation
    - **Property 21: Structured output validates against the schema** — **Validates: Req 13.2, 13.3, 13.4**

  - [x]* 9.6 Write property test for tool pre-hook rejection
    - **Property 22: Tool pre-hook rejection blocks execution** — **Validates: Req 14.2, 14.3**

  - [x]* 9.7 Write property test for confirmation gate
    - **Property 23: Confirmation gate blocks unapproved tools** — **Validates: Req 14.4**

  - [x]* 9.8 Write integration test for session persistence via the builder
    - **Property 24: Session persistence round-trip via the builder** — **Validates: Req 15.2, 15.3, 15.4**

  - [x]* 9.9 Write property test for attached retriever invocation
    - **Property 25: Attached retriever is invocable as a tool** — **Validates: Req 16.2, 16.3**

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP.
- Each task references specific requirements (granular sub-clauses) for traceability.
- Property tests use `hypothesis` (min. 100 examples each), tagged `# Feature: agent-api, Property {n}`, with providers/HTTP/MCP mocked to keep iterations cheap.
- All 25 correctness properties from the design are covered by exactly one test sub-task.
- The `loomable.kernel` package is never modified; the feature is purely additive. Parallel orchestration reuses `SubagentManager`; parallel tools reuse `ToolRuntime`; hooks/HITL reuse `GuardrailHarness`; knowledge reuses `RetrieverTool`; sessions reuse `SessionStore`.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3"] },
    { "id": 3, "tasks": ["2.4", "2.5", "2.6", "2.7", "2.8", "3.1"] },
    { "id": 4, "tasks": ["3.2", "3.3"] },
    { "id": 5, "tasks": ["3.4", "3.5", "3.6", "3.7", "3.8"] },
    { "id": 6, "tasks": ["5.1", "6.1", "8.1", "8.2", "9.1", "9.2", "9.3", "9.4"] },
    { "id": 7, "tasks": ["5.2", "5.3", "6.2", "6.3", "7.1", "7.2", "8.3", "8.4", "8.5", "8.6", "9.5", "9.6", "9.7", "9.8", "9.9"] }
  ]
}
```
