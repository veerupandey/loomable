# Implementation Plan: agent-ergonomics

## Overview

This plan adds high-level ergonomics to `loomable.agent` incrementally, in Python (managed with `uv`), reusing existing kernel primitives and never modifying `loomable.kernel`. It starts with flexible input + `input_schema`, then the `@tool` decorator, the automatic tool-use loop, wiring `skills=`/`mcp_servers=`, memory compaction, tiered routing, and finally built-in embedders + knowledge. Each subsystem lands with tests mapped to the 17 correctness properties.

Property tests use `hypothesis` (min. 100 examples) where a quantifier is natural, tagged `# Feature: agent-ergonomics, Property {n}`, with providers/HTTP/MCP mocked. Tests run via `uv run pytest` (single run).

## Tasks

- [x] 1. Flexible structured input + input_schema
  - [x] 1.1 Finish input coercion and schema validation in the builder
    - `to_agent_input` already coerces str/AgentInput/Pydantic/dataclass/dict in `loomable.content`; wire it into `BuiltAgent` via `_coerce_input`, add an `input_schema` builder option and `_validate_against_schema` (Pydantic `model_validate` / dataclass construction), raise `InputValidationError` on failure; route `arun`/`run`/`astream` through `_coerce_input`; strings and `AgentInput` bypass schema
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [ ]* 1.2 Write property test for structured input serialization/pass-through
    - **Property 1: Structured input is serialized and passed through** — **Validates: Requirements 1.1, 1.2, 1.3**

  - [ ]* 1.3 Write property test for input schema validation gating
    - **Property 2: Input schema validation gates the run** — **Validates: Requirements 1.4, 1.5, 1.6**

- [x] 2. Function tools via the `@tool` decorator
  - [x] 2.1 Implement the @tool decorator and FunctionTool
    - Create `loomable/agent/tools.py` with `@tool` and `FunctionTool(Tool)`; derive name (fn name / override), description (docstring / override), and a JSON schema from the signature/annotations (str/int/float/bool/list/dict mapping, required = params without defaults); `invoke` binds args and calls the fn (await async, `asyncio.to_thread` for sync); catch exceptions into a `ToolResult` error naming the tool; export from `loomable/agent/__init__.py`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 2.2 Write property test for decorated-function tool behavior
    - **Property 3: Decorated function becomes an invocable tool** — **Validates: Requirements 2.1, 2.2, 2.4, 2.5**

  - [ ]* 2.3 Write property test for derived schema matching the signature
    - **Property 4: Derived tool schema matches the signature** — **Validates: Requirements 2.3**

  - [ ]* 2.4 Write unit test for function-tool error isolation
    - **Property 5: Function tool errors are isolated and named** — **Validates: Requirements 2.6**

- [x] 3. Automatic tool-use loop
  - [x] 3.1 Implement the model->dispatch->feed-back loop in BuiltAgent
    - Add `_run_tool_loop` used by `arun` when tools are present: advertise tool schemas via `ModelRequest.tools`, and while the model returns tool calls, dispatch them through the existing gated path (hooks/guardrails), append assistant tool-call + tool-result messages, and re-invoke; stop on no tool calls or `max_tool_iterations` (default 6); collect executed outcomes into `RunResult.tool_activity`; when no tools or no tool calls, behave as the single-shot path
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7_

  - [ ]* 3.2 Write property test for loop execution and termination
    - **Property 6: Tool-use loop runs tools and terminates** — **Validates: Requirements 3.1, 3.2, 3.3**

  - [ ]* 3.3 Write unit test for recorded tool activity
    - **Property 7: Tool activity is recorded** — **Validates: Requirements 3.4**

  - [ ]* 3.4 Write unit test for single-shot when no tool calls
    - **Property 8: No tool calls means single-shot** — **Validates: Requirements 3.6**

  - [ ]* 3.5 Write property test for hooks/guardrails inside the loop
    - **Property 9: Loop honors hooks/guardrails** — **Validates: Requirements 3.5**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Wire Skills and MCP servers into the builder
  - [x] 5.1 Wire skills= into build() via the kernel SkillLoader
    - Extend `_build_tool_registry` (or build()) to discover/load configured Skills through the kernel `SkillLoader` and register each Skill's script tools by name; isolate failures as `SkillLoadError` per Skill while others load
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 5.2 Wire mcp_servers= into build() via the kernel MCPClient
    - Connect configured MCP servers through the kernel `MCPClient`, enumerate their tools, and register each as a `Tool` whose invoke calls `MCPClient.call_tool`; isolate failed connections as `MCPConnectionError` per server while others proceed
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 5.3 Write integration test for skills registration with isolation
    - **Property 10: Skills register their script tools with isolation** — **Validates: Requirements 4.1, 4.2, 4.3**

  - [ ]* 5.4 Write integration test for MCP tool exposure with isolation
    - **Property 11: MCP tools are exposed with isolation** — **Validates: Requirements 5.1, 5.2, 5.3**

- [x] 6. Automatic memory compaction
  - [x] 6.1 Implement compaction via the kernel Summarizer
    - When retained turns exceed `compaction_threshold`, summarize the oldest overflow turns with the kernel `Summarizer` into `session.l2`, drop them from `session.l1`, keep the most recent window uncompacted, and have `_memory_prefix` prepend L2 summaries ahead of the retained turns
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 6.2 Write property test for compaction behavior
    - **Property 12: Compaction summarizes overflow and preserves recent turns** — **Validates: Requirements 6.1, 6.2, 6.3, 6.4**

- [x] 7. Tiered model routing through the builder
  - [x] 7.1 Expose tiered routing via the kernel ModelRouter
    - Accept `tiers`/`tier_policy`/`fallback_tiers` on the builder; when set, construct a kernel `ModelRouter` over the `ModelInterface` and route `_run_single`/`_run_tool_loop` calls through it, recording any `TierSubstitution` in `RunResult` metadata; with no tiers, use the single provider unchanged
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 7.2 Write property test for tier selection and fallback
    - **Property 13: Tier routing selects and falls back** — **Validates: Requirements 7.1, 7.2, 7.3**

  - [ ]* 7.3 Write unit test for no-tiers single-model path
    - **Property 14: No tiers means unchanged single model** — **Validates: Requirements 7.4**

- [x] 8. Built-in embedders and knowledge
  - [x] 8.1 Implement Embedder protocol and OpenAI/Azure embedders
    - Add an `Embedder` protocol and `OpenAIEmbedder` / `AzureOpenAIEmbedder` in `loomable.providers` (httpx, `/embeddings`); unavailable endpoint raises `ModelProviderError` naming the embedder
    - _Requirements: 8.1, 8.4_

  - [x] 8.2 Wire knowledge= into the builder with recall into context
    - Accept `knowledge=[docs]` + `embedder=`; on build, embed and index each document into a `LongTermStore`; at run time embed the input, recall top-k via `LongTermStore.query`, and prepend the retrieved snippets to the model context
    - _Requirements: 8.2, 8.3, 8.5_

  - [ ]* 8.3 Write unit test for embedder round-trip and unavailability
    - **Property 15: Embedder round-trip and unavailability** — **Validates: Requirements 8.1, 8.4**

  - [ ]* 8.4 Write integration test for knowledge indexing and recall
    - **Property 16: Attached knowledge is indexed and recalled** — **Validates: Requirements 8.2, 8.3, 8.5**

- [x] 9. Kernel independence guard
  - [ ]* 9.1 Extend the kernel-independence test to cover providers
    - **Property 17: Kernel remains independent** — assert `loomable.kernel` imports nothing from `loomable.agent`/`content`/`serve`/`providers` — **Validates: Requirements 9.2, 9.3**

- [x] 10. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional test tasks and can be skipped for a faster MVP.
- Each task references specific requirements for traceability.
- Property tests use `hypothesis` (min. 100 examples) where natural, tagged `# Feature: agent-ergonomics, Property {n}`, with providers/HTTP/MCP mocked.
- All 17 correctness properties are covered by exactly one test sub-task.
- The `loomable.kernel` package is never modified; the feature is purely additive and reuses `ToolRuntime`, `Tool`, `SkillLoader`, `MCPClient`, `Summarizer`, `ModelRouter`, and `LongTermStore`.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "1.3", "2.2", "2.3", "2.4", "3.1"] },
    { "id": 2, "tasks": ["3.2", "3.3", "3.4", "3.5"] },
    { "id": 3, "tasks": ["5.1", "5.2", "6.1", "7.1", "8.1"] },
    { "id": 4, "tasks": ["5.3", "5.4", "6.2", "7.2", "7.3", "8.2", "8.3", "9.1"] },
    { "id": 5, "tasks": ["8.4"] }
  ]
}
```
