# Requirements Document

## Introduction

This feature adds a **high-level agent creation API** on top of the existing `loomable` kernel, in the spirit of agno-style agents: create and run a capable agent in a few lines, with sensible defaults and minimal boilerplate, while never hiding the low-level kernel primitives. The feature also adds **first-class multimodal support** (text, image, and video) for both agent **input** and **output**, exposed at two levels — a **low level** (typed content primitives and kernel plumbing) and a **high level** (ergonomic helpers on the builder API). Finally, an agent must be **exposable over a FastAPI HTTP server and as an MCP server**, so the same agent can be reached through both transports without changing agent logic.

The design principle of `loomable` is preserved: the **Kernel remains stable and generic**. The high-level API is an additive convenience layer (`loomable.agent`) that composes existing kernel primitives (`AgentLoop`, `ModelInterface`, `ContextManager`, `MemoryManager`, `ToolRuntime`, `GuardrailHarness`, `Planner`, `SubagentManager`, `SessionStore`, `ExtensionRegistry`) and does not require kernel source changes. The transports (FastAPI, MCP) are edge adapters that wrap a built agent.

The project stack is Python managed with uv (as established by the existing framework).

## Glossary

- **Kernel**: The existing stable, generic core (`loomable.kernel`) that runs the agent loop, memory, model routing, tools, and extensions.
- **High-Level API**: The new ergonomic layer (`loomable.agent`) for creating and running agents with minimal boilerplate.
- **Agent_Builder**: The high-level, fluent/declarative entry point that assembles a runnable agent from a compact configuration.
- **Built_Agent**: A runnable agent produced by the Agent_Builder, wrapping a kernel `AgentLoop` and its subsystems.
- **Low_Level_API**: Direct use of kernel primitives and typed multimodal content objects, without the builder.
- **Multimodal_Content**: A typed content item that is one of text, image, or video.
- **Media_Part**: A single unit of Multimodal_Content (e.g., one image or one video) with a media type and a reference to its bytes/URI.
- **Modality**: One of `text`, `image`, `video`.
- **Agent_Input**: The message(s) supplied to an agent for a run, which may contain one or more Multimodal_Content parts.
- **Agent_Output**: The response produced by an agent for a run, which may contain one or more Multimodal_Content parts.
- **Model_Capabilities**: The declared set of modalities a configured model provider supports for input and output.
- **FastAPI_Adapter**: The HTTP edge adapter that exposes a Built_Agent as REST endpoints using FastAPI.
- **MCP_Server_Adapter**: The edge adapter that exposes a Built_Agent as an MCP server (agent-as-tool over the Model Context Protocol).
- **Run_Request**: A single invocation of a Built_Agent with an Agent_Input, optionally streamed.
- **Run_Result**: The Agent_Output plus run metadata (usage, session id, tool activity) returned from a Run_Request.

## Requirements

### Requirement 1: High-Level Agent Creation API

**User Story:** As an agent developer, I want to create a fully working agent in a few lines like agno, so that I don't have to wire every kernel subsystem by hand.

#### Acceptance Criteria

1. THE High_Level_API SHALL provide an Agent_Builder that creates a Built_Agent from a compact configuration specifying at minimum a model.
2. WHEN a developer creates an agent supplying only a model, THE Agent_Builder SHALL supply default implementations for context management, memory, tool runtime, guardrails, planning, and session persistence.
3. THE Agent_Builder SHALL accept optional configuration for instructions/system prompt, tools, skills, MCP servers, token budget, and session identifier.
4. THE Agent_Builder SHALL expose a run method that accepts an Agent_Input and returns a Run_Result.
5. THE Agent_Builder SHALL expose a streaming run method that yields incremental output for a Run_Request.
6. WHERE a required configuration value is missing or invalid, THE Agent_Builder SHALL raise an error identifying the offending field before any run is attempted.
7. THE Built_Agent SHALL wrap kernel primitives without requiring changes to `loomable.kernel` source code.

### Requirement 2: Low-Level Access Preserved

**User Story:** As an advanced developer, I want the high-level API to expose the underlying kernel primitives, so that I can drop down to the low level when I need full control.

#### Acceptance Criteria

1. THE Built_Agent SHALL expose read access to the underlying kernel subsystems it composes (loop, model interface, memory, tool runtime, session).
2. THE High_Level_API SHALL allow a developer to supply pre-constructed kernel primitives that override the builder defaults.
3. WHEN a developer supplies a custom kernel primitive, THE Agent_Builder SHALL use that primitive instead of constructing its default.
4. THE Low_Level_API SHALL remain fully usable without the Agent_Builder.

### Requirement 3: Multimodal Content Model

**User Story:** As an agent developer, I want typed content for text, image, and video, so that multimodal data is represented consistently across the framework.

#### Acceptance Criteria

1. THE Low_Level_API SHALL define a Multimodal_Content type that represents exactly one Modality among text, image, and video.
2. THE Multimodal_Content type SHALL carry a media type indicator and a reference to its payload as either inline bytes or a URI.
3. THE Low_Level_API SHALL support an Agent_Input composed of an ordered sequence of one or more Multimodal_Content parts.
4. THE Low_Level_API SHALL support an Agent_Output composed of an ordered sequence of one or more Multimodal_Content parts.
5. IF a Media_Part is constructed with neither inline bytes nor a URI, THEN THE Low_Level_API SHALL reject the construction with a validation error.
6. IF a Media_Part declares a Modality inconsistent with its media type, THEN THE Low_Level_API SHALL reject the construction with a validation error.

### Requirement 4: Multimodal Input Handling

**User Story:** As an agent developer, I want to pass images and video into an agent at both the low and high level, so that the agent can reason over media.

#### Acceptance Criteria

1. THE Agent_Builder run method SHALL accept an Agent_Input containing text, image, and/or video parts.
2. THE High_Level_API SHALL provide convenience helpers to construct image and video input parts from a file path, raw bytes, or a URI.
3. WHEN an agent runs with multimodal input, THE Built_Agent SHALL forward the multimodal parts to the configured model through the Model_Interface.
4. IF the configured model does not declare support for an input Modality present in the Agent_Input, THEN THE Built_Agent SHALL return an error identifying the unsupported Modality and the model.
5. THE Low_Level_API SHALL accept the same multimodal Agent_Input directly through the kernel model request path.

### Requirement 5: Multimodal Output Handling

**User Story:** As an agent developer, I want agents to return images/video in addition to text, so that generative and media-producing tasks are supported.

#### Acceptance Criteria

1. THE Built_Agent SHALL return a Run_Result whose Agent_Output may contain text, image, and/or video parts.
2. WHEN the configured model returns media output, THE Built_Agent SHALL represent each returned medium as a Multimodal_Content part in the Agent_Output.
3. WHERE a model returns only text, THE Agent_Output SHALL contain a single text part.
4. IF the configured model does not declare support for a requested output Modality, THEN THE Built_Agent SHALL return an error identifying the unsupported Modality and the model.
5. THE Low_Level_API SHALL expose the same multimodal Agent_Output structure returned by the kernel model response path.

### Requirement 6: Model Capability Declaration and Enforcement

**User Story:** As an agent developer, I want the framework to know which modalities a model supports, so that unsupported requests fail fast with a clear message.

#### Acceptance Criteria

1. THE High_Level_API SHALL allow a model configuration to declare its supported input and output Modalities as Model_Capabilities.
2. WHERE Model_Capabilities are not declared, THE High_Level_API SHALL default to text-only input and output.
3. BEFORE forwarding a Run_Request, THE Built_Agent SHALL validate the Agent_Input modalities against the model's declared input capabilities.
4. IF a requested modality is not supported, THEN THE Built_Agent SHALL raise or return an error naming the modality and the model without invoking the provider.

### Requirement 7: FastAPI HTTP Exposure

**User Story:** As an integrator, I want to expose a built agent over HTTP with FastAPI, so that clients can call the agent through a REST API.

#### Acceptance Criteria

1. THE FastAPI_Adapter SHALL produce a FastAPI application that exposes a Built_Agent over HTTP.
2. THE FastAPI_Adapter SHALL provide an endpoint that accepts an Agent_Input (including multimodal parts) and returns a Run_Result.
3. THE FastAPI_Adapter SHALL provide a streaming endpoint that streams incremental output for a Run_Request.
4. THE FastAPI_Adapter SHALL provide a health endpoint reporting agent readiness.
5. WHEN a client submits a request referencing a session identifier, THE FastAPI_Adapter SHALL route the run to that session so state persists across calls.
6. IF a request payload is malformed or references an unsupported modality, THEN THE FastAPI_Adapter SHALL respond with a client-error status and a message identifying the problem.
7. THE FastAPI_Adapter SHALL expose the agent without requiring changes to `loomable.kernel` source code.

### Requirement 8: MCP Server Exposure

**User Story:** As an integrator, I want to expose a built agent as an MCP server, so that MCP-compatible clients can invoke the agent as a tool.

#### Acceptance Criteria

1. THE MCP_Server_Adapter SHALL expose a Built_Agent as an MCP server implementing the Model Context Protocol at the server boundary.
2. WHEN an MCP client connects, THE MCP_Server_Adapter SHALL advertise at least one tool that runs the agent with a provided Agent_Input.
3. WHEN an MCP client invokes the agent tool, THE MCP_Server_Adapter SHALL execute a Run_Request and return the Agent_Output as the tool result.
4. THE MCP_Server_Adapter SHALL represent multimodal output parts in the tool result according to MCP content conventions.
5. IF an MCP invocation fails, THEN THE MCP_Server_Adapter SHALL return an error result identifying the failure.
6. THE MCP_Server_Adapter SHALL expose the agent without requiring changes to `loomable.kernel` source code.

### Requirement 9: Transport Parity

**User Story:** As an integrator, I want the same agent to behave consistently whether called in-process, over FastAPI, or over MCP, so that transport choice does not change agent behavior.

#### Acceptance Criteria

1. WHEN the same Built_Agent is invoked with an equivalent Agent_Input in-process, over the FastAPI_Adapter, and over the MCP_Server_Adapter, THE Framework SHALL execute the same agent logic through the same kernel loop.
2. THE FastAPI_Adapter and MCP_Server_Adapter SHALL each operate on a Built_Agent produced by the Agent_Builder or the Low_Level_API.
3. THE transports SHALL NOT embed agent business logic beyond request/response translation and session routing.

### Requirement 10: Python and uv Tooling

**User Story:** As a framework maintainer, I want the new layer implemented in Python and managed with uv, so that it matches the existing project stack.

#### Acceptance Criteria

1. THE High_Level_API, adapters, and multimodal types SHALL be implemented in Python.
2. THE feature SHALL manage any new dependencies (e.g., FastAPI, an ASGI server, MCP server support) through uv.
3. THE feature SHALL NOT modify the existing `loomable.kernel` package to achieve its goals.

### Requirement 11: Parallel Multi-Agent Orchestration

**User Story:** As an agent developer, I want to give an agent a set of sub-agents and have them run in parallel, so that independent specialists execute concurrently instead of one after another (like agno teams and LangChain deep-agent sub-agents).

#### Acceptance Criteria

1. THE High_Level_API SHALL allow a Built_Agent to be configured with a set of child agents (Sub_Agents).
2. THE High_Level_API SHALL support a parallel orchestration mode in which all Sub_Agents run concurrently on the same Agent_Input and their results are aggregated.
3. WHEN a Built_Agent runs in parallel mode with N Sub_Agents, THE Framework SHALL execute the N Sub_Agents concurrently rather than sequentially.
4. WHEN parallel Sub_Agents complete, THE Framework SHALL return each Sub_Agent result keyed to its originating Sub_Agent.
5. IF one Sub_Agent fails, THEN THE Framework SHALL return a failure for that Sub_Agent identifying it AND SHALL still return the results of the Sub_Agents that succeeded.
6. THE High_Level_API SHALL support a route orchestration mode in which exactly one Sub_Agent is selected to handle the Agent_Input.
7. THE High_Level_API SHALL support a coordinate orchestration mode in which a leader delegates to Sub_Agents and synthesizes their results into a single Agent_Output.
8. THE parallel orchestration SHALL reuse the existing kernel Subagent concurrency primitive without modifying `loomable.kernel` source code.

### Requirement 12: Parallel Tool Calling at the High Level

**User Story:** As an agent developer, I want independent tool calls in a single step to run in parallel through the high-level API, so that multi-tool steps complete faster.

#### Acceptance Criteria

1. WHEN an agent step yields multiple independent tool calls, THE Built_Agent SHALL dispatch them concurrently.
2. WHEN concurrent tool calls complete, THE Built_Agent SHALL associate each result with its originating tool call.
3. IF one concurrent tool call fails, THEN THE Built_Agent SHALL return that failure and still return the results of the tool calls that succeeded.
4. THE parallel tool dispatch SHALL reuse the existing kernel Tool Runtime without modifying `loomable.kernel` source code.

### Requirement 13: Structured Output

**User Story:** As an agent developer, I want to request a typed/structured response, so that the agent returns validated data I can use programmatically (like agno response models).

#### Acceptance Criteria

1. THE High_Level_API SHALL allow a Run_Request to specify an expected output schema.
2. WHEN an output schema is specified, THE Built_Agent SHALL return the Run_Result output parsed and validated against that schema.
3. IF the model output cannot be parsed or validated against the specified schema, THEN THE Built_Agent SHALL return an error identifying the validation failure.
4. WHERE no output schema is specified, THE Built_Agent SHALL return the Agent_Output unchanged.

### Requirement 14: Tool Hooks and Human-in-the-Loop

**User Story:** As an agent developer, I want pre/post hooks around tool calls and optional human confirmation, so that I can validate, audit, transform, or gate tool execution (like agno tool hooks and HITL).

#### Acceptance Criteria

1. THE High_Level_API SHALL allow registration of pre-hooks that run before a tool call and post-hooks that run after a tool call.
2. WHEN a tool call is dispatched, THE Built_Agent SHALL invoke registered pre-hooks before execution and post-hooks after execution.
3. WHERE a pre-hook rejects a tool call, THE Built_Agent SHALL block that tool call and record the rejection without executing the tool.
4. WHERE a tool is configured to require confirmation, THE Built_Agent SHALL pause and require an approval decision before executing that tool.
5. THE hook and confirmation mechanism SHALL build on the existing kernel Guardrail Harness without modifying `loomable.kernel` source code.

### Requirement 15: Persistent Memory and Sessions at the High Level

**User Story:** As an agent developer, I want persistent memory and resumable sessions exposed through the builder, so that an agent remembers across runs without manual wiring (like agno durable memory).

#### Acceptance Criteria

1. THE High_Level_API SHALL allow a Built_Agent to be created or resumed by session identifier.
2. WHEN an agent runs with a session identifier, THE Built_Agent SHALL persist conversational and session state after the run using the kernel Session Store.
3. WHEN an agent is resumed by an existing session identifier, THE Built_Agent SHALL restore the persisted state so prior turns are available.
4. IF a resume is requested for an unknown session identifier, THEN THE High_Level_API SHALL return a not-found error identifying the session identifier.

### Requirement 16: Knowledge / Retriever Integration at the High Level

**User Story:** As an agent developer, I want to attach a knowledge source (retriever) through the builder, so that the agent can ground responses via Agentic RAG without hand-wiring tools.

#### Acceptance Criteria

1. THE High_Level_API SHALL allow one or more Retrievers to be attached to a Built_Agent through configuration.
2. WHEN a Retriever is attached, THE Built_Agent SHALL expose it to the agent as an invocable tool using the existing kernel retriever-as-tool mechanism.
3. WHEN the agent invokes an attached Retriever with a query, THE Built_Agent SHALL return the retrieved content to the agent.
4. THE knowledge integration SHALL NOT require changes to `loomable.kernel` source code.
