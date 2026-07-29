# Requirements Document

## Introduction

`loomable` is a lightweight, ultra-fast, agno-style, general-purpose agent framework. Its whole point is being domain-agnostic: it is not tied to any particular business, industry, or use case. Its central design principle is a **kernel + capabilities** model: a stable, generic, lightweight core (the Kernel) is built once and never modified to support a new use case. New projects and domain capabilities are added exclusively through two extension points — **Anthropic-style Skills** (which may bundle script tools) and **MCP servers**. Direct HTTP API tools are supported as a first-class tool type available to those extensions. This means the framework can be applied to ANY domain by writing Skills and MCP servers rather than modifying the engine.

The name `loomable` reflects the design: a stable loom (the Kernel) that weaves together Skills and tools into any capability.

The framework is model-agnostic (a single interface across many model providers), instantiates agents quickly with a low memory footprint, and ships with SQLite-backed session persistence out of the box. Memory is treated as first-class infrastructure: a multi-tier model (recent raw turns, compressed summaries/entities, vector episodic retrieval), an explicit token budget for the context window, and periodic checkpoint summarization to survive long-horizon tasks and prevent context drift.

The framework also supports parallel tool calling, subagents, a planning capability that can optionally use a separately configured model, retriever integration (any retriever framework exposed as a tool over MCP or API) enabling Agentic RAG on top, and tiered model routing for cost/latency control.

Domain capabilities are added purely through the extension points (Skills and MCP servers) without modifying the Kernel, and this property is validated using representative example Skills rather than any specific business domain.

### Design Rationale (Research Context, mid-2026)

- The protocol stack has settled: MCP for agent↔tools, A2A for agent↔agent. MCP is the tool foundation; A2A is relevant later for multi-agent scenarios.
- Most production agent failures are memory/context failures, not model failures. Context drift degrades agents before token limits are reached.
- Retrieval-based memory reduces token usage by 70–90% versus full-history approaches (Mem0/Zep/Letta benchmarks).
- Aligning static content (system prompt + tool schemas) at the top of the context enables prefix caching and reduces time-to-first-token.
- Loop engineering (single-agent perceive→plan→act→observe cycle) with harness-level guardrails (constraints the agent cannot override, verification gates, persistent state) is the default; graph engineering (multiple loops wired via nodes/edges/shared state) is added only when tasks are genuinely parallelizable.

## Glossary

- **Kernel**: The stable, generic core framework that orchestrates the agent loop, memory, planning, parallelism, and extension loading. The Kernel is built once and is not modified to onboard a new project.
- **Framework**: The complete `loomable` system, comprising the Kernel plus all installed extensions (Skills and MCP servers).
- **Agent**: A configured instance of the perceive→plan→act→observe execution loop produced by the Kernel.
- **Extension Point**: One of exactly two supported mechanisms for adding new capability — a Skill or an MCP server.
- **Skill**: An Anthropic-style capability package, loaded at the edge, that may bundle instructions and script tools. Skills are one of the two Extension Points.
- **Script Tool**: An executable script bundled within a Skill and invocable by an Agent as a tool.
- **MCP_Server**: An external server implementing the Model Context Protocol that exposes tools and/or data to an Agent. MCP servers are the second Extension Point.
- **MCP_Client**: The Kernel component that connects to MCP_Servers and mediates tool/data calls using the Model Context Protocol.
- **API_Tool**: A tool that lets an Agent interact with an external service over an HTTP API.
- **Model_Provider**: An external large-language-model service accessible through the Framework's single model interface.
- **Model_Interface**: The single, model-agnostic interface through which the Kernel invokes any Model_Provider.
- **Model_Router**: The Kernel component that selects among tiered models for a given call based on configured cost/latency policy.
- **Planner**: The Kernel capability that produces an execution plan and may use a separately configured planning model.
- **Planning_Model**: The Model_Provider configuration used by the Planner, which may differ from the Agent's primary model.
- **Subagent**: A child Agent instance spawned by a parent Agent to perform a delegated unit of work.
- **Memory_Manager**: The Kernel component coordinating short-term memory, long-term memory, context budget, and summarization.
- **Short_Term_Store**: The RDBMS-backed store for recent conversational and session state. Default backend is SQLite.
- **Long_Term_Store**: The vector-database-backed store for episodic long-term memory. Default backend is zvec.
- **Memory_Backend**: A pluggable implementation of Short_Term_Store or Long_Term_Store selectable by configuration.
- **Session_Store**: The out-of-the-box SQLite-backed persistence for Agent sessions.
- **Context_Manager**: The Kernel component that tracks the context window against a configured token budget and controls admission, retention, and eviction of context items.
- **Token_Budget**: The configured maximum number of tokens permitted in an Agent's context window.
- **Summarizer**: The Kernel component that compresses conversation history into a structured summary at a configured checkpoint interval.
- **Checkpoint_Interval**: The configured number of Agent steps (K) between summarization checkpoints.
- **Memory_Tier_L1**: Raw recent conversation turns.
- **Memory_Tier_L2**: Compressed entity and summary representations derived from history.
- **Memory_Tier_L3**: Vector-indexed episodic memory retrieved on demand.
- **Retriever**: A component, built with any framework, that returns relevant content for a query and is exposed to an Agent as a tool over MCP or API.
- **Agentic_RAG**: A retrieval-augmented-generation capability implemented as a Skill/retriever capability that runs on top of Retrievers, not baked into the Kernel.
- **Domain_Skill**: A Skill implementing an arbitrary business capability, used to validate that domains can be added through the Skill Extension Point without modifying the Kernel.

## Requirements

### Requirement 1: Kernel Stability and Extension-Only Onboarding

**User Story:** As a framework maintainer, I want the Kernel to remain stable and generic, so that new projects are onboarded without modifying the engine.

#### Acceptance Criteria

1. THE Kernel SHALL expose exactly two Extension Points: Skills and MCP_Servers.
2. WHEN a new use case is onboarded, THE Framework SHALL enable the required capability through a Skill, an MCP_Server, or both, without changes to Kernel source code.
3. WHERE a capability is domain-specific, THE Framework SHALL implement the capability as a Skill or MCP_Server rather than within the Kernel.
4. IF an onboarding request would require modifying Kernel source code, THEN THE Framework SHALL reject the request and report that only Skills and MCP_Servers are supported extension mechanisms.
5. THE Kernel SHALL load Skills and connect MCP_Servers through configuration without requiring recompilation of the Kernel.

### Requirement 2: Model-Agnostic Single Interface

**User Story:** As an agent developer, I want a single interface across many model providers, so that I can switch providers without rewriting agent logic.

#### Acceptance Criteria

1. THE Model_Interface SHALL accept model invocation requests using one consistent request format across all configured Model_Providers.
2. WHEN an Agent invokes a model through the Model_Interface, THE Kernel SHALL route the request to the configured Model_Provider without requiring Agent-specific provider code.
3. WHERE a Model_Provider is changed in configuration, THE Framework SHALL execute existing Agents against the new Model_Provider without changes to Agent logic.
4. IF a configured Model_Provider is unavailable, THEN THE Model_Interface SHALL return an error identifying the affected Model_Provider.

### Requirement 3: Fast Instantiation and Low Memory Footprint

**User Story:** As an agent developer, I want agents to instantiate quickly with a low memory footprint, so that the framework stays lightweight and responsive at scale.

#### Acceptance Criteria

1. WHEN an Agent is instantiated with a fixed reference configuration on the reference hardware, THE Kernel SHALL complete instantiation within 50 milliseconds.
2. WHEN an Agent is instantiated with a fixed reference configuration on the reference hardware, THE Kernel SHALL allocate no more than 15 megabytes of resident memory for the Agent instance.
3. THE Kernel SHALL load a Skill or connect an MCP_Server only when the corresponding capability is enabled in configuration.

### Requirement 4: MCP Protocol Support

**User Story:** As an integrator, I want complete MCP protocol support at the agent-to-tools boundary, so that any MCP-compliant tool or data source can be wired in.

#### Acceptance Criteria

1. THE MCP_Client SHALL implement the Model Context Protocol for the agent-to-tools boundary.
2. WHEN an MCP_Server is configured, THE MCP_Client SHALL connect to the MCP_Server and enumerate the tools and data resources the MCP_Server exposes.
3. WHEN an Agent invokes an MCP_Server tool, THE MCP_Client SHALL transmit the request and return the MCP_Server response to the Agent.
4. IF an MCP_Server connection fails, THEN THE MCP_Client SHALL report a connection error identifying the affected MCP_Server and SHALL continue operating with the remaining configured MCP_Servers.
5. IF an MCP_Server tool invocation returns an error, THEN THE MCP_Client SHALL return the error to the Agent identifying the failed tool.

### Requirement 5: Anthropic-Style Skills Support

**User Story:** As a capability author, I want to package capabilities as Anthropic-style Skills that can bundle script tools, so that I can add domain behavior at the edge without touching the Kernel.

#### Acceptance Criteria

1. THE Kernel SHALL load Skills that conform to the Anthropic-style Skill structure.
2. WHERE a Skill bundles one or more Script Tools, THE Kernel SHALL make each Script Tool invocable by an Agent as a tool.
3. WHEN an Agent invokes a Script Tool, THE Kernel SHALL execute the Script Tool and return its result to the Agent.
4. IF a Skill fails to load, THEN THE Kernel SHALL report a load error identifying the affected Skill and SHALL continue loading the remaining configured Skills.
5. IF a Script Tool execution fails, THEN THE Kernel SHALL return the failure to the Agent identifying the failed Script Tool.

### Requirement 6: Direct API Tool Support

**User Story:** As a capability author, I want agents to interact with services over HTTP APIs, so that agents can use external services that are not exposed through MCP.

#### Acceptance Criteria

1. THE Kernel SHALL support API_Tools that invoke external services over HTTP.
2. WHEN an Agent invokes an API_Tool, THE Kernel SHALL send the configured HTTP request and return the service response to the Agent.
3. IF an API_Tool request returns a non-success HTTP status, THEN THE Kernel SHALL return an error to the Agent that includes the HTTP status code.
4. IF an API_Tool request exceeds its configured timeout, THEN THE Kernel SHALL cancel the request and return a timeout error to the Agent.

### Requirement 7: Short-Term Memory with Pluggable RDBMS Backend

**User Story:** As an agent developer, I want short-term memory backed by an RDBMS with a pluggable backend, so that I can start on SQLite and move to Postgres, DynamoDB, or AWS Bedrock AgentCore Memory later.

#### Acceptance Criteria

1. THE Short_Term_Store SHALL persist recent conversational and session state in an RDBMS-backed Memory_Backend.
2. THE Short_Term_Store SHALL use SQLite as the default Memory_Backend.
3. WHERE an alternative Memory_Backend is configured, THE Short_Term_Store SHALL use the configured Memory_Backend without changes to Agent logic.
4. WHEN an Agent writes short-term state, THE Short_Term_Store SHALL persist the state to the active Memory_Backend.
5. WHEN an Agent reads short-term state, THE Short_Term_Store SHALL return the persisted state from the active Memory_Backend.
6. IF the active Memory_Backend is unavailable, THEN THE Short_Term_Store SHALL return an error identifying the affected Memory_Backend.

### Requirement 8: Long-Term Memory with Pluggable Vector Backend

**User Story:** As an agent developer, I want long-term memory backed by a vector database with a pluggable backend, so that I can start on zvec and move to AWS AgentCore or Bedrock Knowledge Bases later.

#### Acceptance Criteria

1. THE Long_Term_Store SHALL persist episodic long-term memory in a vector-database Memory_Backend.
2. THE Long_Term_Store SHALL use zvec as the default Memory_Backend.
3. WHERE an alternative vector Memory_Backend is configured, THE Long_Term_Store SHALL use the configured Memory_Backend without changes to Agent logic.
4. WHEN an Agent stores a long-term memory item, THE Long_Term_Store SHALL index the item in the active vector Memory_Backend.
5. WHEN an Agent queries long-term memory with a query, THE Long_Term_Store SHALL return the most relevant stored items from the active vector Memory_Backend ranked by similarity.
6. IF the active vector Memory_Backend is unavailable, THEN THE Long_Term_Store SHALL return an error identifying the affected Memory_Backend.

### Requirement 9: Context Window Tracking and Token Budget

**User Story:** As an agent developer, I want the context window tracked against a token budget, so that I can control what enters, stays in, and is evicted from context like managing RAM.

#### Acceptance Criteria

1. THE Context_Manager SHALL track the current token count of the Agent context window against the configured Token_Budget.
2. WHEN a context item is added, THE Context_Manager SHALL update the tracked token count to include the added item.
3. IF adding a context item would cause the tracked token count to exceed the Token_Budget, THEN THE Context_Manager SHALL evict lower-priority context items until the tracked token count is at or below the Token_Budget before admitting the item.
4. THE Context_Manager SHALL retain the active system prompt and tool schemas when evicting context items.
5. WHILE assembling the context window, THE Context_Manager SHALL place the system prompt and tool schemas at the start of the context window to enable prefix caching.

### Requirement 10: Checkpoint Summarization

**User Story:** As an agent developer, I want conversation history compressed into a structured summary at a checkpoint interval, so that long-horizon tasks survive without context drift.

#### Acceptance Criteria

1. WHEN an Agent completes a number of steps equal to the Checkpoint_Interval, THE Summarizer SHALL compress the accumulated conversation history into a structured summary.
2. WHEN the Summarizer produces a structured summary, THE Memory_Manager SHALL store the structured summary as Memory_Tier_L2 content.
3. WHERE a structured summary covers earlier conversation turns, THE Context_Manager SHALL replace those covered raw turns with the structured summary in the context window.
4. THE Summarizer SHALL preserve task objectives and decisions in the structured summary.

### Requirement 11: Multi-Tier Memory Model

**User Story:** As an agent developer, I want a multi-tier memory model, so that recent detail, compressed summaries, and episodic recall are managed distinctly.

#### Acceptance Criteria

1. THE Memory_Manager SHALL maintain Memory_Tier_L1 as raw recent conversation turns.
2. THE Memory_Manager SHALL maintain Memory_Tier_L2 as compressed entity and summary representations derived from history.
3. THE Memory_Manager SHALL maintain Memory_Tier_L3 as vector-indexed episodic memory retrieved on demand.
4. WHEN an Agent requests episodic recall with a query, THE Memory_Manager SHALL retrieve relevant items from Memory_Tier_L3 ranked by similarity.

### Requirement 12: Session Persistence Out of the Box

**User Story:** As an agent developer, I want SQLite-backed session persistence out of the box, so that agent sessions survive process restarts without extra setup.

#### Acceptance Criteria

1. THE Session_Store SHALL persist Agent sessions using SQLite by default without additional configuration.
2. WHEN an Agent session is created or updated, THE Session_Store SHALL persist the session state.
3. WHEN an Agent session is resumed by session identifier, THE Session_Store SHALL restore the persisted session state.
4. IF a requested session identifier does not exist, THEN THE Session_Store SHALL return a not-found error identifying the requested session identifier.

### Requirement 13: Parallel Tool Calling

**User Story:** As an agent developer, I want the agent to call independent tools in parallel, so that multi-tool steps complete faster.

#### Acceptance Criteria

1. WHEN an Agent step requests multiple independent tool calls, THE Kernel SHALL execute the tool calls concurrently.
2. WHEN concurrent tool calls complete, THE Kernel SHALL return each tool result associated with its originating tool call.
3. IF one concurrent tool call fails, THEN THE Kernel SHALL return the failure for that tool call and SHALL return the results of the tool calls that succeeded.

### Requirement 14: Subagents

**User Story:** As an agent developer, I want an agent to spawn subagents, so that delegated units of work can run as separate agent loops.

#### Acceptance Criteria

1. WHEN a parent Agent delegates a unit of work, THE Kernel SHALL instantiate a Subagent to perform the delegated work.
2. WHEN a Subagent completes its delegated work, THE Kernel SHALL return the Subagent result to the parent Agent.
3. WHERE multiple Subagents are delegated independent work, THE Kernel SHALL execute the Subagents concurrently.
4. IF a Subagent fails, THEN THE Kernel SHALL return the failure to the parent Agent identifying the failed Subagent.

### Requirement 15: Planning with Optional Separate Model

**User Story:** As an agent developer, I want a planning capability that can use a separately configured model, so that I can optimize planning quality independently of the primary model.

#### Acceptance Criteria

1. WHEN planning is requested, THE Planner SHALL produce an execution plan for the Agent task.
2. WHERE a Planning_Model is configured, THE Planner SHALL invoke the Planning_Model for plan generation.
3. WHERE no Planning_Model is configured, THE Planner SHALL invoke the Agent primary model for plan generation.
4. IF the Planning_Model is unavailable, THEN THE Planner SHALL return an error identifying the Planning_Model.

### Requirement 16: Retriever Integration and Agentic RAG

**User Story:** As a capability author, I want to expose any retriever as a tool over MCP or API and build Agentic RAG on top, so that retrieval stays at the edge and out of the Kernel.

#### Acceptance Criteria

1. THE Kernel SHALL support Retrievers that are exposed to an Agent as a tool over MCP or over an HTTP API.
2. WHEN an Agent invokes a Retriever tool with a query, THE Kernel SHALL return the retrieved content to the Agent.
3. THE Framework SHALL implement Agentic_RAG as a Skill or retriever capability that runs on top of Retrievers.
4. WHERE Agentic_RAG is enabled, THE Kernel SHALL NOT require Kernel source changes to add or replace a Retriever.
5. IF a Retriever tool invocation fails, THEN THE Kernel SHALL return the failure to the Agent identifying the failed Retriever.

### Requirement 17: Tiered Model Routing

**User Story:** As an agent developer, I want tiered model routing, so that I can balance cost and latency across model tiers.

#### Acceptance Criteria

1. WHERE tiered model routing is configured, THE Model_Router SHALL select a model tier for each model call according to the configured cost and latency policy.
2. WHEN the Model_Router selects a model tier, THE Kernel SHALL route the model call to the selected tier through the Model_Interface.
3. IF the selected model tier is unavailable, THEN THE Model_Router SHALL select a configured fallback tier and SHALL record the tier substitution.

### Requirement 18: Agent Loop and Harness-Level Guardrails

**User Story:** As a framework maintainer, I want a single-agent perceive→plan→act→observe loop with harness-level guardrails, so that agents operate within constraints they cannot override.

#### Acceptance Criteria

1. THE Kernel SHALL execute the Agent using a perceive→plan→act→observe loop.
2. THE Kernel SHALL enforce configured guardrail constraints that an Agent cannot override.
3. IF an Agent action violates a configured guardrail constraint, THEN THE Kernel SHALL block the action and record the blocked action.
4. WHERE a verification gate is configured for an Agent step, THE Kernel SHALL require the verification gate to pass before proceeding to the next step.
5. THE Kernel SHALL persist Agent loop state so that the loop can resume after interruption.

### Requirement 19: Extension-Model Validation via Example Skills

**User Story:** As a framework maintainer, I want any representative domain capability delivered purely as an example Skill through the Skill Extension Point, so that the extension model is validated to support arbitrary domains without touching the Kernel.

#### Acceptance Criteria

1. THE Framework SHALL provide a representative example Domain_Skill implemented through the Skill Extension Point.
2. WHEN a representative example Domain_Skill is enabled, THE Kernel SHALL make the Domain_Skill capability available to an Agent without changes to Kernel source code.
3. IF a representative example Domain_Skill requires a capability not available through Skills, MCP_Servers, or API_Tools, THEN THE Framework SHALL report the unsupported capability rather than adding the capability to the Kernel.

### Requirement 20: Python and uv Tooling

**User Story:** As a framework maintainer, I want the framework implemented in Python and managed with uv, so that the project matches the agreed technical stack.

#### Acceptance Criteria

1. THE Framework SHALL be implemented in Python.
2. THE Framework SHALL use uv for package and dependency management.
