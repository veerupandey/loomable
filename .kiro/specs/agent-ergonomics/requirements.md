# Requirements Document

## Introduction

This feature raises the ergonomic ceiling of the `loomable` high-level API (`loomable.agent`) so common agent patterns are one-liners, while keeping the framework trimmed: every capability is an **additive edge layer** that composes existing kernel primitives and does **not** modify `loomable.kernel`. It closes the gaps identified after the first high-level API pass:

- **Flexible structured input** — pass a Pydantic model, dataclass, or dict directly as the agent input (agno-style), with an optional `input_schema` for validation.
- **Function tools** — a `@tool` decorator turns a plain Python function into a `Tool` with an auto-derived JSON schema, so users don't subclass `Tool`.
- **Automatic tool-use loop** — the agent actually calls its attached tools in a model→act→observe loop until it produces a final answer (today `arun` makes a single model call and never invokes tools).
- **Skills and MCP servers wired in** — `Agent(skills=[...])` and `Agent(mcp_servers=[...])` are honored (loaded/connected and exposed as tools), reusing the kernel `SkillLoader` and `MCPClient`.
- **Automatic memory compaction** — long conversations are summarized via the kernel `Summarizer` so context stays coherent and cheap.
- **Tiered model routing** — expose the kernel `ModelRouter` through the builder for cost/latency control.
- **Built-in embedders + knowledge** — an `Embedder` abstraction with OpenAI/Azure implementations plus a one-line `knowledge=[...]` that embeds, indexes, and auto-recalls via the kernel `LongTermStore`.

The stack is Python managed with uv. The Kernel remains stable and generic; this feature only touches `loomable.agent`, `loomable.content`, and `loomable.providers`.

## Glossary

- **Kernel**: The stable generic core (`loomable.kernel`), never modified for this feature.
- **High-Level API**: The ergonomic layer (`loomable.agent`) that composes kernel primitives.
- **Agent_Builder**: The high-level `Agent` entry point that assembles a `Built_Agent`.
- **Built_Agent**: The runnable agent produced by the Agent_Builder.
- **Agent_Input**: The message(s) supplied for a run; may be a string, `AgentInput`, Pydantic model, dataclass, or dict.
- **Input_Schema**: An optional Pydantic model or dataclass used to validate/coerce a dict or model input before a run.
- **Function_Tool**: A `Tool` produced by decorating a plain Python function with `@tool`, exposing an auto-derived JSON schema.
- **Tool_Use_Loop**: The automatic model→dispatch-tools→feed-results loop the agent runs when tools are attached.
- **Skill**: An Anthropic-style capability package loaded by the kernel `SkillLoader`; may bundle Script Tools.
- **MCP_Server**: An external Model Context Protocol server whose tools are exposed to the agent via the kernel `MCPClient`.
- **Memory_Compaction**: Summarizing older conversation turns (via the kernel `Summarizer`) into a compressed summary when the conversation exceeds a threshold.
- **Model_Router**: The kernel component that selects a model tier per call with fallback.
- **Embedder**: A component that turns text into an embedding vector; built-in OpenAI/Azure implementations are provided.
- **Knowledge_Source**: Text documents attached to an agent that are embedded, indexed in long-term memory, and recalled into context.

## Requirements

### Requirement 1: Flexible Structured Input

**User Story:** As an agent developer, I want to pass a Pydantic model, dataclass, or dict directly as the agent input, so that I don't have to hand-serialize structured data into a prompt.

#### Acceptance Criteria

1. THE Built_Agent run methods SHALL accept an Agent_Input that is a string, an `AgentInput`, a Pydantic model instance, a dataclass instance, or a dict.
2. WHEN a Pydantic model, dataclass, or dict is supplied, THE High_Level_API SHALL serialize it to JSON and present it as the current user message.
3. WHEN a string or `AgentInput` is supplied, THE High_Level_API SHALL use it unchanged (existing behavior).
4. WHERE an Input_Schema is configured, THE High_Level_API SHALL validate a dict or model input against the Input_Schema before the run.
5. IF an input fails Input_Schema validation, THEN THE High_Level_API SHALL raise an error identifying the validation failure before invoking the model.
6. WHERE an Input_Schema is configured and a plain string is supplied, THE High_Level_API SHALL pass the string through without schema validation.

### Requirement 2: Function Tools via Decorator

**User Story:** As a tool author, I want to turn a plain Python function into a tool with a decorator, so that I don't have to subclass the Tool base class.

#### Acceptance Criteria

1. THE High_Level_API SHALL provide a `@tool` decorator that produces a `Tool` from a plain Python function.
2. THE decorator SHALL derive the tool name from the function name and the description from the function docstring, allowing explicit overrides.
3. THE decorator SHALL derive a JSON input schema for the tool from the function's parameters and type annotations.
4. WHEN a Function_Tool is invoked with arguments, THE High_Level_API SHALL call the underlying function with those arguments and return its result as a `ToolResult`.
5. THE decorator SHALL support both synchronous and asynchronous functions.
6. IF a Function_Tool invocation raises an exception, THEN THE High_Level_API SHALL return the failure as a tool error identifying the tool.

### Requirement 3: Automatic Tool-Use Loop

**User Story:** As an agent developer, I want the agent to automatically call its attached tools and use the results, so that a single `arun` can complete multi-step tool-using tasks.

#### Acceptance Criteria

1. WHEN an agent with attached tools runs and the model requests tool calls, THE Built_Agent SHALL dispatch those tool calls and feed the results back to the model.
2. THE Built_Agent SHALL repeat the model→dispatch→feed-back cycle until the model produces a response with no further tool calls or a configured maximum iteration count is reached.
3. WHEN the loop ends, THE Built_Agent SHALL return the model's final response as the run output.
4. THE Built_Agent SHALL record the tool calls executed during the loop in the run result's tool activity.
5. WHERE tool hooks or guardrails are configured, THE Built_Agent SHALL apply them to tool calls dispatched within the loop.
6. WHEN a model returns no tool calls, THE Built_Agent SHALL behave exactly as a single-shot run (no extra model calls).
7. THE Tool_Use_Loop SHALL dispatch tool calls through the existing kernel tool runtime without modifying `loomable.kernel` source code.

### Requirement 4: Skills Attached Through the Builder

**User Story:** As an agent developer, I want to attach Skills through the builder, so that their bundled script tools are available to the agent without manual wiring.

#### Acceptance Criteria

1. THE Agent_Builder SHALL accept a set of Skill locations and load them through the kernel SkillLoader on build.
2. WHEN a Skill is loaded, THE Built_Agent SHALL register each of the Skill's Script Tools as an invocable tool.
3. IF a Skill fails to load, THEN THE Agent_Builder SHALL report a load error identifying the Skill and SHALL continue loading the remaining Skills.
4. THE Skills integration SHALL NOT require changes to `loomable.kernel` source code.

### Requirement 5: MCP Servers Attached Through the Builder

**User Story:** As an agent developer, I want to attach MCP servers through the builder, so that their tools are available to the agent without manual wiring.

#### Acceptance Criteria

1. THE Agent_Builder SHALL accept a set of MCP_Server specifications and connect to them through the kernel MCPClient.
2. WHEN an MCP_Server is connected, THE Built_Agent SHALL expose the server's enumerated tools as invocable tools.
3. IF an MCP_Server connection fails, THEN THE Agent_Builder SHALL report a connection error identifying the server and SHALL continue with the remaining servers.
4. THE MCP integration SHALL NOT require changes to `loomable.kernel` source code.

### Requirement 6: Automatic Memory Compaction

**User Story:** As an agent developer, I want long conversations compacted automatically, so that context stays coherent and token cost stays bounded.

#### Acceptance Criteria

1. WHERE conversation memory is enabled, THE Built_Agent SHALL track the number of retained conversation turns against a configured compaction threshold.
2. WHEN the retained turns exceed the compaction threshold, THE Built_Agent SHALL compress the oldest turns into a structured summary using the kernel Summarizer.
3. WHEN a summary is produced, THE Built_Agent SHALL retain the summary in place of the compacted raw turns and SHALL include it in the context supplied to subsequent runs.
4. THE Built_Agent SHALL preserve the most recent turns up to the configured window uncompacted.
5. THE Memory_Compaction SHALL reuse the kernel Summarizer without modifying `loomable.kernel` source code.

### Requirement 7: Tiered Model Routing Through the Builder

**User Story:** As an agent developer, I want tiered model routing exposed through the builder, so that I can balance cost and latency without hand-wiring the router.

#### Acceptance Criteria

1. THE Agent_Builder SHALL accept a tier configuration (tiers, tier policy, and fallback tiers) and route model calls through the kernel Model_Router.
2. WHEN tiered routing is configured, THE Built_Agent SHALL select a tier for each model call according to the configured policy.
3. IF a selected tier is unavailable, THEN THE Built_Agent SHALL route to a configured fallback tier and record the tier substitution.
4. WHERE no tier configuration is supplied, THE Built_Agent SHALL use the single configured model unchanged.
5. THE tiered routing integration SHALL reuse the kernel Model_Router without modifying `loomable.kernel` source code.

### Requirement 8: Built-in Embedders and Knowledge

**User Story:** As an agent developer, I want built-in embedders and a one-line way to attach knowledge, so that retrieval-augmented answers work without hand-writing embedding and indexing code.

#### Acceptance Criteria

1. THE High_Level_API SHALL define an Embedder abstraction and provide OpenAI-compatible and Azure OpenAI Embedder implementations.
2. THE Agent_Builder SHALL accept one or more Knowledge_Sources and an Embedder, embedding and indexing each source into the kernel Long-Term Store on build.
3. WHEN an agent with attached knowledge runs, THE Built_Agent SHALL recall the most relevant indexed content for the input and include it in the context supplied to the model.
4. IF an Embedder is unavailable, THEN THE High_Level_API SHALL return an error identifying the Embedder.
5. THE knowledge integration SHALL reuse the kernel Long-Term Store without modifying `loomable.kernel` source code.

### Requirement 9: Kernel Stability and Tooling

**User Story:** As a framework maintainer, I want these ergonomics added without touching the Kernel, so that the core stays stable and the stack is consistent.

#### Acceptance Criteria

1. THE feature SHALL be implemented in Python and managed with uv.
2. THE feature SHALL NOT modify the `loomable.kernel` package to achieve its goals.
3. THE `loomable.kernel` package tree SHALL import no module from `loomable.agent`, `loomable.content`, `loomable.serve`, or `loomable.providers`.
