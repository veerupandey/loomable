# Loomable API

Public beta **0.2.0b0**. One contract: `arun` → `RunResult`.

| Use | API |
|-----|-----|
| One model + tools | `Agent` |
| Named orchestration | `Team` |
| Multi-step process | `Workflow` |
| Goal + WorkItems board | `Case` |
| Long-horizon research/code | `create_deep_agent(profile=...)` |
| Memory | `Memory.compose` |
| Searchable RAG | `knowledge_base=` / `retrievers=` |

Prefer `Workflow` / `Team`. Low-level `Flow` / `loomable.flow.helpers` are an escape hatch. Breaking changes are in [CHANGELOG.md](../CHANGELOG.md).

Beta limits: local workspace FS, cooperative cancel (not hard-kill of provider HTTP), serve auth is a shared API key (not RBAC).

---

## Agent

```python
from loomable import Agent, tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return f"Results for: {query}"

agent = Agent(
    model="openai:gpt-4o-mini",  # or "anthropic:...", "gemini:...", provider instance
    role="Researcher",
    goal="Find accurate information",
    instructions="Be concise.",
    tools=[search],
    knowledge_base=["./handbook.pdf"],  # vector DB → search_knowledge
)
result = await agent.arun("When was Python created?")
print(result.output.text())
```

`run()` is a sync wrapper. With tools, the tool loop runs automatically; without tools, a single model call. `complexity_router=` is opt-in.

```python
result.output.text()
result.session_id
result.usage            # {"input_tokens", "output_tokens"}
result.structured       # when response_model= is set
result.tool_activity
result.verification     # when verifier= is set
result.trace            # debug=True or events=JSONTracer()
```

Persona fields (`role`, `goal`, `instructions`, `description`) assemble into the system prompt. `role` is reused in delegation labels.

**Require tools:** `require_tools=["write_file:output/x.md"]` nudges until those side effects happen. `strict_require_tools=True` raises `RequireToolsError`. Same knobs inherit onto Agent steps via `Workflow(require_tools=...)` or `.step(..., require_tools=...)`.

**Verifier:** `verifier=` is a callable `(output, context) -> bool` or a `Verifier`. `retry_on_failure=True` + `max_verify_retries=` re-asks with feedback.

**HITL (tools):** `require_confirmation=["send_email"]` + `approver=` (default deny-all, headless-safe). Workflow step HITL is separate (`confirm=True`).

**Cancel:** `agent.cancel()` / `built.cancel()` — cooperative at tool-loop boundaries.

---

## Team

```python
from loomable import Agent, Team

team = Team(
    members=[researcher, writer, critic],
    model="openai:gpt-4o-mini",
    mode="coordinate",  # coordinate | route | broadcast | sequential
)
result = await team.arun("Review our API design")
```

| Mode | Default | Behavior |
|------|---------|----------|
| `coordinate` | soft | LLM delegates to all members, synthesizes |
| `route` | soft | LLM picks one member |
| `broadcast` | hard | Same input to all, merge labeled results |
| `sequential` | hard | Chain members in order |

`hard=True` is only valid with `broadcast` / `sequential`. Soft `coordinate` auto-requires `delegate_to_*` and runs skipped members (`metadata["team_coordinate_fallback"]`).

Or pass `subagents=[...]` on a parent `Agent` — each becomes `delegate_to_<role>`. Nested subagents are allowed.

---

## Workflow

```python
from loomable import Agent, Workflow, FlowPaused, JsonFileCheckpointer

wf = (
    Workflow("article", session_id="job-1", checkpointer=JsonFileCheckpointer("./ckpts"))
    .step("research", researcher)
    .parallel(analyst=analyst, visual=visual)
    .branch(when=needs_human, then=approver, else_=auto)
    .loop(polisher, until=quality_ok, max_iterations=3)
    .step("publish", publisher, confirm=True)
)
try:
    result = await wf.arun("AI agents in 2025")
except FlowPaused:
    await wf.approve("publish")
    result = await wf.arun(resume=True)

print(wf.explain())
print(wf.state.get("research").text())
```

Declarative: `Workflow("pipe", steps=[Step("a", a), Step("b", b)])`.

`confirm=True` requires `checkpointer=` + `session_id=`. Not supported inside `.parallel()` / `.branch()` / `.loop()`.

`Workflow(memory=True)` is a callable-step blackboard on `RunContext.memory` — **not** Agent chat memory. Agent steps share via SharedState output chaining.

`Loop(body=agent, verifier=..., max_iterations=3)` is a Runnable on its own. Workflow uses `.loop(..., until=)`.

---

## Case

```python
from loomable import Case

case = Case(
    model=provider,
    goal="Close INC-88421 with SEV packet",
    board=True,           # open → in_progress → blocked → done
    dispatch="spawn",     # or "reuse"
    accept=lambda out, ctx: "SEV-" in out.text(),
    max_rounds=3,
)
result = await case.arun(email)
print(result.metadata["board"])

# Same pipeline:
Agent(model=provider, mode="case", dispatch="reuse", accept=check)
```

`Case.as_workflow()` returns a nestable `Workflow`. Board mutations stream as `STATE_SNAPSHOT` / `STATE_DELTA`. Case has no `astream` / NDJSON — use `astream_events` / SSE. `checkpointer=` / `max_rounds=` / `dispatch=` on `Agent` require `mode="case"`.

---

## Deep agent

```python
from loomable import create_deep_agent

agent = create_deep_agent(
    model="openai:gpt-4o-mini",
    profile="research",          # skills=["research"] + report/citation gates
    workspace="./.deep_workspace",
    knowledge_base=store,        # same Agent kwarg
)
await agent.arun("Research the topic; write reports/brief.md")

agent = create_deep_agent(model, profile="code", repo="./my-app")
```

Planning (`TodoTools`), local workspace FS, `task` / `task_batch` specialists, skills (`load_skill`), discovery (`search_tools` / `activate_tool`). `discovery_core="research-slim"` is experimental. Sandbox: `code_exec=True` / `shell=True`. Case-only kwargs (`dispatch`, `max_rounds`, `checkpointer`) require `mode="case"`. `board=False` is allowed without case mode.

---

## Memory

```python
from loomable import Agent, Memory, ConversationMemory, UserMemory, open_session_store, open_vector_store
from loomable.agent import NoteStore
from loomable.providers import OpenAIEmbedder

notes = NoteStore(long_term=open_vector_store(engine="memory"), embedder=OpenAIEmbedder())
memory = Memory.compose(
    conversation=ConversationMemory(store=open_session_store("sqlite", path="sessions.db"), window=8),
    user=UserMemory(note_store=notes, memory_tool=True, auto_extract=True),
)
agent = Agent(model=..., memory=memory, session_id="conv-1", user_id="alice",
              scopes={"claim_id": "CLM-4421"})
```

| Layer | Class | Stores |
|-------|--------|--------|
| Conversation | `ConversationMemory` | L1 turns + L2 summaries for `session_id` |
| User | `UserMemory` | Cross-session notes (`NoteStore`, scoped) |
| Knowledge | `KnowledgeMemory` | Passive RAG, or `knowledge_base` → `search_*` |
| Working | `WorkingMemory` | `Workflow(memory=True)` blackboard — not `Agent(memory=)` |

Do not pass both `memory=` and flat `session_store=` / `note_store=` / `memory_backend=`. `resume=True` means the session row must already exist. `memory_tool=True` / `UserMemory(auto_extract=True)` without a note store raises. `user_id` / `scopes` stamp UserMemory (`MemoryScope.of(...)`).

Default L3 vector store is Alibaba zvec (`pip install 'loomable[zvec]'`, `.loomable/memory_zvec`). Also: `engine="faiss"|"chroma"|"milvus"|"memory"` or `postgres_url=`. Embedders: `OpenAIEmbedder`, `GeminiEmbedder`, `AzureOpenAIEmbedder`, `HuggingFaceEmbedder`.

Same compose object on Team (coordinator) and Case (`from_agent` copies memory). Team has no `scopes=` — set on members or coordinator memory.

---

## Knowledge / RAG

```python
agent = Agent(model=..., knowledge_base=["./handbook.pdf", "./runbooks"])
# named collections → search_personal, search_company
agent = Agent(model=..., knowledge_base={"personal": ["./notes"], "company": store})
agent = Agent(model=..., knowledge_base=store, retrievers=[custom_retriever])

# Passive snippets (no tool) — requires embedder=
agent = Agent(model=..., knowledge=["FAQ..."], embedder=OpenAIEmbedder(), knowledge_top_k=5)
```

`knowledge=` without `embedder=` raises. Prefer `knowledge_base=` for search tools. Team / Case / Workflow inherit `knowledge_base=` onto Agent members that do not already have one.

`loomable.retrieval` (`ingest`, `build_agentic_retriever`) is experimental — prefer `knowledge_base=` / `retrievers=` on Agent. See `examples/advanced/06_agentic_retriever.py`.

---

## Models

```python
Agent(model="openai:gpt-4o-mini")
Agent(model="anthropic:claude-sonnet-4-20250514")
Agent(model="gemini:gemini-2.0-flash")
Agent(model="groq:llama-3.3-70b-versatile")
Agent(model="ollama:mistral")
Agent(model="azure:gpt-4.1-mini")
Agent(model="gpt-4o-mini")  # bare name → OpenAI
```

Keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`, `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_DEPLOYMENT_NAME`.

OpenAI-compatible: `OpenAIProvider(model="llama-3", base_url="http://localhost:8000/v1")`.

---

## Structured output

```python
from pydantic import BaseModel

class CityInfo(BaseModel):
    name: str
    population: int

agent = Agent(model="openai:gpt-4o-mini", response_model=CityInfo)
city = agent.run("Info about Tokyo").structured
# or per-call: await agent.arun("...", output_schema=CityInfo)
```

---

## Multimodal

Media is on by default. Restrict with `modalities="text"` or `text_only=True`.

```python
from loomable.media import Image, Audio, Video, File

result = await agent.arun("Describe this", images=["./chart.png"])
result = await agent.arun("Summarize", videos=["./demo.mp4"], audio=["./clip.wav"])
result.images[0].save("./out.png")   # also result.audio / result.videos / result.files
```

`Image` / `Audio` / `Video` / `File` take exactly one of `url=`, `filepath=`, `content=`. Tools may return media; `feedback_media=True` (default) injects it into later turns when the model supports that modality.

---

## Streaming, SSE, serving

```python
async for chunk in agent.astream("hello"):
    if chunk.delta.data:
        print(chunk.delta.data.decode(), end="")
```

`astream` is token-level only for single-shot (no tools). With tools it falls back to `arun` then chunks. Case / `mode="case"` do not support `astream`.

```python
async for ev in agent.astream_events(prompt):
    print(ev.type)  # RUN_* / TEXT_* / TOOL_*
async for ev in workflow.astream_events(prompt):
    ...  # + NODE_*
async for ev in case.astream_events(prompt):
    ...  # + STATE_*
```

| Family | Events |
|--------|--------|
| Lifecycle | `RUN_STARTED`, `RUN_FINISHED`, `RUN_ERROR` |
| Text | `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_END` |
| Tools | `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END`, `TOOL_CALL_RESULT` |
| Graph | `NODE_STARTED`, `NODE_FINISHED` |
| State | `STATE_SNAPSHOT`, `STATE_DELTA` |

```python
from fastapi import FastAPI
from loomable.serve import mount_agent, mount_case

app = FastAPI()
mount_agent(app, agent, prefix="/agent", api_key="secret")
mount_case(app, case, prefix="/cases", api_key="secret")
# POST /agent/run          JSON
# POST /agent/run/events   SSE  (disconnect → cancel)
# POST /agent/run/stream   NDJSON, Agent only (omitted for Case / mode=case)
# POST /cases/run          JSON
# POST /cases/run/events   SSE
```

Auth when `api_key=` is set: `Authorization: Bearer …` or `X-API-Key`. No `mount_team` / `mount_workflow`. See [SECURITY.md](../SECURITY.md).

---

## Checkpointing

```python
from loomable.persist import JsonFileCheckpointer, SQLiteCheckpointer, CheckpointListener, CheckpointConfig
from loomable import PostgresCheckpointer

cp = JsonFileCheckpointer(".checkpoints", max_checkpoints=20)
# SQLiteCheckpointer("agent.db")
# PostgresCheckpointer("postgresql://loomable:loomable@127.0.0.1:5432/loomable")  # loomable[postgres]

built = agent.build()
built.events = CheckpointListener(
    checkpointer=cp,
    config=CheckpointConfig(on_events=["run_end", "tool_call"], max_checkpoints=10),
    thread_id="session-123",
)
```

`PostgresCheckpointer` is for Case/Workflow resume — not Agent chat history.

---

## Production knobs

All opt-in.

```python
from loomable.providers import RetryPolicy

agent = Agent(
    model="openai:gpt-4o-mini",
    tools=[search],
    resilience=RetryPolicy(max_attempts=3, base_delay=0.5),
    tool_timeout=5.0,
    tool_concurrency=3,
    token_budget=8192,
    loop_repeat_threshold=3,
    think_tool=True,
    plan_tool=True,
    debug=True,
    mcp_servers=[
        {"command": "uvx", "args": ["some-mcp-server"]},
        {"url": "http://localhost:8080/sse", "headers": {"Authorization": "Bearer ..."}},
    ],
)
```

---

## Display

```python
from loomable.display import pp, delegation_outputs, step_outputs, show_graph, mermaid_graph

pp(result)  # RunResult only; anything else is print()
delegation_outputs(result)["researcher"]
step_outputs(result)["research"]          # Workflow step names, not node_0
show_graph(wf.flow)                       # prints Mermaid (paste into mermaid.live)
```

---

## Flow (escape hatch)

Prefer `Workflow.step` / `.parallel` / `.branch` / `.map`. `Flow`, `Node`, `Edge`, engines, and `loomable.flow.helpers` (`sequential`, `parallel`, `route`, `coordinate`) remain for power users. `FlowClass` / `start` / `listen` / `router` are experimental.

```python
from loomable.flow import Flow, Loop, Workflow, Step, FlowPaused
from loomable.flow.helpers import sequential  # advanced only
```

---

## Removed names (no shims)

| Old | Use |
|-----|-----|
| `Agent(multimodal=)` | `modalities=` / `text_only=` |
| `Memory.compose(short=` / `long=`) | `conversation=` / `user=` |
| `Memory.with_user_id()` | `with_scopes(user_id=...)` |
| `Loop(end_condition=)` | `verifier=` (Workflow: `until=`) |
| `from loomable import sequential` | `Workflow` / `Team` |
| `Agent(memory=MemoryManager())` | `Memory.compose(...)` |
| `HITLPause` | `FlowPaused` |
