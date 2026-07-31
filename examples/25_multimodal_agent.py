"""25 — Multimodal Agent: Text + Image + Web Search + PDF Reading

Demonstrates:
- Sending text + image together in a single message
- Web search tool for live information
- PDF reading tool for document analysis
- All using the real Azure OpenAI multimodal model
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool, image
from loomable.content import AgentInput, Message, Modality, ModelCapabilities, Text, Image as ImagePart
from loomable.providers.openai import AzureOpenAIProvider


# ============================================================================
# Tools: Web Search + PDF Reader
# ============================================================================


@tool
def web_search(query: str) -> str:
    """Search the web for current information on a topic."""
    # In production, integrate with a real search API (Bing, Google, Tavily, etc.)
    results = {
        "python 3.13": "Python 3.13 was released October 2024 with a new JIT compiler and improved error messages.",
        "openai gpt-4o": "GPT-4o is OpenAI's multimodal model supporting text, image, and audio inputs.",
        "fastapi": "FastAPI is a modern Python web framework with automatic OpenAPI docs and async support.",
        "langchain vs crewai": "LangChain focuses on chains/agents; CrewAI focuses on multi-agent role-based collaboration.",
    }
    for key, val in results.items():
        if key in query.lower():
            return val
    return f"Search results for '{query}': Multiple relevant results found. Python remains the most popular language for AI/ML development."


@tool
def read_pdf(file_path: str) -> str:
    """Read and extract text content from a PDF file."""
    # In production, use PyPDF2, pdfplumber, or similar
    path = Path(file_path)
    if not path.exists():
        return f"Error: File '{file_path}' not found."
    if not path.suffix.lower() == ".pdf":
        return f"Error: '{file_path}' is not a PDF file."

    # Simulate PDF reading (in production: extract real text)
    return (
        f"[Extracted from {path.name}]\n"
        f"Document contains 15 pages covering system architecture, "
        f"API design patterns, and deployment guidelines. "
        f"Key sections: Introduction, Architecture Overview, "
        f"API Endpoints, Security Model, Deployment."
    )


@tool
def read_document(file_path: str) -> str:
    """Read content from a document file (PDF, TXT, MD)."""
    path = Path(file_path)
    if not path.exists():
        return f"Error: File '{file_path}' not found."

    suffix = path.suffix.lower()
    if suffix in (".txt", ".md", ".py", ".json", ".yaml", ".yml"):
        content = path.read_text(encoding="utf-8")
        # Truncate if too long
        if len(content) > 3000:
            content = content[:3000] + "\n... [truncated]"
        return content
    elif suffix == ".pdf":
        return read_pdf.invoke({"file_path": file_path})
    else:
        return f"Unsupported file type: {suffix}"


# ============================================================================
# Build the Multimodal Agent
# ============================================================================

provider = AzureOpenAIProvider()

agent = Agent(
    model=provider,
    capabilities=ModelCapabilities(
        input=frozenset({Modality.TEXT, Modality.IMAGE}),
        output=frozenset({Modality.TEXT}),
    ),
    instructions=(
        "You are a multimodal research assistant. You can:\n"
        "1. Analyze images sent to you\n"
        "2. Search the web for current information\n"
        "3. Read PDF and text documents\n\n"
        "When given an image, describe what you see. "
        "When asked about current topics, use web_search. "
        "When asked about documents, use read_document or read_pdf."
    ),
    tools=[web_search, read_pdf, read_document],
)


# ============================================================================
# Example 1: Text-only query with web search
# ============================================================================

print("=== Example 1: Web Search ===\n")
result = asyncio.run(agent.arun("What's new in Python 3.13?"))
print(f"Answer: {result.output.text()}\n")

# ============================================================================
# Example 2: Text + Image input (multimodal)
# ============================================================================

print("=== Example 2: Text + Image ===\n")

# High-level API — just pass images= with file paths:
result = asyncio.run(agent.arun(
    "I'm sending you a small test image. What can you tell me about it?",
    images=["examples/test_image.png"],
))
print(f"Answer: {result.output.text()}\n")

# ============================================================================
# Example 3: Reading a document
# ============================================================================

print("=== Example 3: Document Reading ===\n")

# Read the project's own README as a demo
result = asyncio.run(agent.arun("Read the file README.md and tell me what this project is about."))
print(f"Answer: {result.output.text()}\n")

# ============================================================================
# Summary
# ============================================================================

print("=== Capabilities Shown ===")
print("  ✓ Text input → standard chat")
print("  ✓ Image input → multimodal vision (via AgentInput + Image part)")
print("  ✓ Web search → @tool function (plug in Tavily/Bing/Google)")
print("  ✓ PDF reading → @tool function (plug in PyPDF2/pdfplumber)")
print("  ✓ All tools work in the automatic tool-use loop")
