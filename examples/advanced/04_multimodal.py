"""Multimodal Agent — Image input, tool media output, and feedback.

USE WHEN: Your agent needs to analyze images/videos/audio OR when tools
generate media (charts, diagrams, processed images) that the model
should reason about in follow-up steps.

Covers:
  1. Image/video INPUT — pass files, URLs, or bytes to the agent
  2. Tool media OUTPUT — tools return Image/Audio/Video objects directly
  3. RunResult convenience — access result.text, result.images, etc.
  4. Feedback injection — model "sees" tool-generated media automatically
  5. Audio INPUT — pass audio files for transcription/analysis
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from loomable.agent import Agent, tool, Image
from loomable.display import pp
from loomable.providers.openai import AzureOpenAIProvider

provider = AzureOpenAIProvider()


# ============================================================
# Part 1: Image INPUT (existing patterns — still fully supported)
# ============================================================

agent = Agent(
    model=provider,
    role="Visual Analyst",
    goal="Analyze images and describe their contents",
    instructions="Describe what you see in detail. Be specific about visual elements.",
    multimodal=True,  # enables image + text input
)

# --- Image from file path (simplest) ---
test_image = Path(__file__).parent.parent / "test_image.png"

if test_image.exists():
    result = asyncio.run(agent.arun(
        "Describe this image in detail.",
        images=[str(test_image)],  # just pass a file path string
    ))
    pp(result)
else:
    # Fallback when no image is available
    result = asyncio.run(agent.arun("What would a sunset over mountains look like?"))
    pp(result)

# --- Other ways to provide images ---
# From URL (auto-detected):
#   images=["https://example.com/chart.png"]
#
# Multiple images:
#   images=["chart1.png", "chart2.png"]
#
# From raw bytes:
#   images=[open("photo.jpg", "rb").read()]
#
# Explicit control via helper:
#   from loomable.agent import image
#   images=[image(path="photo.jpg"), image(uri="https://...")]
#
# Video input (same auto-coercion rules):
#   videos=["demo.mp4"]
#
# Audio input (requires model audio capability):
#   audio=["recording.wav"]


# ============================================================
# Part 2: Tool media OUTPUT — tools that generate images
# ============================================================

# A @tool function can return an Image (or Audio, Video) directly.
# The framework automatically detects media return values, stores them
# in the tool result metadata, and surfaces them on RunResult.

@tool
def generate_chart(title: str, chart_type: str) -> Image:
    """Generate a chart image based on the given title and type."""
    # In real usage, this would call matplotlib, DALL-E, or any image API.
    # Here we simulate with a small PNG (1x1 pixel placeholder).
    import struct
    import zlib

    # Minimal valid 1x1 red PNG for demonstration
    width, height = 1, 1
    raw_data = b"\x00\xff\x00\x00"  # filter byte + RGB
    compressed = zlib.compress(raw_data)

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr_data)
        + png_chunk(b"IDAT", compressed)
        + png_chunk(b"IEND", b"")
    )

    # Return an Image instance — the framework handles the rest
    return Image(content=png_bytes, format="png")


# You can also return an Image from a URL (no bytes needed):
@tool
def fetch_diagram(topic: str) -> Image:
    """Fetch a diagram illustrating the given topic."""
    # In production this might call an API and return the result URL
    return Image(url=f"https://example.com/diagrams/{topic}.png")


chart_agent = Agent(
    model=provider,
    role="Data Visualization Assistant",
    goal="Create charts and explain data patterns",
    instructions="When asked to visualize data, use the generate_chart tool.",
    tools=[generate_chart, fetch_diagram],
    multimodal=True,
)

result = asyncio.run(chart_agent.arun("Create a bar chart showing Q1 sales"))

# --- Accessing results via convenience properties ---
# result.text — the model's text response (same as result.output.text())
print("Model response:", result.text)

# result.images — all images (model-generated + tool-generated), as Image objects
if result.images:
    print(f"Got {len(result.images)} image(s) from the run")
    # Save the first image to disk
    result.images[0].save("output.png")
    print("Saved to output.png")


# ============================================================
# Part 3: Feedback injection — model reasons about tool output
# ============================================================

# With feedback_media=True (the default), when a tool generates an image
# the model can "see" it in the next reasoning step. This enables multi-step
# workflows: generate → analyze → refine.

feedback_agent = Agent(
    model=provider,
    role="Creative Director",
    goal="Generate images and iteratively refine them based on analysis",
    instructions=(
        "Use generate_chart to create visuals. After each image is generated, "
        "analyze what you see and suggest improvements. If the user asks for "
        "refinement, generate again with adjustments."
    ),
    tools=[generate_chart],
    multimodal=True,
    feedback_media=True,  # model sees tool-generated images (default behavior)
)

# The model will: 1) call generate_chart → 2) see the resulting image →
# 3) describe what it observes → 4) potentially call the tool again
result = asyncio.run(feedback_agent.arun(
    "Create a chart of monthly revenue, then tell me if it looks clear"
))
print("\nFeedback agent response:", result.text)

# To DISABLE feedback (tool media won't be shown to the model, but still
# accessible on result.images):
#
#   agent = Agent(..., feedback_media=False)
#
# This is useful when you only want the end-user to see the image,
# not spend tokens having the model analyze it.


# ============================================================
# Part 4: Audio INPUT (for models that support it)
# ============================================================

# Pass audio files just like images — same auto-coercion rules apply.
# If the model doesn't support audio input, UnsupportedModalityError is raised
# before any API call is made.

# from loomable.media import Audio
#
# result = await agent.arun("Transcribe this recording", audio=["./meeting.wav"])
# result = await agent.arun("Summarize", audio=[Audio(url="https://example.com/podcast.mp3")])
# result = await agent.arun("Analyze tone", audio=[audio_bytes])
