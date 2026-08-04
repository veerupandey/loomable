# Feature: multimodal-io, Property 10: RunResult Text Property
"""Property 10: RunResult Text Property.

For any RunResult, the `.text` property SHALL return a value identical to
`self.output.text()`.

**Validates: Requirements 6.1**
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.agent.run import RunResult
from loomable.content import AgentOutput
from loomable.content.parts import MediaPart, Modality


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Generate arbitrary text strings including empty, single line, multiline, unicode
text_content = st.text(
    alphabet=st.characters(categories=("L", "M", "N", "P", "S", "Z")),
    min_size=0,
    max_size=200,
)


def _make_text_part(text: str) -> MediaPart:
    """Create a TEXT MediaPart from a string."""
    return MediaPart(
        modality=Modality.TEXT,
        media_type="text/plain",
        data=text.encode("utf-8"),
    )


def _make_image_part() -> MediaPart:
    """Create a non-text MediaPart (image with a URI) for variety."""
    return MediaPart(
        modality=Modality.IMAGE,
        media_type="image/png",
        uri="https://example.com/img.png",
    )


# Strategy: list of text strings to become text parts in AgentOutput
text_parts_strategy = st.lists(text_content, min_size=1, max_size=5)

# Strategy: whether to include non-text parts (images) in the output
include_non_text = st.booleans()


@st.composite
def agent_output_strategy(draw: st.DrawFn) -> AgentOutput:
    """Generate an AgentOutput with various combinations of text and non-text parts."""
    text_strings = draw(text_parts_strategy)
    add_images = draw(include_non_text)

    parts: list[MediaPart] = []

    # Interleave text and image parts for variety
    for text_str in text_strings:
        parts.append(_make_text_part(text_str))
        if add_images:
            parts.append(_make_image_part())

    # Ensure at least one part (AgentOutput requires non-empty parts)
    if not parts:
        parts.append(_make_text_part(""))

    return AgentOutput(parts=parts)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


class TestRunResultTextProperty:
    """RunResult.text always equals self.output.text()."""

    @settings(max_examples=100)
    @given(output=agent_output_strategy())
    def test_text_equals_output_text(self, output: AgentOutput) -> None:
        """For any RunResult, .text returns the same value as output.text()."""
        run_result = RunResult(
            output=output,
            session_id="test-session",
        )

        assert run_result.text == run_result.output.text()
