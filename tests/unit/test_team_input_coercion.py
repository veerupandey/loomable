"""Team hard modes must accept AgentOutput like Agent does."""

from __future__ import annotations

import pytest

from loomable.agent.team import _input_as_text
from loomable.content import AgentOutput, Text


def test_team_input_as_text_from_agent_output() -> None:
    out = AgentOutput(parts=[Text("prior agent said hello")])
    text = _input_as_text(out)
    assert text == "prior agent said hello"
    assert "AgentOutput" not in text
    assert "MediaPart" not in text


def test_team_input_as_text_from_str() -> None:
    assert _input_as_text("plain") == "plain"
