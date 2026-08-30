"""Shared plan-step parsing for kernel Planner and Agent plan paths."""

from __future__ import annotations

import json
import re

__all__ = ["parse_plan_steps"]


def parse_plan_steps(text: str, *, max_steps: int = 5) -> list[str]:
    """Parse a planner response into a clean list of step strings."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "plan_steps" in data:
            data = data["plan_steps"]
        if isinstance(data, list):
            steps = [str(s).strip() for s in data if str(s).strip()]
            return steps[:max_steps] or [cleaned or "Complete the task"]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    steps: list[str] = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[.)]\s*", "", line)
        if line:
            steps.append(line)
    return steps[:max_steps] or [cleaned or "Complete the task"]
