"""loomable.display - Pretty-printing and result introspection helpers.

Terminal-friendly utilities for agent, delegation, and flow results.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "pp",
    "delegation_outputs",
    "step_outputs",
    "show_graph",
    "mermaid_graph",
]


def _tool_name_from_outcome(outcome: Any) -> str:
    """Extract the tool name recorded on a tool-activity outcome."""
    if getattr(outcome, "result", None) is not None:
        metadata = outcome.result.metadata or {}
        return str(metadata.get("tool_name", ""))
    if getattr(outcome, "error", None) is not None:
        details = getattr(outcome.error, "details", None) or {}
        return str(details.get("tool_name", ""))
    return ""


def _text_from_outcome(outcome: Any) -> str:
    """Extract displayable text from a tool-activity outcome."""
    if getattr(outcome, "result", None) is not None:
        content = outcome.result.content
        if content is None:
            return ""
        return str(content)
    if getattr(outcome, "error", None) is not None:
        return str(outcome.error.message)
    return ""


def pp(result: Any) -> None:
    """Pretty-print a run result (agent, flow, or team).

    Auto-detects :class:`~loomable.agent.run.RunResult` and prints output text,
    token usage, and tool-call counts.
    """
    if hasattr(result, "output") and callable(getattr(result.output, "text", None)):
        print("═══ Agent Result ═══")
        print("Output:")
        text = result.output.text()
        if text:
            for line in text.splitlines():
                print(f"  {line}")
        else:
            print("  (empty)")

        usage = getattr(result, "usage", None) or {}
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            print(f"Tokens: {input_tokens} in / {output_tokens} out")

        tool_activity = getattr(result, "tool_activity", None) or []
        if tool_activity:
            print(f"Tools called: {len(tool_activity)}")

        thoughts = getattr(result, "thoughts", None) or []
        if thoughts:
            print("Thoughts:")
            for thought in thoughts:
                for line in str(thought).splitlines() or [""]:
                    print(f"  {line}")

        plan = getattr(result, "plan", None)
        if plan:
            print("Plan:")
            for i, step in enumerate(plan, 1):
                print(f"  {i}. {step}")

        reasoning = getattr(result, "reasoning", None) or []
        if reasoning:
            print("Reasoning:")
            for segment in reasoning:
                for line in str(segment).splitlines() or [""]:
                    print(f"  {line}")

        sub_results = getattr(result, "sub_results", None)
        if sub_results:
            print(f"Steps: {len(sub_results)}")
        return

    print(result)


def delegation_outputs(result: Any) -> dict[str, str]:
    """Extract subagent outputs from ``delegate_to_*`` tool activity.

    Returns a mapping from role slug (without the ``delegate_to_`` prefix) to
    the subagent's text response.
    """
    outputs: dict[str, str] = {}
    for outcome in getattr(result, "tool_activity", None) or []:
        tool_name = _tool_name_from_outcome(outcome)
        if not tool_name.startswith("delegate_to_"):
            continue
        slug = tool_name[len("delegate_to_") :]
        outputs[slug] = _text_from_outcome(outcome)
    return outputs


def step_outputs(result: Any) -> dict[str, str]:
    """Extract per-node outputs from a flow :class:`~loomable.agent.run.RunResult`."""
    sub_results = getattr(result, "sub_results", None)
    if not sub_results:
        return {}

    outputs: dict[str, str] = {}
    for node_id, sub in sub_results.items():
        if hasattr(sub, "output") and callable(getattr(sub.output, "text", None)):
            outputs[node_id] = sub.output.text()
        else:
            outputs[node_id] = str(sub)
    return outputs


def _node_label(node_id: str, runnable: Any) -> str:
    """Build a Mermaid node label, including Agent role when available."""
    role = getattr(runnable, "_role", None)
    if role:
        safe_role = str(role).replace('"', "'")
        return f'{node_id}[{node_id}<br/><small>{safe_role}</small>]'
    return node_id


def mermaid_graph(flow: Any, *, title: str | None = None) -> str:
    """Return a Mermaid graph definition for a :class:`~loomable.flow.flow.Flow`."""
    nodes = getattr(flow, "nodes", None) or getattr(flow, "_nodes", None) or {}
    edges = getattr(flow, "edges", None) or getattr(flow, "_edges", None) or []

    lines = ["graph TD"]
    if title:
        lines.append(f"  %% {title}")

    if isinstance(nodes, dict):
        for node_id, node in nodes.items():
            runnable = getattr(node, "runnable", node)
            lines.append(f"  {_node_label(node_id, runnable)}")

    for edge in edges:
        source = getattr(edge, "source", None)
        target = getattr(edge, "target", None)
        if source is None and isinstance(edge, (tuple, list)) and len(edge) == 2:
            source, target = edge
        if source and target:
            lines.append(f"  {source} --> {target}")

    return "\n".join(lines)


def show_graph(flow: Any, *, title: str | None = None) -> None:
    """Print a flow graph as Mermaid syntax (paste into mermaid.live)."""
    print(mermaid_graph(flow, title=title))
