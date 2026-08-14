"""loomable.kernel.skills - Anthropic-style Skills subsystem.

Implements progressive disclosure: only lightweight metadata (name, description)
is loaded at discovery time; full instruction body and script tools are
materialized only on load.

Skills are file-system folders anchored by a SKILL.md file containing YAML
frontmatter (name, description) and a Markdown instruction body, plus optional
bundled scripts.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loomable.kernel.contracts import Skill, Tool
from loomable.kernel.errors import ScriptToolError, SkillLoadError
from loomable.kernel.models import ToolResult


# ---------------------------------------------------------------------------
# Script tool specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScriptToolSpec:
    """Specification for a bundled script tool within a Skill."""

    name: str
    description: str
    path: Path


# ---------------------------------------------------------------------------
# Skill manifest (progressive disclosure - lightweight metadata)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkillManifest:
    """Lightweight metadata for a discovered Skill.

    Only the name, description, body_path, and script tool specs are captured
    at discovery time. The full body and script tools are materialized later
    on load.
    """

    name: str
    description: str
    body_path: Path
    script_tools: list[ScriptToolSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ScriptTool - Tool that executes a script in subprocess
# ---------------------------------------------------------------------------


class ScriptTool(Tool):
    """A tool that executes a bundled script in a subprocess.

    On invocation, the script is run with the provided arguments serialized
    as command-line arguments. stdout is captured as the tool result content.
    A non-zero exit code raises ScriptToolError naming the tool.
    """

    def __init__(self, name: str, description: str, script_path: Path) -> None:
        self.name = name
        self.description = description
        self._script_path = script_path

    async def invoke(self, args: dict[str, Any]) -> ToolResult:
        """Execute the script in a subprocess and return its output.

        Args are passed as command-line arguments in --key=value format.
        """
        cmd = [sys.executable, str(self._script_path)]
        for key, value in args.items():
            cmd.append(f"--{key}={value}")

        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            raise ScriptToolError(self.name)
        except OSError:
            raise ScriptToolError(self.name)

        if result.returncode != 0:
            raise ScriptToolError(self.name)

        return ToolResult(
            content=result.stdout.strip(),
            metadata={"exit_code": result.returncode, "stderr": result.stderr},
        )


# ---------------------------------------------------------------------------
# LoadedSkill - fully materialized Skill
# ---------------------------------------------------------------------------


class LoadedSkill(Skill):
    """A fully loaded Skill with body content and script tool instances."""

    def __init__(
        self,
        name: str,
        description: str,
        body: str,
        script_tools: list[str],
        tools: list[ScriptTool],
    ) -> None:
        self.name = name
        self.description = description
        self.body = body
        self.script_tools = script_tools
        self._tools = tools

    def get_tools(self) -> list[Tool]:
        """Return the Tool instances bundled with this Skill."""
        return list(self._tools)


# ---------------------------------------------------------------------------
# YAML frontmatter parser (minimal, no external dependency)
# ---------------------------------------------------------------------------


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Parse YAML frontmatter from a SKILL.md file.

    Expects the file to start with '---' followed by key: value lines,
    closed by '---'. Returns (frontmatter_dict, markdown_body).
    """
    lines = content.split("\n")

    if not lines or lines[0].strip() != "---":
        return {}, content

    # Find closing ---
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return {}, content

    # Parse key: value pairs from frontmatter
    frontmatter: dict[str, str] = {}
    for line in lines[1:end_idx]:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip().strip("\"'")

    # Body is everything after the closing ---
    body = "\n".join(lines[end_idx + 1 :]).strip()
    return frontmatter, body


# ---------------------------------------------------------------------------
# SkillLoader - discovery and loading
# ---------------------------------------------------------------------------


class SkillLoader:
    """Loads Anthropic-style Skills with progressive disclosure.

    - discover() scans directories for SKILL.md files and parses only
      lightweight metadata (name, description) from YAML frontmatter.
    - load() materializes the full body and registers script tools.
    """

    SKILL_FILENAME = "SKILL.md"
    SCRIPTS_DIR = "scripts"

    def discover(self, roots: list[Path]) -> list[SkillManifest]:
        """Scan directories for SKILL.md files and return manifests.

        Only parses YAML frontmatter for name/description (progressive
        disclosure). Does not load full body or scripts.

        Args:
            roots: List of root directories to scan for Skills.

        Returns:
            List of SkillManifest objects for each discovered Skill.
        """
        manifests: list[SkillManifest] = []

        for root in roots:
            if not root.is_dir():
                continue
            # Direct skill folder: <root>/SKILL.md
            direct_md = root / self.SKILL_FILENAME
            if direct_md.is_file():
                try:
                    content = direct_md.read_text(encoding="utf-8")
                    frontmatter, _ = _parse_frontmatter(content)
                    name = frontmatter.get("name", root.name)
                    description = frontmatter.get("description", "")
                    script_tool_specs = self._discover_scripts(root)
                    manifests.append(
                        SkillManifest(
                            name=name,
                            description=description,
                            body_path=direct_md,
                            script_tools=script_tool_specs,
                        )
                    )
                except Exception:
                    pass
                continue

            # Catalog folder: <root>/<skill>/SKILL.md
            for skill_dir in root.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / self.SKILL_FILENAME
                if not skill_md.is_file():
                    continue

                try:
                    content = skill_md.read_text(encoding="utf-8")
                    frontmatter, _ = _parse_frontmatter(content)

                    name = frontmatter.get("name", skill_dir.name)
                    description = frontmatter.get("description", "")

                    # Discover script tools
                    script_tool_specs = self._discover_scripts(skill_dir)

                    manifests.append(
                        SkillManifest(
                            name=name,
                            description=description,
                            body_path=skill_md,
                            script_tools=script_tool_specs,
                        )
                    )
                except Exception:
                    # Discovery is best-effort; skip malformed entries
                    continue

        return manifests

    def load(self, manifest: SkillManifest) -> LoadedSkill:
        """Load a Skill from its manifest, materializing the full body and tools.

        Raises SkillLoadError naming the skill on any failure.

        Args:
            manifest: The SkillManifest to load.

        Returns:
            A fully loaded LoadedSkill instance.
        """
        try:
            content = manifest.body_path.read_text(encoding="utf-8")
            _, body = _parse_frontmatter(content)

            # Create ScriptTool instances for each script tool spec
            tools: list[ScriptTool] = []
            for spec in manifest.script_tools:
                tool = ScriptTool(
                    name=spec.name,
                    description=spec.description,
                    script_path=spec.path,
                )
                tools.append(tool)

            return LoadedSkill(
                name=manifest.name,
                description=manifest.description,
                body=body,
                script_tools=[spec.name for spec in manifest.script_tools],
                tools=tools,
            )
        except Exception as exc:
            if isinstance(exc, SkillLoadError):
                raise
            raise SkillLoadError(manifest.name) from exc

    def _discover_scripts(self, skill_dir: Path) -> list[ScriptToolSpec]:
        """Discover bundled script tools in a Skill directory.

        Looks for executable scripts in the 'scripts/' subdirectory.
        Script tool names are derived from the filename (without extension).
        """
        scripts_dir = skill_dir / self.SCRIPTS_DIR
        if not scripts_dir.is_dir():
            return []

        specs: list[ScriptToolSpec] = []
        for script_file in sorted(scripts_dir.iterdir()):
            if script_file.is_file() and script_file.suffix in (".py", ".sh", ".bat"):
                tool_name = script_file.stem
                specs.append(
                    ScriptToolSpec(
                        name=tool_name,
                        description=f"Script tool: {tool_name}",
                        path=script_file,
                    )
                )

        return specs
