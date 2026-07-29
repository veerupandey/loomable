"""Unit tests for wiring skills= into the Agent builder (task 5.1).

Verify that:
- Agent(skills=[...]).build() discovers and loads skills, registering their
  script tools by name in the ToolRuntime.
- A failing skill is isolated: its SkillLoadError is captured in
  BuiltAgent.skill_errors while other skills load successfully.
- An agent without skills= has an empty skill_errors list and no extra tools.
- No kernel code is modified (Req 4.4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loomable.agent import Agent, BuiltAgent
from loomable.kernel.errors import SkillLoadError
from loomable.kernel.models import ModelRequest, ModelResponse


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ModelProvider implementation."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(content="ok")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skill_root_with_one_skill(tmp_path: Path) -> Path:
    """Create a skill root with a single valid skill containing one script tool."""
    root = tmp_path / "skills"
    root.mkdir()

    skill_dir = root / "greeter"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        '---\nname: greeter\ndescription: "Greets the user"\n---\n\n'
        "# Greeter\n\nSay hello.\n",
        encoding="utf-8",
    )

    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "say_hello.py").write_text(
        'import sys\nprint("Hello!")\n',
        encoding="utf-8",
    )

    return root


@pytest.fixture
def skill_root_with_multiple_skills(tmp_path: Path) -> Path:
    """Create a skill root with two valid skills."""
    root = tmp_path / "skills"
    root.mkdir()

    # Skill 1: greeter with say_hello tool
    skill1 = root / "greeter"
    skill1.mkdir()
    (skill1 / "SKILL.md").write_text(
        '---\nname: greeter\ndescription: "Greets"\n---\nGreet body.\n',
        encoding="utf-8",
    )
    scripts1 = skill1 / "scripts"
    scripts1.mkdir()
    (scripts1 / "say_hello.py").write_text(
        'print("hello")\n', encoding="utf-8"
    )

    # Skill 2: math with add tool
    skill2 = root / "math-skill"
    skill2.mkdir()
    (skill2 / "SKILL.md").write_text(
        '---\nname: math-skill\ndescription: "Math"\n---\nMath body.\n',
        encoding="utf-8",
    )
    scripts2 = skill2 / "scripts"
    scripts2.mkdir()
    (scripts2 / "add.py").write_text(
        'import sys\nprint("result")\n', encoding="utf-8"
    )

    return root


# ---------------------------------------------------------------------------
# Tests: Skills wired into the builder
# ---------------------------------------------------------------------------


class TestSkillsWiring:
    def test_skills_registers_script_tools(
        self, skill_root_with_one_skill: Path
    ) -> None:
        """A skill's script tools are registered in the ToolRuntime by name."""
        built = Agent(
            model=_FakeProvider(),
            skills=[skill_root_with_one_skill],
        ).build()

        assert "say_hello" in built.tool_runtime._tools
        assert built.skill_errors == []

    def test_multiple_skills_register_all_tools(
        self, skill_root_with_multiple_skills: Path
    ) -> None:
        """Multiple skills each register their script tools."""
        built = Agent(
            model=_FakeProvider(),
            skills=[skill_root_with_multiple_skills],
        ).build()

        assert "say_hello" in built.tool_runtime._tools
        assert "add" in built.tool_runtime._tools
        assert built.skill_errors == []

    def test_no_skills_means_no_extra_tools(self) -> None:
        """Without skills=, no skill tools are registered."""
        built = Agent(model=_FakeProvider()).build()

        assert built.skill_errors == []
        assert len(built.tool_runtime._tools) == 0

    def test_skills_coexists_with_explicit_tools(
        self, skill_root_with_one_skill: Path
    ) -> None:
        """Skills-loaded tools coexist with explicitly passed tools."""
        from loomable.agent.tools import tool

        @tool
        def my_explicit_tool(x: str) -> str:
            """An explicit tool."""
            return x

        built = Agent(
            model=_FakeProvider(),
            tools=[my_explicit_tool],
            skills=[skill_root_with_one_skill],
        ).build()

        assert "my_explicit_tool" in built.tool_runtime._tools
        assert "say_hello" in built.tool_runtime._tools

    def test_nonexistent_skill_root_produces_no_tools(
        self, tmp_path: Path
    ) -> None:
        """A nonexistent skill root path is silently ignored (no tools, no errors)."""
        built = Agent(
            model=_FakeProvider(),
            skills=[tmp_path / "nonexistent"],
        ).build()

        assert len(built.tool_runtime._tools) == 0
        assert built.skill_errors == []


class TestSkillLoadErrorIsolation:
    def test_broken_skill_error_isolated_via_monkeypatch(
        self, skill_root_with_multiple_skills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing skill yields SkillLoadError captured per-skill, others load."""
        from loomable.kernel.skills import SkillLoader

        original_load = SkillLoader.load

        def _failing_load(self, manifest):
            if manifest.name == "greeter":
                raise SkillLoadError(manifest.name)
            return original_load(self, manifest)

        monkeypatch.setattr(SkillLoader, "load", _failing_load)

        built = Agent(
            model=_FakeProvider(),
            skills=[skill_root_with_multiple_skills],
        ).build()

        # The math-skill's "add" tool loaded successfully
        assert "add" in built.tool_runtime._tools
        # The greeter skill failed — its tool is NOT registered
        assert "say_hello" not in built.tool_runtime._tools
        # The error is captured
        assert len(built.skill_errors) == 1
        assert built.skill_errors[0].skill_name == "greeter"

    def test_all_skills_fail_yields_multiple_errors(
        self, skill_root_with_multiple_skills: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all skills fail, all errors are collected, no tools registered."""
        from loomable.kernel.skills import SkillLoader

        def _always_fail(self, manifest):
            raise SkillLoadError(manifest.name)

        monkeypatch.setattr(SkillLoader, "load", _always_fail)

        built = Agent(
            model=_FakeProvider(),
            skills=[skill_root_with_multiple_skills],
        ).build()

        assert len(built.tool_runtime._tools) == 0
        assert len(built.skill_errors) == 2
        error_names = {e.skill_name for e in built.skill_errors}
        assert "greeter" in error_names
        assert "math-skill" in error_names

    def test_skill_errors_are_skill_load_error_instances(
        self, skill_root_with_one_skill: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each captured error is a SkillLoadError identifying the skill."""
        from loomable.kernel.skills import SkillLoader

        def _fail(self, manifest):
            raise SkillLoadError(manifest.name)

        monkeypatch.setattr(SkillLoader, "load", _fail)

        built = Agent(
            model=_FakeProvider(),
            skills=[skill_root_with_one_skill],
        ).build()

        assert len(built.skill_errors) == 1
        err = built.skill_errors[0]
        assert isinstance(err, SkillLoadError)
        assert err.skill_name == "greeter"
