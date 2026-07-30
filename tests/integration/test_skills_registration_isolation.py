# Feature: agent-ergonomics, Property 10

"""Property 10: Skills register their script tools with isolation.

For any set of Skills of which an arbitrary subset fails to load, building the
agent SHALL register the script tools of every loadable Skill and SHALL report a
SkillLoadError for each failing Skill without aborting the others.

**Validates: Requirements 4.1, 4.2, 4.3**
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings, strategies as st

from loomable.agent.builder import Agent, ModelSpec
from loomable.kernel.errors import SkillLoadError
from loomable.kernel.skills import SkillLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Minimal ModelProvider implementation (satisfies the structural protocol)."""

    async def invoke(self, request: Any) -> Any:
        from loomable.kernel.models import ModelResponse, MediaPart

        return ModelResponse(
            parts=[MediaPart(modality_type="text", data=b"ok")],
            usage={"input_tokens": 0, "output_tokens": 0},
        )


def _create_valid_skill(tmp_path: Path, name: str) -> Path:
    """Create a valid skill directory with a script tool under tmp_path/name."""
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    # SKILL.md with proper frontmatter
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        textwrap.dedent(f"""\
        ---
        name: {name}
        description: A test skill named {name}.
        ---

        # {name} Skill

        This is a test skill for integration testing.
        """),
        encoding="utf-8",
    )

    # scripts/ directory with a Python script tool
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    script_file = scripts_dir / f"{name}_tool.py"
    script_file.write_text(
        textwrap.dedent("""\
        import sys
        print("tool output")
        """),
        encoding="utf-8",
    )

    return skill_dir


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestSkillsRegistrationWithIsolation:
    """Integration tests for Property 10: Skills register their script tools with isolation."""

    def test_single_valid_skill_registers_tools(self, tmp_path: Path) -> None:
        """A single valid skill's script tools are registered in the built agent."""
        root = tmp_path / "skills_root"
        root.mkdir()
        _create_valid_skill(root, "alpha")

        built = Agent(model=_FakeProvider(), skills=[root]).build()

        # The script tool from the alpha skill should be registered
        assert "alpha_tool" in built.tool_runtime._tools
        assert built.skill_errors == []

    def test_multiple_valid_skills_all_registered(self, tmp_path: Path) -> None:
        """Multiple valid skills each have their script tools registered."""
        root = tmp_path / "skills_root"
        root.mkdir()
        _create_valid_skill(root, "alpha")
        _create_valid_skill(root, "beta")
        _create_valid_skill(root, "gamma")

        built = Agent(model=_FakeProvider(), skills=[root]).build()

        assert "alpha_tool" in built.tool_runtime._tools
        assert "beta_tool" in built.tool_runtime._tools
        assert "gamma_tool" in built.tool_runtime._tools
        assert built.skill_errors == []

    def test_broken_skill_reports_error_without_aborting_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken skill produces a SkillLoadError while valid skills still load."""
        root = tmp_path / "skills_root"
        root.mkdir()
        _create_valid_skill(root, "good_skill")
        _create_valid_skill(root, "bad_skill")

        # Monkeypatch load to fail only for "bad_skill"
        original_load = SkillLoader.load

        def _fail_bad(self_loader, manifest):
            if manifest.name == "bad_skill":
                raise SkillLoadError(manifest.name)
            return original_load(self_loader, manifest)

        monkeypatch.setattr(SkillLoader, "load", _fail_bad)

        built = Agent(model=_FakeProvider(), skills=[root]).build()

        # Good skill's tool is registered
        assert "good_skill_tool" in built.tool_runtime._tools
        # Bad skill's tool is NOT registered
        assert "bad_skill_tool" not in built.tool_runtime._tools
        # Error reported for the bad skill
        assert len(built.skill_errors) == 1
        assert built.skill_errors[0].skill_name == "bad_skill"

    def test_all_skills_broken_produces_errors_and_empty_registry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all skills fail to load, errors are collected and no tools registered."""
        root = tmp_path / "skills_root"
        root.mkdir()
        _create_valid_skill(root, "broken_a")
        _create_valid_skill(root, "broken_b")

        # Monkeypatch SkillLoader.load to always fail
        original_load = SkillLoader.load

        def _always_fail(self_loader, manifest):
            raise SkillLoadError(manifest.name)

        monkeypatch.setattr(SkillLoader, "load", _always_fail)

        built = Agent(model=_FakeProvider(), skills=[root]).build()

        # No skill tools registered
        # (only check no skill-originated tools; the agent may have internal tools)
        assert "broken_a_tool" not in built.tool_runtime._tools
        assert "broken_b_tool" not in built.tool_runtime._tools
        # Errors collected for each
        assert len(built.skill_errors) == 2
        error_names = {e.skill_name for e in built.skill_errors}
        assert "broken_a" in error_names
        assert "broken_b" in error_names

    def test_partial_failure_isolates_broken_and_loads_good(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mix of good and bad skills: good ones load, bad ones produce errors."""
        root = tmp_path / "skills_root"
        root.mkdir()
        _create_valid_skill(root, "good_one")
        _create_valid_skill(root, "bad_one")
        _create_valid_skill(root, "good_two")

        # Monkeypatch load to fail only for "bad_one"
        original_load = SkillLoader.load

        def _selective_fail(self_loader, manifest):
            if manifest.name == "bad_one":
                raise SkillLoadError(manifest.name)
            return original_load(self_loader, manifest)

        monkeypatch.setattr(SkillLoader, "load", _selective_fail)

        built = Agent(model=_FakeProvider(), skills=[root]).build()

        # Good skills registered
        assert "good_one_tool" in built.tool_runtime._tools
        assert "good_two_tool" in built.tool_runtime._tools
        # Bad skill not registered
        assert "bad_one_tool" not in built.tool_runtime._tools
        # Exactly one error for the bad skill
        assert len(built.skill_errors) == 1
        assert built.skill_errors[0].skill_name == "bad_one"
        assert isinstance(built.skill_errors[0], SkillLoadError)

    def test_skill_errors_are_skill_load_error_instances(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each reported error is a SkillLoadError naming the failed skill."""
        root = tmp_path / "skills_root"
        root.mkdir()
        _create_valid_skill(root, "will_fail")

        def _fail(self_loader, manifest):
            raise SkillLoadError(manifest.name)

        monkeypatch.setattr(SkillLoader, "load", _fail)

        built = Agent(model=_FakeProvider(), skills=[root]).build()

        assert len(built.skill_errors) == 1
        err = built.skill_errors[0]
        assert isinstance(err, SkillLoadError)
        assert err.skill_name == "will_fail"

    def test_loaded_skill_tools_are_usable_script_tools(self, tmp_path: Path) -> None:
        """Script tools from loaded skills are registered as callable Tool instances."""
        root = tmp_path / "skills_root"
        root.mkdir()
        _create_valid_skill(root, "usable")

        built = Agent(model=_FakeProvider(), skills=[root]).build()

        tool = built.tool_runtime._tools["usable_tool"]
        assert tool.name == "usable_tool"
        # Tool has an invoke method (it's a ScriptTool)
        assert hasattr(tool, "invoke")
        assert callable(tool.invoke)


# ---------------------------------------------------------------------------
# Property-based test (hypothesis): arbitrary subset failures
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    skill_names=st.lists(
        st.from_regex(r"[a-z]{3,8}", fullmatch=True),
        min_size=1,
        max_size=6,
        unique=True,
    ),
    # A boolean mask indicating which skills fail
    failure_mask=st.lists(st.booleans(), min_size=1, max_size=6),
)
def test_property_arbitrary_subset_failures(
    tmp_path_factory, skill_names: list[str], failure_mask: list[bool]
) -> None:
    """Property 10: For any set of skills with an arbitrary subset failing,
    the agent registers tools of loadable skills and reports SkillLoadError
    for each failing skill without aborting the others.

    **Validates: Requirements 4.1, 4.2, 4.3**
    """
    # Align failure_mask to skill_names length
    mask = failure_mask[: len(skill_names)]
    # Pad if mask is shorter
    while len(mask) < len(skill_names):
        mask.append(False)

    tmp_path = tmp_path_factory.mktemp("skills")
    root = tmp_path / "skills_root"
    root.mkdir()

    # Create all skills on disk
    for name in skill_names:
        _create_valid_skill(root, name)

    # Determine which skills should fail
    failing_names = {name for name, fails in zip(skill_names, mask) if fails}

    # Monkeypatch SkillLoader.load to fail for the selected subset
    original_load = SkillLoader.load

    def _controlled_fail(self_loader, manifest):
        if manifest.name in failing_names:
            raise SkillLoadError(manifest.name)
        return original_load(self_loader, manifest)

    # Use pytest's monkeypatch via a direct setattr (hypothesis doesn't have fixtures)
    original_method = SkillLoader.load
    SkillLoader.load = _controlled_fail
    try:
        built = Agent(model=_FakeProvider(), skills=[root]).build()

        # All non-failing skills should have their tools registered
        for name, fails in zip(skill_names, mask):
            tool_name = f"{name}_tool"
            if fails:
                assert tool_name not in built.tool_runtime._tools, (
                    f"Failing skill '{name}' should NOT have tool '{tool_name}' registered"
                )
            else:
                assert tool_name in built.tool_runtime._tools, (
                    f"Loadable skill '{name}' should have tool '{tool_name}' registered"
                )

        # SkillLoadErrors should be reported for each failing skill
        error_names = {e.skill_name for e in built.skill_errors}
        assert error_names == failing_names, (
            f"Expected errors for {failing_names}, got {error_names}"
        )

        # All errors are SkillLoadError instances
        for err in built.skill_errors:
            assert isinstance(err, SkillLoadError)

    finally:
        SkillLoader.load = original_method
