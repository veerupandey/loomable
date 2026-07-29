"""Unit tests for the Skills subsystem (loomable.kernel.skills).

Tests cover:
- SkillLoader.discover() with progressive disclosure (only metadata loaded)
- SkillLoader.load() materializing full body and script tools
- ScriptTool invocation via subprocess
- SkillLoadError on failing Skills (isolation: others continue)
- ScriptToolError on script execution failure
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from loomable.kernel.errors import ScriptToolError, SkillLoadError
from loomable.kernel.skills import (
    LoadedSkill,
    ScriptTool,
    ScriptToolSpec,
    SkillLoader,
    SkillManifest,
    _parse_frontmatter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skill_root(tmp_path: Path) -> Path:
    """Create a skill root directory with a well-formed skill."""
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        '---\nname: my-skill\ndescription: "A test skill"\n---\n\n'
        "# Instructions\n\nDo the thing.\n",
        encoding="utf-8",
    )

    # Add a script tool
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    script_file = scripts_dir / "echo_tool.py"
    script_file.write_text(
        'import sys\nprint("hello from echo_tool")\n',
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def skill_root_multiple(tmp_path: Path) -> Path:
    """Create a root with multiple skills, one malformed."""
    # Good skill
    good_dir = tmp_path / "good-skill"
    good_dir.mkdir()
    (good_dir / "SKILL.md").write_text(
        "---\nname: good-skill\ndescription: Good one\n---\n\nGood body.\n",
        encoding="utf-8",
    )

    # Another good skill
    good2_dir = tmp_path / "second-skill"
    good2_dir.mkdir()
    (good2_dir / "SKILL.md").write_text(
        "---\nname: second-skill\ndescription: Second\n---\n\nSecond body.\n",
        encoding="utf-8",
    )

    # Bad skill (not a directory)
    (tmp_path / "not-a-dir.txt").write_text("nope", encoding="utf-8")

    return tmp_path


@pytest.fixture
def echo_script(tmp_path: Path) -> Path:
    """Create a simple echo script that prints its args."""
    script = tmp_path / "echo.py"
    script.write_text(
        "import sys\nprint(' '.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    return script


@pytest.fixture
def failing_script(tmp_path: Path) -> Path:
    """Create a script that exits with non-zero."""
    script = tmp_path / "fail.py"
    script.write_text(
        "import sys\nsys.exit(1)\n",
        encoding="utf-8",
    )
    return script


# ---------------------------------------------------------------------------
# Tests: frontmatter parser
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_valid_frontmatter(self) -> None:
        content = '---\nname: test\ndescription: "Hello"\n---\n\nBody text.'
        fm, body = _parse_frontmatter(content)
        assert fm["name"] == "test"
        assert fm["description"] == "Hello"
        assert body == "Body text."

    def test_no_frontmatter(self) -> None:
        content = "Just body text."
        fm, body = _parse_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_unclosed_frontmatter(self) -> None:
        content = "---\nname: test\nNo closing marker"
        fm, body = _parse_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_empty_body(self) -> None:
        content = "---\nname: test\n---\n"
        fm, body = _parse_frontmatter(content)
        assert fm["name"] == "test"
        assert body == ""


# ---------------------------------------------------------------------------
# Tests: SkillLoader.discover()
# ---------------------------------------------------------------------------


class TestSkillLoaderDiscover:
    def test_discovers_skill_metadata(self, skill_root: Path) -> None:
        loader = SkillLoader()
        manifests = loader.discover([skill_root])

        assert len(manifests) == 1
        m = manifests[0]
        assert m.name == "my-skill"
        assert m.description == "A test skill"
        assert m.body_path == skill_root / "my-skill" / "SKILL.md"

    def test_discovers_script_tools(self, skill_root: Path) -> None:
        loader = SkillLoader()
        manifests = loader.discover([skill_root])

        assert len(manifests) == 1
        m = manifests[0]
        assert len(m.script_tools) == 1
        assert m.script_tools[0].name == "echo_tool"
        assert m.script_tools[0].path.name == "echo_tool.py"

    def test_discovers_multiple_skills(self, skill_root_multiple: Path) -> None:
        loader = SkillLoader()
        manifests = loader.discover([skill_root_multiple])

        names = {m.name for m in manifests}
        assert "good-skill" in names
        assert "second-skill" in names
        assert len(manifests) == 2

    def test_ignores_nonexistent_root(self, tmp_path: Path) -> None:
        loader = SkillLoader()
        manifests = loader.discover([tmp_path / "no-such-dir"])
        assert manifests == []

    def test_ignores_dirs_without_skill_md(self, tmp_path: Path) -> None:
        (tmp_path / "empty-dir").mkdir()
        loader = SkillLoader()
        manifests = loader.discover([tmp_path])
        assert manifests == []

    def test_multiple_roots(self, tmp_path: Path) -> None:
        root1 = tmp_path / "root1"
        root1.mkdir()
        skill1 = root1 / "skill-a"
        skill1.mkdir()
        (skill1 / "SKILL.md").write_text(
            "---\nname: skill-a\ndescription: A\n---\nBody A\n",
            encoding="utf-8",
        )

        root2 = tmp_path / "root2"
        root2.mkdir()
        skill2 = root2 / "skill-b"
        skill2.mkdir()
        (skill2 / "SKILL.md").write_text(
            "---\nname: skill-b\ndescription: B\n---\nBody B\n",
            encoding="utf-8",
        )

        loader = SkillLoader()
        manifests = loader.discover([root1, root2])
        names = {m.name for m in manifests}
        assert names == {"skill-a", "skill-b"}


# ---------------------------------------------------------------------------
# Tests: SkillLoader.load()
# ---------------------------------------------------------------------------


class TestSkillLoaderLoad:
    def test_loads_full_body(self, skill_root: Path) -> None:
        loader = SkillLoader()
        manifests = loader.discover([skill_root])
        skill = loader.load(manifests[0])

        assert isinstance(skill, LoadedSkill)
        assert skill.name == "my-skill"
        assert skill.description == "A test skill"
        assert "# Instructions" in skill.body
        assert "Do the thing." in skill.body

    def test_loads_script_tools(self, skill_root: Path) -> None:
        loader = SkillLoader()
        manifests = loader.discover([skill_root])
        skill = loader.load(manifests[0])

        tools = skill.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"

    def test_load_failure_raises_skill_load_error(self, tmp_path: Path) -> None:
        manifest = SkillManifest(
            name="broken-skill",
            description="Will fail",
            body_path=tmp_path / "nonexistent" / "SKILL.md",
            script_tools=[],
        )
        loader = SkillLoader()

        with pytest.raises(SkillLoadError) as exc_info:
            loader.load(manifest)

        assert exc_info.value.skill_name == "broken-skill"

    def test_isolation_other_skills_load_despite_failure(
        self, skill_root_multiple: Path
    ) -> None:
        """A failing skill does not prevent other skills from loading."""
        loader = SkillLoader()
        manifests = loader.discover([skill_root_multiple])

        # Corrupt one manifest to simulate failure
        good_manifests = [m for m in manifests if m.name == "good-skill"]
        bad_manifest = SkillManifest(
            name="broken",
            description="will fail",
            body_path=skill_root_multiple / "nonexistent" / "SKILL.md",
            script_tools=[],
        )

        loaded = []
        errors = []
        for m in [bad_manifest] + good_manifests:
            try:
                loaded.append(loader.load(m))
            except SkillLoadError as e:
                errors.append(e)

        assert len(loaded) == 1
        assert loaded[0].name == "good-skill"
        assert len(errors) == 1
        assert errors[0].skill_name == "broken"


# ---------------------------------------------------------------------------
# Tests: ScriptTool invocation
# ---------------------------------------------------------------------------


class TestScriptTool:
    @pytest.mark.asyncio
    async def test_invoke_success(self, echo_script: Path) -> None:
        tool = ScriptTool(
            name="echo",
            description="Echoes args",
            script_path=echo_script,
        )
        result = await tool.invoke({"greeting": "hello", "target": "world"})

        assert result.content is not None
        assert "--greeting=hello" in result.content
        assert "--target=world" in result.content
        assert result.error is None

    @pytest.mark.asyncio
    async def test_invoke_failure_raises_script_tool_error(
        self, failing_script: Path
    ) -> None:
        tool = ScriptTool(
            name="failing-tool",
            description="Always fails",
            script_path=failing_script,
        )

        with pytest.raises(ScriptToolError) as exc_info:
            await tool.invoke({})

        assert exc_info.value.tool_name == "failing-tool"

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_script_raises_error(
        self, tmp_path: Path
    ) -> None:
        tool = ScriptTool(
            name="ghost-tool",
            description="Script does not exist",
            script_path=tmp_path / "no_such_script.py",
        )

        with pytest.raises(ScriptToolError) as exc_info:
            await tool.invoke({})

        assert exc_info.value.tool_name == "ghost-tool"

    @pytest.mark.asyncio
    async def test_invoke_empty_args(self, tmp_path: Path) -> None:
        script = tmp_path / "no_args.py"
        script.write_text('print("no args")\n', encoding="utf-8")

        tool = ScriptTool(
            name="no-args-tool",
            description="Needs no args",
            script_path=script,
        )
        result = await tool.invoke({})

        assert result.content == "no args"


# ---------------------------------------------------------------------------
# Tests: LoadedSkill
# ---------------------------------------------------------------------------


class TestLoadedSkill:
    def test_get_tools_returns_script_tools(self, echo_script: Path) -> None:
        tool = ScriptTool(name="echo", description="Echo", script_path=echo_script)
        skill = LoadedSkill(
            name="test-skill",
            description="Test",
            body="Body text",
            script_tools=["echo"],
            tools=[tool],
        )

        tools = skill.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo"

    def test_get_tools_empty_when_no_scripts(self) -> None:
        skill = LoadedSkill(
            name="no-tools",
            description="No tools",
            body="Body",
            script_tools=[],
            tools=[],
        )
        assert skill.get_tools() == []
