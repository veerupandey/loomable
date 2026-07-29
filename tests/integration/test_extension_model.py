"""Integration tests for the Extension Model (Requirement 19).

Validates:
- A representative Domain_Skill can be discovered and loaded by SkillLoader.
- The Kernel package (loomable.kernel) imports NO example module.
- The Domain_Skill is enabled purely through configuration/wiring, not Kernel modification.
- An unsupported capability request is properly rejected.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

from loomable.kernel.errors import UnsupportedExtensionError
from loomable.kernel.models import ExtensionMechanism, OnboardingRequest
from loomable.kernel.registry import ExtensionRegistry
from loomable.kernel.skills import LoadedSkill, SkillLoader


# Path to the example skills directory
EXAMPLES_SKILLS_DIR = Path(__file__).resolve().parents[2] / "examples" / "skills"


class TestDomainSkillDiscoveryAndLoad:
    """Validates Requirements 19.1 and 19.2: example Domain_Skill is enabled via config."""

    def test_weather_skill_discovered(self) -> None:
        """The weather-lookup skill is discovered by SkillLoader from the examples dir."""
        loader = SkillLoader()
        manifests = loader.discover([EXAMPLES_SKILLS_DIR])

        names = [m.name for m in manifests]
        assert "weather-lookup" in names

    def test_weather_skill_has_script_tool(self) -> None:
        """The discovered manifest includes the get_weather script tool."""
        loader = SkillLoader()
        manifests = loader.discover([EXAMPLES_SKILLS_DIR])

        weather_manifest = next(m for m in manifests if m.name == "weather-lookup")
        tool_names = [s.name for s in weather_manifest.script_tools]
        assert "get_weather" in tool_names

    def test_weather_skill_loads_successfully(self) -> None:
        """The weather-lookup skill loads into a full LoadedSkill instance."""
        loader = SkillLoader()
        manifests = loader.discover([EXAMPLES_SKILLS_DIR])
        weather_manifest = next(m for m in manifests if m.name == "weather-lookup")

        skill = loader.load(weather_manifest)

        assert isinstance(skill, LoadedSkill)
        assert skill.name == "weather-lookup"
        assert "get_weather" in skill.script_tools
        assert len(skill.get_tools()) == 1
        assert skill.get_tools()[0].name == "get_weather"

    def test_weather_skill_body_contains_instructions(self) -> None:
        """The loaded skill body contains the markdown instruction content."""
        loader = SkillLoader()
        manifests = loader.discover([EXAMPLES_SKILLS_DIR])
        weather_manifest = next(m for m in manifests if m.name == "weather-lookup")

        skill = loader.load(weather_manifest)

        assert "Weather Lookup Skill" in skill.body
        assert "get_weather" in skill.body


class TestKernelDoesNotImportExamples:
    """Validates Requirement 19.2: Kernel has no dependency on example modules."""

    def test_no_example_module_in_kernel_package(self) -> None:
        """The loomable.kernel package does not import any 'examples' module."""
        import loomable.kernel

        # Check all submodules of loomable.kernel
        kernel_path = Path(loomable.kernel.__file__).parent
        submodule_names = [
            name
            for _, name, _ in pkgutil.iter_modules([str(kernel_path)])
        ]

        # No submodule should reference "example" or "weather"
        for name in submodule_names:
            assert "example" not in name.lower(), (
                f"Kernel submodule '{name}' references examples"
            )
            assert "weather" not in name.lower(), (
                f"Kernel submodule '{name}' references domain skill"
            )

    def test_kernel_init_does_not_import_examples(self) -> None:
        """The loomable.kernel __init__.py exports do not reference examples."""
        import loomable.kernel

        all_exports = getattr(loomable.kernel, "__all__", [])
        for export in all_exports:
            assert "example" not in export.lower(), (
                f"Kernel export '{export}' references examples"
            )
            assert "weather" not in export.lower(), (
                f"Kernel export '{export}' references domain skill"
            )

    def test_no_examples_in_sys_modules_for_kernel(self) -> None:
        """No module under loomable.kernel references 'examples' in sys.modules."""
        # Ensure kernel is imported
        import loomable.kernel  # noqa: F401

        kernel_modules = [
            mod_name
            for mod_name in sys.modules
            if mod_name.startswith("loomable.kernel")
        ]

        for mod_name in kernel_modules:
            assert "example" not in mod_name.lower(), (
                f"Module '{mod_name}' under loomable.kernel references examples"
            )


class TestExtensionOnlyWiring:
    """Validates that Domain_Skill is enabled via configuration, not Kernel code."""

    def test_skill_onboarded_via_extension_registry(self) -> None:
        """A domain skill can be onboarded through the ExtensionRegistry as a SKILL."""
        registry = ExtensionRegistry()
        request = OnboardingRequest(
            capability="weather-lookup",
            mechanism=ExtensionMechanism.SKILL,
        )

        handle = registry.onboard(request)

        assert handle.capability == "weather-lookup"
        assert handle.mechanism == ExtensionMechanism.SKILL

    def test_skill_tool_registered_and_resolved_lazily(self) -> None:
        """A skill's tool can be registered and lazily resolved via the registry."""
        loader = SkillLoader()
        manifests = loader.discover([EXAMPLES_SKILLS_DIR])
        weather_manifest = next(m for m in manifests if m.name == "weather-lookup")

        registry = ExtensionRegistry()
        request = OnboardingRequest(
            capability="weather-lookup",
            mechanism=ExtensionMechanism.SKILL,
        )
        handle = registry.onboard(request)

        # Register the tool factory (lazy loading)
        def tool_factory():
            skill = loader.load(weather_manifest)
            return skill.get_tools()[0]

        registry.register_tool_for_extension(handle.id, "get_weather", tool_factory)

        # Resolve lazily
        tool = registry.resolve_tool("get_weather")
        assert tool.name == "get_weather"


class TestUnsupportedCapabilityRejection:
    """Validates Requirement 19.3: unsupported capabilities are rejected."""

    def test_kernel_modification_rejected(self) -> None:
        """Onboarding via KERNEL_MODIFICATION raises UnsupportedExtensionError."""
        registry = ExtensionRegistry()
        request = OnboardingRequest(
            capability="some-domain-capability",
            mechanism=ExtensionMechanism.KERNEL_MODIFICATION,
        )

        with pytest.raises(UnsupportedExtensionError) as exc_info:
            registry.onboard(request)

        # The error carries the list of supported mechanisms
        assert "skill" in exc_info.value.supported_mechanisms
        assert "mcp_server" in exc_info.value.supported_mechanisms
        assert "api_tool" in exc_info.value.supported_mechanisms

    def test_unsupported_error_message_informative(self) -> None:
        """The UnsupportedExtensionError message names the supported mechanisms."""
        registry = ExtensionRegistry()
        request = OnboardingRequest(
            capability="weather-lookup",
            mechanism=ExtensionMechanism.KERNEL_MODIFICATION,
        )

        with pytest.raises(UnsupportedExtensionError) as exc_info:
            registry.onboard(request)

        error_msg = str(exc_info.value)
        assert "Unsupported extension mechanism" in error_msg
        assert "skill" in error_msg
        assert "mcp_server" in error_msg
