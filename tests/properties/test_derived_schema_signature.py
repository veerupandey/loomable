# Feature: agent-ergonomics, Property 4
"""Property 4: Derived tool schema matches the signature.

For any function with annotated parameters, the derived JSON schema SHALL contain
one property per parameter with a type mapped from its annotation
(str→string, int→integer, float→number, bool→boolean, list→array, dict→object),
and SHALL mark exactly the parameters without defaults as required.

**Validates: Requirements 2.3**
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.agent.tools import FunctionTool, tool, _JSON_TYPES, _json_type


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# The supported annotation types and their expected JSON schema type strings
ANNOTATION_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

# Strategy: choose a subset of annotations for parameters
annotation_types = st.sampled_from(list(ANNOTATION_MAP.keys()))

# Strategy: valid Python identifier names for parameter names
param_names = st.from_regex(r"[a-z][a-z0-9_]{0,14}", fullmatch=True)

# Strategy: a parameter spec — (name, annotation, has_default)
param_spec = st.tuples(param_names, annotation_types, st.booleans())

# Strategy: a list of parameter specs (1–6 params with unique names)
param_specs = (
    st.lists(param_spec, min_size=1, max_size=6)
    .map(lambda specs: _deduplicate_names(specs))
    .filter(lambda specs: len(specs) >= 1)
)


def _deduplicate_names(
    specs: list[tuple[str, type, bool]],
) -> list[tuple[str, type, bool]]:
    """Keep only the first occurrence of each parameter name."""
    seen: set[str] = set()
    result: list[tuple[str, type, bool]] = []
    for name, ann, has_default in specs:
        if name not in seen:
            seen.add(name)
            result.append((name, ann, has_default))
    return result


def _build_function_from_specs(
    specs: list[tuple[str, type, bool]],
) -> Any:
    """Dynamically build a function with the given parameter specs.

    Each spec is (name, annotation_type, has_default).
    Parameters without defaults must come before those with defaults in Python.
    """
    # Sort: required params first, then optional params
    required_params = [(n, a) for n, a, has_default in specs if not has_default]
    optional_params = [(n, a) for n, a, has_default in specs if has_default]

    # Build parameter strings
    parts: list[str] = []
    for name, ann in required_params:
        parts.append(f"{name}: {ann.__name__}")
    for name, ann in optional_params:
        # Use a simple default value appropriate for the type
        default = _default_for(ann)
        parts.append(f"{name}: {ann.__name__} = {default!r}")

    params_str = ", ".join(parts)
    func_code = (
        f"def generated_func({params_str}) -> str:\n"
        f"    '''A generated function.'''\n"
        f"    return 'ok'\n"
    )

    exec_globals: dict[str, Any] = {}
    exec(func_code, exec_globals)
    return exec_globals["generated_func"]


def _default_for(ann: type) -> Any:
    """Return a simple default value for a given annotation type."""
    defaults: dict[type, Any] = {
        str: "",
        int: 0,
        float: 0.0,
        bool: False,
        list: [],
        dict: {},
    }
    return defaults.get(ann, None)


# ---------------------------------------------------------------------------
# Property tests: Schema properties match parameters
# ---------------------------------------------------------------------------


class TestDerivedSchemaMatchesSignature:
    """Derived JSON schema matches the function signature."""

    @settings(max_examples=100)
    @given(specs=param_specs)
    def test_schema_has_one_property_per_parameter(
        self, specs: list[tuple[str, type, bool]]
    ) -> None:
        """The schema contains exactly one property per non-var parameter."""
        fn = _build_function_from_specs(specs)
        ft = FunctionTool(fn)

        schema = ft.parameters
        properties = schema["properties"]

        # There should be exactly as many properties as parameter specs
        assert len(properties) == len(specs)

        # Each parameter name should be a key in properties
        for name, _ann, _has_default in specs:
            assert name in properties, f"Parameter '{name}' missing from schema properties"

    @settings(max_examples=100)
    @given(specs=param_specs)
    def test_schema_types_match_annotations(
        self, specs: list[tuple[str, type, bool]]
    ) -> None:
        """Each property's type matches the mapped annotation type."""
        fn = _build_function_from_specs(specs)
        ft = FunctionTool(fn)

        schema = ft.parameters
        properties = schema["properties"]

        for name, ann, _has_default in specs:
            expected_type = ANNOTATION_MAP[ann]
            actual_type = properties[name]["type"]
            assert actual_type == expected_type, (
                f"Parameter '{name}' with annotation {ann.__name__}: "
                f"expected schema type '{expected_type}', got '{actual_type}'"
            )

    @settings(max_examples=100)
    @given(specs=param_specs)
    def test_required_matches_params_without_defaults(
        self, specs: list[tuple[str, type, bool]]
    ) -> None:
        """Parameters without defaults are marked as required; others are not."""
        fn = _build_function_from_specs(specs)
        ft = FunctionTool(fn)

        schema = ft.parameters
        required = schema.get("required", [])

        expected_required = {name for name, _ann, has_default in specs if not has_default}

        assert set(required) == expected_required, (
            f"Expected required={expected_required}, got required={set(required)}"
        )

    @settings(max_examples=100)
    @given(specs=param_specs)
    def test_schema_is_object_type(
        self, specs: list[tuple[str, type, bool]]
    ) -> None:
        """The top-level schema type is always 'object'."""
        fn = _build_function_from_specs(specs)
        ft = FunctionTool(fn)

        schema = ft.parameters
        assert schema["type"] == "object"

    @settings(max_examples=100)
    @given(specs=param_specs)
    def test_schema_via_tool_decorator(
        self, specs: list[tuple[str, type, bool]]
    ) -> None:
        """The @tool decorator produces the same schema as FunctionTool directly."""
        fn = _build_function_from_specs(specs)
        ft_direct = FunctionTool(fn)
        ft_decorated = tool(fn)

        assert ft_direct.parameters == ft_decorated.parameters
