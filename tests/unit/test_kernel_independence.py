"""Unit test asserting the kernel remains independent of the additive layers.

# Feature: agent-api, Property 16
# Feature: agent-ergonomics, Property 17

Property 16: Kernel remains independent — the ``loomable.kernel`` package tree
must import nothing from the additive convenience layers ``loomable.agent``,
``loomable.content``, or ``loomable.serve``.

Property 17: Kernel remains independent — assert ``loomable.kernel`` imports
nothing from ``loomable.agent``/``content``/``serve``/``providers``.

Validates: Req 1.7, 2.4, 7.7, 8.6, 10.3
Validates: Requirements 9.2, 9.3
"""

import ast
import sys
from pathlib import Path

import pytest

import loomable.kernel

# The additive layers the kernel must never depend on.
FORBIDDEN_PREFIXES = (
    "loomable.agent",
    "loomable.content",
    "loomable.serve",
    "loomable.providers",
)

KERNEL_DIR = Path(loomable.kernel.__file__).parent
KERNEL_FILES = sorted(KERNEL_DIR.rglob("*.py"))


def _imported_module_names(tree: ast.AST):
    """Yield the fully-qualified module names referenced by import statements."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            # Absolute imports only carry meaningful module prefixes here.
            # ``from X import Y`` -> module is X; relative imports (level > 0)
            # stay inside loomable.kernel and are therefore always safe.
            if node.level == 0 and node.module is not None:
                yield node.module


def _is_forbidden(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in FORBIDDEN_PREFIXES
    )


def test_kernel_directory_has_python_files():
    """Sanity check: the kernel tree exists and contains modules to scan."""
    assert KERNEL_FILES, f"No .py files found under {KERNEL_DIR}"


@pytest.mark.parametrize("path", KERNEL_FILES, ids=lambda p: str(p.name))
def test_kernel_file_does_not_import_additive_layers(path: Path):
    """Each kernel source file must not import loomable.agent/content/serve."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    forbidden = sorted(
        {name for name in _imported_module_names(tree) if _is_forbidden(name)}
    )

    assert not forbidden, (
        f"{path.relative_to(KERNEL_DIR.parent)} imports forbidden additive "
        f"layer(s): {forbidden}. The kernel must remain independent of "
        f"loomable.agent/content/serve/providers."
    )


def test_kernel_tree_imports_no_additive_layer_aggregate():
    """Aggregate scan across the whole kernel tree for a single clear report."""
    violations: dict[str, list[str]] = {}
    for path in KERNEL_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        forbidden = sorted(
            {name for name in _imported_module_names(tree) if _is_forbidden(name)}
        )
        if forbidden:
            violations[str(path.relative_to(KERNEL_DIR.parent))] = forbidden

    assert not violations, (
        "Kernel files import additive layers (loomable.agent/content/serve/providers): "
        f"{violations}"
    )


def test_importing_kernel_does_not_pull_in_additive_layers():
    """Runtime check: kernel modules in sys.modules do not reference the layers.

    Other tests in the suite may have already imported the additive layers, so
    we do not assert that the layers are absent from ``sys.modules`` globally.
    Instead we assert that no module whose name starts with ``loomable.kernel``
    has, among its own module-level imports, a reference to a forbidden layer.
    The authoritative guarantee is provided by the AST/source scans above; this
    check guards against dynamic/aliased imports that a static scan of the
    kernel's own files would still have caught, but confirms it at runtime.
    """
    kernel_modules = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "loomable.kernel" or name.startswith("loomable.kernel.")
    }
    assert kernel_modules, "loomable.kernel was not imported"

    offenders: dict[str, list[str]] = {}
    for name, module in kernel_modules.items():
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        path = Path(module_file)
        if not path.exists() or path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden = sorted(
            {n for n in _imported_module_names(tree) if _is_forbidden(n)}
        )
        if forbidden:
            offenders[name] = forbidden

    assert not offenders, (
        f"Imported kernel modules reference additive layers: {offenders}"
    )
