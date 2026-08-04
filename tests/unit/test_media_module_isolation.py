"""Unit test verifying loomable.media module isolation from loomable.kernel.

The ``loomable.media`` package must depend only on ``loomable.content`` (for
``MediaPart``, ``Modality``) and the Python standard library. It must NEVER
import from ``loomable.kernel``.

Validates: Requirements 8.2, 9.1, 9.2, 9.3, 9.4
"""

import ast
from pathlib import Path

import pytest

import loomable.media

# Forbidden: loomable.media must not import anything from loomable.kernel.
FORBIDDEN_PREFIXES = ("loomable.kernel",)

# Allowed external (non-stdlib) imports for loomable.media
ALLOWED_PREFIXES = ("loomable.content",)

MEDIA_DIR = Path(loomable.media.__file__).parent
MEDIA_FILES = sorted(MEDIA_DIR.rglob("*.py"))


def _imported_module_names(tree: ast.AST):
    """Yield fully-qualified module names referenced by import statements."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            # Only absolute imports carry meaningful module prefixes.
            # Relative imports (level > 0) stay within loomable.media itself.
            if node.level == 0 and node.module is not None:
                yield node.module


def _is_forbidden(module_name: str) -> bool:
    """Return True if module_name is in a forbidden package."""
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in FORBIDDEN_PREFIXES
    )


def test_media_directory_has_python_files():
    """Sanity check: the media package exists and has modules to scan."""
    assert MEDIA_FILES, f"No .py files found under {MEDIA_DIR}"


@pytest.mark.parametrize("path", MEDIA_FILES, ids=lambda p: str(p.name))
def test_media_file_does_not_import_kernel(path: Path):
    """Each loomable.media source file must not import from loomable.kernel."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    forbidden = sorted(
        {name for name in _imported_module_names(tree) if _is_forbidden(name)}
    )

    assert not forbidden, (
        f"{path.relative_to(MEDIA_DIR.parent)} imports forbidden package(s): "
        f"{forbidden}. loomable.media must NOT depend on loomable.kernel."
    )


def test_media_types_only_imports_content_and_stdlib():
    """types.py should only import from loomable.content and standard library."""
    types_file = MEDIA_DIR / "types.py"
    assert types_file.exists(), "types.py not found in loomable.media"

    source = types_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(types_file))

    external_imports = []
    for name in _imported_module_names(tree):
        # Skip stdlib imports (they don't start with "loomable")
        if not name.startswith("loomable"):
            continue
        # Skip allowed imports (loomable.content)
        is_allowed = any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in ALLOWED_PREFIXES
        )
        if not is_allowed:
            external_imports.append(name)

    assert not external_imports, (
        f"loomable/media/types.py imports non-allowed loomable packages: "
        f"{external_imports}. Only loomable.content and stdlib are permitted."
    )


def test_media_aggregate_no_kernel_imports():
    """Aggregate scan across all loomable.media files for a single report."""
    violations: dict[str, list[str]] = {}
    for path in MEDIA_FILES:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        forbidden = sorted(
            {name for name in _imported_module_names(tree) if _is_forbidden(name)}
        )
        if forbidden:
            violations[str(path.relative_to(MEDIA_DIR.parent))] = forbidden

    assert not violations, (
        "loomable.media files import loomable.kernel (forbidden): "
        f"{violations}"
    )
