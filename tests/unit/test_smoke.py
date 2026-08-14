"""Smoke test to verify the test suite runs correctly."""


def test_import_loomable():
    """Verify the loomable package is importable."""
    import loomable

    assert loomable.__version__ == "0.2.0b0"


def test_import_kernel():
    """Verify the loomable.kernel package is importable."""
    import loomable.kernel

    assert loomable.kernel is not None
