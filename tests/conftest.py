"""Shared pytest configuration and fixtures for loomable tests."""

from __future__ import annotations

import pytest


def pytest_addoption(parser):
    parser.addoption("--live", action="store_true", default=False,
                     help="Run explicitly marked live provider tests (may incur API charges).")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="live provider tests require --live")
    for item in items:
        if "live" in item.keywords or item.name.startswith("test_live_") or item.path.stem.endswith("_live"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolate_default_zvec(tmp_path, monkeypatch):
    """Do not share ``.loomable/memory_zvec`` across tests (zvec exclusive lock)."""
    path = tmp_path / "memory_zvec"
    monkeypatch.setattr("loomable.kernel.long_term.DEFAULT_ZVEC_PATH", path)
    monkeypatch.setattr("loomable.providers.vector_store.DEFAULT_ZVEC_PATH", path)
