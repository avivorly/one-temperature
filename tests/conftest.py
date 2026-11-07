"""Shared pytest fixtures for the test suite (synthetic data only, except the reproduction tests)."""
import tempfile

import pytest


@pytest.fixture
def tmp():
    """A fresh temporary directory, as a path string (the form the sampling tests use)."""
    with tempfile.TemporaryDirectory() as d:
        yield d
