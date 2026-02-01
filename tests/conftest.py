"""Pytest configuration and fixtures for Membrain tests.

Provides fast test configurations with small dimensions for CI.
"""

import pytest

from membrain.config import MembrainConfig


@pytest.fixture
def small_config() -> MembrainConfig:
    """Fast config for CI with small dimensions.

    Uses:
    - input_dim=64 (vs 1536 default)
    - n_neurons=50 (vs 1000 default)
    - Fixed seed for reproducibility
    """
    return MembrainConfig.for_testing()


@pytest.fixture
def test_config() -> MembrainConfig:
    """Alias for small_config for clarity."""
    return MembrainConfig.for_testing()


@pytest.fixture
def small_input_dim() -> int:
    """Small input dimension for fast tests."""
    return 64


@pytest.fixture
def small_n_neurons() -> int:
    """Small neuron count for fast tests."""
    return 50


@pytest.fixture
def test_seed() -> int:
    """Fixed seed for reproducible tests."""
    return 42
