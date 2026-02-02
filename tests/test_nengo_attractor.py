"""Tests for NengoAttractorMemory.

Tests the Nengo-based spiking attractor network.
"""

from __future__ import annotations

import numpy as np
import pytest

from membrain.nengo_attractor import NengoAttractorMemory, NengoCleanupResult


class TestNengoAttractorInit:
    """Tests for initialization."""

    def test_valid_init(self) -> None:
        """Should initialize with valid parameters."""
        attractor = NengoAttractorMemory(dimensions=64, n_neurons=100)
        assert attractor.dimensions == 64
        assert attractor.n_neurons == 100
        assert attractor.pattern_count == 0

    def test_invalid_dimensions(self) -> None:
        """Should reject non-positive dimensions."""
        with pytest.raises(ValueError, match="dimensions must be positive"):
            NengoAttractorMemory(dimensions=0)

    def test_invalid_neurons(self) -> None:
        """Should reject non-positive n_neurons."""
        with pytest.raises(ValueError, match="n_neurons must be positive"):
            NengoAttractorMemory(dimensions=64, n_neurons=0)


class TestNengoAttractorStore:
    """Tests for pattern storage."""

    def test_store_single_pattern(self) -> None:
        """Should store a pattern."""
        attractor = NengoAttractorMemory(dimensions=64, n_neurons=100, seed=42)
        pattern = np.random.default_rng(42).standard_normal(64).astype(np.float32)
        pattern /= np.linalg.norm(pattern)

        attractor.store(pattern)
        assert attractor.pattern_count == 1

    def test_store_wrong_shape(self) -> None:
        """Should reject patterns with wrong shape."""
        attractor = NengoAttractorMemory(dimensions=64, n_neurons=100)
        wrong_pattern = np.zeros(128, dtype=np.float32)

        with pytest.raises(ValueError, match="Expected shape"):
            attractor.store(wrong_pattern)

    def test_store_near_zero(self) -> None:
        """Should skip near-zero patterns."""
        attractor = NengoAttractorMemory(dimensions=64, n_neurons=100)
        zero_pattern = np.zeros(64, dtype=np.float32)

        attractor.store(zero_pattern)
        assert attractor.pattern_count == 0


class TestNengoAttractorComplete:
    """Tests for pattern completion."""

    def test_complete_returns_result(self) -> None:
        """Should return NengoCleanupResult."""
        attractor = NengoAttractorMemory(
            dimensions=64,
            n_neurons=100,
            settling_time_ms=50,
            seed=42,
        )
        rng = np.random.default_rng(42)

        # Store a pattern
        pattern = rng.standard_normal(64).astype(np.float32)
        pattern /= np.linalg.norm(pattern)
        attractor.store(pattern)

        # Complete noisy version
        noisy = pattern + 0.2 * rng.standard_normal(64).astype(np.float32)
        result = attractor.complete(noisy)

        assert isinstance(result, NengoCleanupResult)
        assert result.cleaned.shape == (64,)
        assert result.settling_time_ms == 50

    def test_complete_wrong_shape(self) -> None:
        """Should reject inputs with wrong shape."""
        attractor = NengoAttractorMemory(dimensions=64, n_neurons=100)
        wrong_input = np.zeros(128, dtype=np.float32)

        with pytest.raises(ValueError, match="Expected shape"):
            attractor.complete(wrong_input)

    def test_complete_custom_settling_time(self) -> None:
        """Should accept custom settling time."""
        attractor = NengoAttractorMemory(
            dimensions=64,
            n_neurons=100,
            settling_time_ms=50,
            seed=42,
        )
        rng = np.random.default_rng(42)

        pattern = rng.standard_normal(64).astype(np.float32)
        pattern /= np.linalg.norm(pattern)
        attractor.store(pattern)

        result = attractor.complete(pattern, settling_time_ms=100)
        assert result.settling_time_ms == 100


class TestNengoAttractorReset:
    """Tests for reset functionality."""

    def test_reset_clears_state(self) -> None:
        """Should reset pattern count and simulator."""
        attractor = NengoAttractorMemory(dimensions=64, n_neurons=100, seed=42)
        rng = np.random.default_rng(42)

        pattern = rng.standard_normal(64).astype(np.float32)
        pattern /= np.linalg.norm(pattern)
        attractor.store(pattern)

        assert attractor.pattern_count == 1

        attractor.reset()
        assert attractor.pattern_count == 0


class TestNengoAttractorCleanup:
    """Tests for cleanup effectiveness."""

    @pytest.mark.slow
    def test_cleanup_output_normalized(self) -> None:
        """Output should be normalized."""
        attractor = NengoAttractorMemory(
            dimensions=64,
            n_neurons=200,
            settling_time_ms=50,
            seed=42,
        )
        rng = np.random.default_rng(42)

        pattern = rng.standard_normal(64).astype(np.float32)
        pattern /= np.linalg.norm(pattern)
        attractor.store(pattern)

        noisy = pattern + 0.3 * rng.standard_normal(64).astype(np.float32)
        result = attractor.complete(noisy)

        # Output should be approximately normalized
        output_norm = np.linalg.norm(result.cleaned)
        assert 0.5 < output_norm < 1.5  # Allow some variance from spiking noise
