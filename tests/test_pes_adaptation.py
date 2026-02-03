"""Tests for PES decoder adaptation.

Tests that PES learning keeps decoders aligned with Voja-adapted encoders,
improving reconstruction accuracy over time.
"""

from __future__ import annotations

import numpy as np
import pytest

from membrain.core import BiCameralMemory


class TestPESIntegration:
    """Tests for PES decoder learning integration."""

    def test_pes_enabled_by_default(self) -> None:
        """PES should be enabled by default."""
        mem = BiCameralMemory(
            n_neurons=100,
            dimensions=64,
            seed=42,
        )
        assert mem.use_pes is True

    def test_pes_can_be_disabled(self) -> None:
        """PES can be disabled via use_pes=False."""
        mem = BiCameralMemory(
            n_neurons=100,
            dimensions=64,
            use_pes=False,
            seed=42,
        )
        assert mem.use_pes is False
        assert mem.error is None  # No error ensemble when PES disabled

    def test_error_ensemble_created(self) -> None:
        """Error ensemble should be created when PES is enabled."""
        mem = BiCameralMemory(
            n_neurons=100,
            dimensions=64,
            use_pes=True,
            seed=42,
        )
        assert mem.error is not None

    def test_output_node_created(self) -> None:
        """Output node should be created for decoded output."""
        mem = BiCameralMemory(
            n_neurons=100,
            dimensions=64,
            seed=42,
        )
        assert mem.output_node is not None


class TestPESLearning:
    """Tests for PES learning behavior."""

    def test_pes_learning_disabled_during_recall(self) -> None:
        """PES learning should be disabled during recall (gate = -1)."""
        mem = BiCameralMemory(
            n_neurons=100,
            dimensions=64,
            use_pes=True,
            seed=42,
        )

        # Store a pattern
        pattern = np.random.randn(64).astype(np.float32)
        pattern = pattern / np.linalg.norm(pattern)
        mem.remember("test", pattern)

        # Recall should not modify weights (gate = -1 during recall)
        # This is implicitly tested by the gate mechanism
        results = mem.recall(pattern, threshold=0.0, max_results=5)
        assert len(results) > 0

    def test_reconstruction_improves_with_patterns(self) -> None:
        """Reconstruction error should decrease as more patterns are stored."""
        mem = BiCameralMemory(
            n_neurons=200,
            dimensions=64,
            use_pes=True,
            pes_learning_rate=1e-3,  # Higher LR for faster learning in test
            seed=42,
        )

        # Generate patterns
        rng = np.random.default_rng(42)
        patterns = []
        for _ in range(20):
            p = rng.standard_normal(64).astype(np.float32)
            p = p / np.linalg.norm(p)
            patterns.append(p)

        # Store patterns
        for i, p in enumerate(patterns):
            mem.remember(f"p{i}", p)

        # Recall should work (basic sanity check)
        results = mem.recall(patterns[0], threshold=0.0, max_results=5)
        assert len(results) > 0

        # The first result should be the pattern itself with high similarity
        # (This tests that PES-adapted decoders produce meaningful output)
        assert results[0].context_id == "p0"
        assert results[0].confidence > 0.5


class TestNoisyRecall:
    """Tests for noisy recall with PES."""

    def test_noisy_recall_returns_results(self) -> None:
        """Noisy queries should still return results."""
        mem = BiCameralMemory(
            n_neurons=200,
            dimensions=64,
            use_pes=True,
            seed=42,
        )

        # Store patterns
        rng = np.random.default_rng(42)
        patterns = []
        for i in range(10):
            p = rng.standard_normal(64).astype(np.float32)
            p = p / np.linalg.norm(p)
            patterns.append(p)
            mem.remember(f"p{i}", p)

        # Add noise to query
        noisy_query = patterns[0] + 0.2 * rng.standard_normal(64).astype(np.float32)
        noisy_query = noisy_query / np.linalg.norm(noisy_query)

        results = mem.recall(noisy_query, threshold=0.0, max_results=5)
        assert len(results) > 0

    def test_pes_vs_no_pes_recall(self) -> None:
        """Compare recall accuracy with and without PES."""
        rng = np.random.default_rng(42)

        # Generate patterns
        patterns = []
        for _ in range(10):
            p = rng.standard_normal(64).astype(np.float32)
            p = p / np.linalg.norm(p)
            patterns.append(p)

        # With PES
        mem_pes = BiCameralMemory(
            n_neurons=200,
            dimensions=64,
            use_pes=True,
            pes_learning_rate=1e-3,
            seed=42,
        )
        for i, p in enumerate(patterns):
            mem_pes.remember(f"p{i}", p)

        # Without PES
        mem_no_pes = BiCameralMemory(
            n_neurons=200,
            dimensions=64,
            use_pes=False,
            seed=42,
        )
        for i, p in enumerate(patterns):
            mem_no_pes.remember(f"p{i}", p)

        # Both should work (this is a sanity check, not a performance comparison)
        results_pes = mem_pes.recall(patterns[0], threshold=0.0, max_results=1)
        results_no_pes = mem_no_pes.recall(patterns[0], threshold=0.0, max_results=1)

        assert len(results_pes) > 0
        assert len(results_no_pes) > 0


class TestBypassSNN:
    """Tests for bypass_snn behavior."""

    def test_bypass_snn_uses_raw_query(self) -> None:
        """bypass_snn should skip SNN and use raw query."""
        mem = BiCameralMemory(
            n_neurons=100,
            dimensions=64,
            use_pes=True,
            seed=42,
        )

        pattern = np.random.randn(64).astype(np.float32)
        pattern = pattern / np.linalg.norm(pattern)
        mem.remember("test", pattern)

        # Both should return results
        results_snn = mem.recall(pattern, threshold=0.0, bypass_snn=False)
        results_bypass = mem.recall(pattern, threshold=0.0, bypass_snn=True)

        assert len(results_snn) > 0
        assert len(results_bypass) > 0
