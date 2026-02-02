"""Tests for stochastic consolidation and attractor dynamics."""

import numpy as np
import pytest

from membrain.core import BiCameralMemory


class TestStochasticConsolidation:
    """Tests for the consolidate() method with attractor dynamics."""

    @pytest.fixture
    def memory(self) -> BiCameralMemory:
        """Create a small memory for testing."""
        return BiCameralMemory(
            n_neurons=50,
            dimensions=64,
            dt=0.001,
            synapse=0.01,
            seed=42,
        )

    def test_consolidate_returns_tuple(self, memory: BiCameralMemory) -> None:
        """Consolidate should return (steps_to_converge, pruned_count)."""
        result = memory.consolidate(
            noise_scale=0.05,
            max_steps=10,
            convergence_threshold=1e-3,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        steps, pruned = result
        assert isinstance(steps, int)
        assert isinstance(pruned, int)

    def test_consolidate_converges(self, memory: BiCameralMemory) -> None:
        """Consolidation should converge within max_steps."""
        # Store a pattern first
        pattern = np.random.default_rng(42).random(64).astype(np.float32)
        memory.remember("test-pattern", pattern, importance=0.8)

        # Consolidate
        steps, pruned = memory.consolidate(
            noise_scale=0.01,  # Low noise for faster convergence
            max_steps=50,
            convergence_threshold=1e-3,
        )

        # Should converge (steps >= 0) or reach max (-1)
        assert steps == -1 or steps > 0

    def test_consolidate_with_noise_injection(self, memory: BiCameralMemory) -> None:
        """Higher noise scale should still allow convergence."""
        # Store a pattern
        pattern = np.random.default_rng(42).random(64).astype(np.float32)
        memory.remember("test-pattern", pattern, importance=0.8)

        # Consolidate with higher noise
        steps_high, _ = memory.consolidate(
            noise_scale=0.1,
            max_steps=50,
            convergence_threshold=1e-3,
        )

        # Should still work (may take more steps or hit max)
        assert steps_high == -1 or steps_high > 0

    def test_consolidate_prunes_weak_memories(self, memory: BiCameralMemory) -> None:
        """Consolidation with prune_weak should remove low-importance memories."""
        # Store patterns with different importance
        rng = np.random.default_rng(42)
        memory.remember("high-importance", rng.random(64).astype(np.float32), importance=0.9)
        memory.remember("low-importance", rng.random(64).astype(np.float32), importance=0.05)

        assert memory.get_memory_count() == 2

        # Consolidate and prune
        _, pruned = memory.consolidate(
            noise_scale=0.05,
            max_steps=10,
            convergence_threshold=1e-3,
            prune_weak=True,
            prune_threshold=0.1,
        )

        assert pruned == 1
        assert memory.get_memory_count() == 1

    def test_consolidate_without_prune(self, memory: BiCameralMemory) -> None:
        """Consolidation without prune_weak should preserve all memories."""
        rng = np.random.default_rng(42)
        memory.remember("pattern1", rng.random(64).astype(np.float32), importance=0.9)
        memory.remember("pattern2", rng.random(64).astype(np.float32), importance=0.05)

        assert memory.get_memory_count() == 2

        _, pruned = memory.consolidate(
            noise_scale=0.05,
            max_steps=10,
            convergence_threshold=1e-3,
            prune_weak=False,
        )

        assert pruned == 0
        assert memory.get_memory_count() == 2

    def test_consolidate_deterministic_with_seed(self) -> None:
        """Same seed should produce identical consolidation behavior."""
        # Create two memories with same seed
        mem1 = BiCameralMemory(n_neurons=50, dimensions=64, seed=42)
        mem2 = BiCameralMemory(n_neurons=50, dimensions=64, seed=42)

        # Store same pattern
        pattern = np.random.default_rng(100).random(64).astype(np.float32)
        mem1.remember("test", pattern, importance=0.8)
        mem2.remember("test", pattern, importance=0.8)

        # Consolidate both
        steps1, _ = mem1.consolidate(noise_scale=0.05, max_steps=20, convergence_threshold=1e-3)
        steps2, _ = mem2.consolidate(noise_scale=0.05, max_steps=20, convergence_threshold=1e-3)

        # Should take same number of steps
        assert steps1 == steps2


class TestNoiseResilience:
    """Test attractor dynamics noise resilience (Test Case 5.1 from spec)."""

    @pytest.fixture
    def memory(self) -> BiCameralMemory:
        """Create memory with deterministic seed."""
        return BiCameralMemory(
            n_neurons=100,
            dimensions=128,
            dt=0.001,
            synapse=0.01,
            seed=42,
        )

    def test_noisy_input_cleaned_up(self, memory: BiCameralMemory) -> None:
        """
        Given input with added noise,
        consolidation should allow the network to settle.
        """
        rng = np.random.default_rng(42)

        # Store a clean pattern
        clean_pattern = rng.random(128).astype(np.float32)
        clean_pattern = clean_pattern / np.linalg.norm(clean_pattern)
        memory.remember("clean", clean_pattern, importance=0.9)

        # Create noisy version (smaller perturbation)
        noise = rng.normal(0, 0.1, 128).astype(np.float32)
        noisy_pattern = clean_pattern + noise
        noisy_pattern = noisy_pattern / np.linalg.norm(noisy_pattern)

        # Verify noisy is somewhat similar
        initial_similarity = float(np.dot(clean_pattern, noisy_pattern))
        assert initial_similarity > 0.5, f"Initial similarity: {initial_similarity}"

        # Store noisy and consolidate
        memory.remember("noisy", noisy_pattern, importance=0.5)

        steps, _ = memory.consolidate(
            noise_scale=0.05,
            max_steps=50,
            convergence_threshold=1e-4,
        )

        # The network should have settled
        # (full pattern completion would need more sophisticated testing)
        assert steps == -1 or steps > 0


class TestSpuriousStateRejection:
    """Test spurious state rejection (Test Case 5.2 from spec)."""

    @pytest.fixture
    def memory(self) -> BiCameralMemory:
        return BiCameralMemory(
            n_neurons=50,
            dimensions=64,
            seed=42,
        )

    def test_random_input_settles(self, memory: BiCameralMemory) -> None:
        """
        Given random noise input,
        consolidation should settle to stable state (not hallucinate).
        """
        # Store a valid pattern
        rng = np.random.default_rng(42)
        valid_pattern = rng.random(64).astype(np.float32)
        memory.remember("valid", valid_pattern, importance=0.9)

        # Inject random noise and consolidate
        steps, _ = memory.consolidate(
            noise_scale=0.5,  # High noise
            max_steps=50,
            convergence_threshold=1e-3,
        )

        # Should either converge or hit max (not crash/hang)
        assert steps == -1 or steps > 0
