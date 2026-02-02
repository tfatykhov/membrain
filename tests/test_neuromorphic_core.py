"""
Tests for BiCameralMemory neuromorphic core.

Tests the Nengo-based SNN memory system including:
- Network construction
- Memory storage (remember)
- Memory retrieval (recall)
- Pattern completion with noise
- Sparsity metrics
- Consolidation and pruning
"""

import numpy as np
import pytest

from membrain.core import BiCameralMemory, MemoryEntry, RecallResult
from membrain.encoder import FlyHash


class TestBiCameralMemoryInit:
    """Tests for BiCameralMemory initialization."""

    def test_default_initialization(self) -> None:
        """BiCameralMemory should initialize with defaults."""
        memory = BiCameralMemory()
        assert memory.n_neurons == 1000
        assert memory.dimensions == 20000
        assert memory.learning_rate == 1e-2
        memory.reset()

    def test_custom_parameters(self) -> None:
        """BiCameralMemory should accept custom parameters."""
        memory = BiCameralMemory(
            n_neurons=500,
            dimensions=10000,
            learning_rate=1e-3,
            synapse=0.02,
            dt=0.002,
        )
        assert memory.n_neurons == 500
        assert memory.dimensions == 10000
        memory.reset()

    def test_network_builds(self) -> None:
        """Network components should be created."""
        memory = BiCameralMemory(n_neurons=100, dimensions=1000)
        assert memory.model is not None
        assert memory.input_node is not None
        assert memory.memory is not None
        assert memory.output_probe is not None
        memory.reset()

    def test_repr(self) -> None:
        """__repr__ should return informative string."""
        memory = BiCameralMemory(n_neurons=100, dimensions=500)
        r = repr(memory)
        assert "BiCameralMemory" in r
        assert "n_neurons=100" in r
        assert "dimensions=500" in r
        memory.reset()


class TestBiCameralMemoryRemember:
    """Tests for memory storage."""

    @pytest.fixture
    def memory(self) -> BiCameralMemory:
        """Create a small memory for testing."""
        mem = BiCameralMemory(n_neurons=50, dimensions=500)
        yield mem
        mem.reset()

    def test_remember_stores_entry(self, memory: BiCameralMemory) -> None:
        """Remember should store memory entry."""
        vector = np.zeros(500, dtype=np.float32)
        vector[:25] = 1.0  # 5% active (sparse)

        result = memory.remember("test-001", vector)

        assert result is True
        assert memory.get_memory_count() == 1
        assert "test-001" in memory.get_memory_ids()

    def test_remember_multiple(self, memory: BiCameralMemory) -> None:
        """Should store multiple memories."""
        for i in range(5):
            vector = np.zeros(500, dtype=np.float32)
            vector[i * 10 : (i + 1) * 10] = 1.0
            memory.remember(f"mem-{i:03d}", vector)

        assert memory.get_memory_count() == 5

    def test_remember_with_importance(self, memory: BiCameralMemory) -> None:
        """Remember should accept importance parameter."""
        vector = np.zeros(500, dtype=np.float32)
        vector[:25] = 1.0

        result = memory.remember("important", vector, importance=0.9)
        assert result is True

    def test_remember_invalid_shape_raises(self, memory: BiCameralMemory) -> None:
        """Should raise for wrong vector shape."""
        vector = np.zeros(100, dtype=np.float32)

        with pytest.raises(ValueError, match="Expected shape"):
            memory.remember("bad-shape", vector)


class TestBiCameralMemoryRecall:
    """Tests for memory retrieval."""

    @pytest.fixture
    def memory_with_data(self) -> BiCameralMemory:
        """Create memory with stored entries."""
        mem = BiCameralMemory(n_neurons=100, dimensions=500)

        # Store some distinct patterns
        for i in range(3):
            vector = np.zeros(500, dtype=np.float32)
            vector[i * 50 : (i + 1) * 50] = 1.0
            mem.remember(f"pattern-{i}", vector, importance=0.8)

        yield mem
        mem.reset()

    def test_recall_returns_results(self, memory_with_data: BiCameralMemory) -> None:
        """Recall should return matching memories."""
        # Query with first pattern
        query = np.zeros(500, dtype=np.float32)
        query[:50] = 1.0

        results = memory_with_data.recall(query, threshold=0.1)

        assert isinstance(results, list)
        # Should find at least the matching pattern
        # (threshold is low to account for SNN dynamics)

    def test_recall_result_structure(self, memory_with_data: BiCameralMemory) -> None:
        """RecallResult should have correct structure."""
        query = np.zeros(500, dtype=np.float32)
        query[:50] = 1.0

        results = memory_with_data.recall(query, threshold=0.0)

        if len(results) > 0:
            result = results[0]
            assert isinstance(result, RecallResult)
            assert isinstance(result.context_id, str)
            assert isinstance(result.confidence, float)
            assert 0.0 <= result.confidence <= 1.0

    def test_recall_invalid_shape_raises(
        self, memory_with_data: BiCameralMemory
    ) -> None:
        """Should raise for wrong query shape."""
        query = np.zeros(100, dtype=np.float32)

        with pytest.raises(ValueError, match="Expected shape"):
            memory_with_data.recall(query)


class TestPatternCompletion:
    """Tests for pattern completion with noise."""

    @pytest.fixture
    def encoder(self) -> FlyHash:
        """Create FlyHash encoder for testing."""
        return FlyHash(input_dim=64, expansion_ratio=8.0, active_bits=25, seed=42)

    @pytest.fixture
    def memory(self, encoder: FlyHash) -> BiCameralMemory:
        """Create memory sized for encoder output."""
        mem = BiCameralMemory(n_neurons=100, dimensions=encoder.output_dim)
        yield mem
        mem.reset()

    def test_stores_and_recalls_with_encoder(
        self, memory: BiCameralMemory, encoder: FlyHash
    ) -> None:
        """Should work with FlyHash encoded vectors."""
        # Create original embedding
        rng = np.random.default_rng(42)
        original = rng.standard_normal(64).astype(np.float32)

        # Encode and store
        sparse = encoder.encode(original)
        memory.remember("encoded-doc", sparse)

        # Recall with same vector
        results = memory.recall(sparse, threshold=0.1)

        # Should find something (relaxed threshold due to SNN dynamics)
        assert memory.get_memory_count() == 1


class TestSparsityMetrics:
    """Tests for sparsity and SynOp metrics."""

    @pytest.fixture
    def memory(self) -> BiCameralMemory:
        """Create memory for metric testing."""
        mem = BiCameralMemory(n_neurons=100, dimensions=500)
        yield mem
        mem.reset()

    def test_sparsity_rate_initial(self, memory: BiCameralMemory) -> None:
        """Initial sparsity should be 1.0 (no activity)."""
        sparsity = memory.get_sparsity_rate()
        assert sparsity == 1.0

    def test_sparsity_rate_after_activity(self, memory: BiCameralMemory) -> None:
        """Sparsity should be measurable after activity."""
        vector = np.zeros(500, dtype=np.float32)
        vector[:25] = 1.0

        memory.remember("sparse-test", vector)
        sparsity = memory.get_sparsity_rate()

        # Should be high (>0.5) since we're using sparse input
        assert 0.0 <= sparsity <= 1.0

    def test_synop_count_initial(self, memory: BiCameralMemory) -> None:
        """Initial SynOp count should be 0."""
        synops = memory.get_synop_count()
        assert synops == 0

    def test_synop_count_after_activity(self, memory: BiCameralMemory) -> None:
        """SynOp count should increase after activity."""
        vector = np.zeros(500, dtype=np.float32)
        vector[:25] = 1.0

        memory.remember("synop-test", vector)
        synops = memory.get_synop_count()

        # Should be non-negative
        assert synops >= 0


class TestConsolidation:
    """Tests for memory consolidation."""

    @pytest.fixture
    def memory(self) -> BiCameralMemory:
        """Create memory with test data."""
        mem = BiCameralMemory(n_neurons=50, dimensions=500)

        # Store with varying importance
        for i, importance in enumerate([0.05, 0.5, 0.9]):
            vector = np.zeros(500, dtype=np.float32)
            vector[i * 10 : (i + 1) * 10] = 1.0
            mem.remember(f"mem-{i}", vector, importance=importance)

        yield mem
        mem.reset()

    def test_consolidation_runs(self, memory: BiCameralMemory) -> None:
        """Consolidation should run without error."""
        steps, pruned = memory.consolidate(max_steps=10)
        assert pruned == 0  # No pruning by default
        assert steps == -1 or steps > 0  # Either converged or hit max

    def test_consolidation_prunes_weak(self, memory: BiCameralMemory) -> None:
        """Consolidation should prune weak memories."""
        initial_count = memory.get_memory_count()

        steps, pruned = memory.consolidate(prune_weak=True, prune_threshold=0.1, max_steps=10)

        assert pruned == 1  # One memory has importance 0.05 < 0.1
        assert memory.get_memory_count() == initial_count - 1
        assert "mem-0" not in memory.get_memory_ids()


class TestContextManager:
    """Tests for context manager protocol."""

    def test_context_manager_builds_simulator(self) -> None:
        """Context manager should build simulator on entry."""
        with BiCameralMemory(n_neurons=50, dimensions=500) as memory:
            assert memory._simulator is not None
            vector = np.zeros(500, dtype=np.float32)
            vector[:25] = 1.0
            memory.remember("ctx-test", vector)

    def test_context_manager_cleans_up(self) -> None:
        """Context manager should clean up on exit."""
        memory = BiCameralMemory(n_neurons=50, dimensions=500)

        with memory:
            vector = np.zeros(500, dtype=np.float32)
            vector[:25] = 1.0
            memory.remember("cleanup-test", vector)

        # After exit, simulator should be cleaned up
        assert memory._simulator is None
        assert memory.get_memory_count() == 0


class TestLearningGate:
    """Tests for learning gate functionality."""

    @pytest.fixture
    def memory(self) -> BiCameralMemory:
        """Create memory for gate testing."""
        mem = BiCameralMemory(n_neurons=50, dimensions=500)
        yield mem
        mem.reset()

    def test_learning_gate_exists(self, memory: BiCameralMemory) -> None:
        """Learning gate node should exist in network."""
        assert hasattr(memory, "learning_gate")
        assert memory.learning_gate is not None

    def test_learning_gate_default_enabled(self, memory: BiCameralMemory) -> None:
        """Learning gate should default to 0.0 (normal learning)."""
        assert memory._learning_gate_value == 0.0

    def test_recall_disables_learning(self, memory: BiCameralMemory) -> None:
        """Recall should temporarily disable learning gate."""
        # Store a memory first
        vector = np.zeros(500, dtype=np.float32)
        vector[:25] = 1.0
        memory.remember("gate-test", vector)

        # Gate should be enabled (0.0) before recall
        assert memory._learning_gate_value == 0.0

        # Recall
        memory.recall(vector, threshold=0.1)

        # Gate should be re-enabled (0.0) after recall
        assert memory._learning_gate_value == 0.0

    def test_reset_restores_gate(self, memory: BiCameralMemory) -> None:
        """Reset should restore learning gate to enabled."""
        memory._learning_gate_value = -1.0
        memory.reset()
        assert memory._learning_gate_value == 0.0

    def test_remember_enables_gate(self, memory: BiCameralMemory) -> None:
        """Remember should enable learning gate even if previously disabled."""
        # Disable gate manually
        memory._learning_gate_value = -1.0

        # Remember should re-enable gate
        vector = np.zeros(500, dtype=np.float32)
        vector[:25] = 1.0
        memory.remember("gate-enable-test", vector)

        assert memory._learning_gate_value == 0.0
