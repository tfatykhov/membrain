"""
BiCameralMemory - Spiking Neural Network for Associative Memory.

Implements a hippocampus-like memory system using Nengo for neural simulation.
Supports learning (Voja rule), recall via pattern completion, and consolidation.

Reference: PRD Feature 03 - Neuromorphic Core
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import nengo

# Check for nengo availability
try:
    import nengo

    HAS_NENGO = True
except ImportError:
    HAS_NENGO = False


@dataclass
class MemoryEntry:
    """A stored memory entry."""

    context_id: str
    sparse_vector: NDArray[np.float32]
    importance: float
    stored_at: float


@dataclass
class RecallResult:
    """Result of a memory recall operation."""

    context_id: str
    confidence: float


class BiCameralMemory:
    """
    Spiking Neural Network for associative memory.

    Implements a hippocampus-like memory system using Nengo.
    Stores memories via Voja learning and retrieves via pattern completion.

    Attributes:
        n_neurons: Number of neurons in memory ensemble.
        dimensions: Dimension of sparse input vectors (FlyHash output).
        learning_rate: Base Voja learning rate.
        synapse: Synaptic time constant in seconds.
        dt: Simulation timestep in seconds.

    Example:
        >>> memory = BiCameralMemory(n_neurons=500, dimensions=10000)
        >>> memory.remember("doc-001", sparse_vector, importance=0.8)
        >>> results = memory.recall(query_vector, threshold=0.5)
    """

    def __init__(
        self,
        n_neurons: int = 1000,
        dimensions: int = 20000,
        learning_rate: float = 1e-2,
        synapse: float = 0.01,
        dt: float = 0.001,
    ) -> None:
        """
        Initialize the BiCameralMemory network.

        Args:
            n_neurons: Number of neurons in memory ensemble.
            dimensions: Dimension of sparse input (FlyHash output).
            learning_rate: Base Voja learning rate.
            synapse: Synaptic time constant in seconds.
            dt: Simulation timestep in seconds.

        Raises:
            ImportError: If nengo is not installed.
        """
        if not HAS_NENGO:
            raise ImportError(
                "nengo is required for BiCameralMemory. "
                "Install with: pip install nengo"
            )

        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.synapse = synapse
        self.dt = dt

        # Memory index: context_id -> MemoryEntry
        self._memory_index: dict[str, MemoryEntry] = {}

        # Input value (set before simulation)
        self._input_value: NDArray[np.float32] = np.zeros(
            self.dimensions, dtype=np.float32
        )

        # Learning gate value (1.0 = learn, 0.0 = recall-only)
        self._learning_gate_value: float = 1.0

        # Build the Nengo network
        self._build_network()
        self._simulator: nengo.Simulator | None = None

    def _build_network(self) -> None:
        """Construct the Nengo neural network."""
        self.model = nengo.Network(label="Hippocampus")

        with self.model:
            # 1. Input Node: Receives sparse spike trains from FlyHash
            self.input_node = nengo.Node(
                output=lambda t: self._input_value,
                size_out=self.dimensions,
                label="input",
            )

            # 2. Learning Gate Node: Controls whether learning is active
            # 1.0 = learning enabled, 0.0 = learning disabled (recall-only)
            self.learning_gate = nengo.Node(
                output=lambda t: self._learning_gate_value,
                size_out=1,
                label="learning_gate",
            )

            # 3. Memory Ensemble: The core storage
            # Use standard LIF neurons (Loihi-compatible neurons require nengo-loihi)
            self.memory = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=self.dimensions,
                neuron_type=nengo.SpikingRectifiedLinear(),
                label="memory_ensemble",
            )

            # 4. Learning Connection (Plasticity)
            # Voja learning rule for unsupervised association
            self.learning_conn = nengo.Connection(
                self.input_node,
                self.memory,
                learning_rule_type=nengo.Voja(learning_rate=self.learning_rate),
                synapse=self.synapse,
            )

            # 5. Gate connection: modulates the learning rule
            # When gate=0, learning is effectively disabled
            nengo.Connection(
                self.learning_gate,
                self.learning_conn.learning_rule,
                synapse=None,
            )

            # 6. Output Probe for reading attractor states
            self.output_probe = nengo.Probe(self.memory, synapse=self.synapse)

            # 7. Spike probe for sparsity monitoring
            self.spike_probe = nengo.Probe(self.memory.neurons, "output")

    def _ensure_simulator(self) -> None:
        """Build simulator if not already built."""
        if self._simulator is None:
            self._simulator = nengo.Simulator(self.model, dt=self.dt)

    def remember(
        self,
        context_id: str,
        sparse_vector: NDArray[np.floating],
        importance: float = 1.0,
        duration_ms: int = 50,
    ) -> bool:
        """
        Store a memory with learning enabled.

        Args:
            context_id: UUID of the memory.
            sparse_vector: FlyHash-encoded sparse vector.
            importance: Learning rate modifier (0.0-1.0).
            duration_ms: Simulation duration in milliseconds.

        Returns:
            True if successfully stored.

        Raises:
            ValueError: If sparse_vector has wrong shape.
        """
        self._ensure_simulator()
        assert self._simulator is not None

        # Validate input
        if sparse_vector.shape != (self.dimensions,):
            raise ValueError(
                f"Expected shape ({self.dimensions},), got {sparse_vector.shape}"
            )

        # Set input and run simulation with learning
        self._input_value = sparse_vector.astype(np.float32).copy()
        self._learning_enabled = True

        # Modulate learning rate by importance
        # Note: In a full implementation, we'd modify the learning rule's rate
        # For now, we scale the input signal
        self._input_value *= importance

        steps = int(duration_ms / (self.dt * 1000))
        self._simulator.run_steps(steps)

        # Store in index
        self._memory_index[context_id] = MemoryEntry(
            context_id=context_id,
            sparse_vector=sparse_vector.astype(np.float32).copy(),
            importance=importance,
            stored_at=self._simulator.time,
        )

        # Clear input
        self._input_value = np.zeros(self.dimensions, dtype=np.float32)

        return True

    def recall(
        self,
        query_vector: NDArray[np.floating],
        threshold: float = 0.7,
        max_results: int = 5,
        duration_ms: int = 20,
    ) -> list[RecallResult]:
        """
        Recall memories via pattern completion.

        Args:
            query_vector: FlyHash-encoded query.
            threshold: Minimum similarity threshold.
            max_results: Maximum number of results.
            duration_ms: Simulation duration in milliseconds.

        Returns:
            List of RecallResult with context_id and confidence.

        Raises:
            ValueError: If query_vector has wrong shape.
        """
        self._ensure_simulator()
        assert self._simulator is not None

        # Validate input
        if query_vector.shape != (self.dimensions,):
            raise ValueError(
                f"Expected shape ({self.dimensions},), got {query_vector.shape}"
            )

        # Disable learning during recall via gate
        self._learning_gate_value = 0.0

        # Inject query
        self._input_value = query_vector.astype(np.float32).copy()

        steps = int(duration_ms / (self.dt * 1000))
        self._simulator.run_steps(steps)

        # Re-enable learning
        self._learning_gate_value = 1.0

        # Read output probe (attractor state)
        output_data = self._simulator.data[self.output_probe]
        attractor_state = output_data[-1]  # Last timestep

        # Clear input
        self._input_value = np.zeros(self.dimensions, dtype=np.float32)

        # Match against stored memories
        results: list[RecallResult] = []
        for entry in self._memory_index.values():
            similarity = self._compute_similarity(attractor_state, entry.sparse_vector)
            if similarity >= threshold:
                results.append(RecallResult(entry.context_id, similarity))

        # Sort by confidence descending
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:max_results]

    def consolidate(
        self,
        duration_ms: int = 1000,
        prune_weak: bool = False,
        prune_threshold: float = 0.1,
    ) -> int:
        """
        Run consolidation phase (sleep).

        Args:
            duration_ms: Duration of consolidation in milliseconds.
            prune_weak: Whether to prune weak associations.
            prune_threshold: Importance threshold for pruning.

        Returns:
            Number of memories pruned (if prune_weak=True).
        """
        if self._simulator is None:
            return 0

        # Run with no input (let network settle)
        self._input_value = np.zeros(self.dimensions, dtype=np.float32)
        steps = int(duration_ms / (self.dt * 1000))
        self._simulator.run_steps(steps)

        pruned_count = 0
        if prune_weak:
            to_remove = [
                cid
                for cid, entry in self._memory_index.items()
                if entry.importance < prune_threshold
            ]
            for cid in to_remove:
                del self._memory_index[cid]
                pruned_count += 1

        return pruned_count

    def get_sparsity_rate(self) -> float:
        """
        Calculate neuron activity sparsity.

        Returns:
            Fraction of neurons that are inactive (target: >0.90).
        """
        if self._simulator is None:
            return 1.0

        spike_data = self._simulator.data[self.spike_probe]
        if len(spike_data) == 0:
            return 1.0

        # Count neurons that fired at least once per timestep
        active_per_step = np.sum(spike_data > 0, axis=1)
        avg_active: float = float(np.mean(active_per_step))

        return 1.0 - (avg_active / self.n_neurons)

    def get_synop_count(self) -> int:
        """
        Get synaptic operation count estimate.

        Returns:
            Estimated number of synaptic operations.
        """
        if self._simulator is None:
            return 0

        spike_data = self._simulator.data[self.spike_probe]
        total_spikes = int(np.sum(spike_data > 0))

        # Each spike triggers synaptic operations proportional to connectivity
        # Assume ~10% sparse connectivity
        return int(total_spikes * self.dimensions * 0.1)

    def get_memory_count(self) -> int:
        """Return number of stored memories."""
        return len(self._memory_index)

    def get_memory_ids(self) -> list[str]:
        """Return list of stored memory IDs."""
        return list(self._memory_index.keys())

    def _compute_similarity(
        self, vec1: NDArray[np.floating], vec2: NDArray[np.floating]
    ) -> float:
        """Compute cosine similarity between vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def reset(self) -> None:
        """Reset the memory system and release resources."""
        if self._simulator is not None:
            self._simulator.close()
            self._simulator = None
        self._memory_index.clear()
        self._input_value = np.zeros(self.dimensions, dtype=np.float32)
        self._learning_gate_value = 1.0

    def __enter__(self) -> BiCameralMemory:
        """Context manager entry."""
        self._ensure_simulator()
        assert self._simulator is not None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.reset()

    def __repr__(self) -> str:
        return (
            f"BiCameralMemory(n_neurons={self.n_neurons}, "
            f"dimensions={self.dimensions}, memories={self.get_memory_count()})"
        )
