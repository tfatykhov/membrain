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

from membrain.logging import get_logger

if TYPE_CHECKING:
    import nengo

logger = get_logger(__name__)

# Check for nengo availability
try:
    import nengo

    HAS_NENGO = True
except ImportError:
    HAS_NENGO = False

# Constants
DEFAULT_LEARN_DURATION_MS = 50
DEFAULT_RECALL_DURATION_MS = 20
DEFAULT_CONSOLIDATE_DURATION_MS = 1000
CONNECTIVITY_FACTOR = 0.1  # Estimated sparse connectivity for synop count
SIMILARITY_EPS = 1e-9  # Epsilon for floating-point comparisons


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
        seed: int | None = None,
    ) -> None:
        """
        Initialize the BiCameralMemory network.

        Args:
            n_neurons: Number of neurons in memory ensemble.
            dimensions: Dimension of sparse input (FlyHash output).
            learning_rate: Base Voja learning rate.
            synapse: Synaptic time constant in seconds.
            dt: Simulation timestep in seconds.
            seed: Random seed for reproducibility.

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
        self._seed = seed

        # RNG for stochastic operations (consolidation noise)
        self._rng = np.random.default_rng(seed)

        # Memory index: context_id -> MemoryEntry
        self._memory_index: dict[str, MemoryEntry] = {}

        # Input value (set before simulation)
        self._input_value: NDArray[np.float32] = np.zeros(
            self.dimensions, dtype=np.float32
        )

        # Learning gate value: 0.0 = normal learning, -1.0 = disable learning
        # (Voja modulation: output is multiplier-1, so 0=learn, -1=no learn)
        self._learning_gate_value: float = 0.0

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
            # Voja modulation: 0.0 = normal learning, -1.0 = disable learning
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
        duration_ms: int = DEFAULT_LEARN_DURATION_MS,
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
            ValueError: If sparse_vector has wrong shape or importance out of range.
        """
        self._ensure_simulator()
        assert self._simulator is not None

        # Validate importance
        if not 0.0 <= importance <= 1.0:
            raise ValueError(f"importance must be between 0.0 and 1.0, got {importance}")

        # Validate input
        if sparse_vector.shape != (self.dimensions,):
            raise ValueError(
                f"Expected shape ({self.dimensions},), got {sparse_vector.shape}"
            )

        # Set input and run simulation with learning
        # Scale input by importance - this modulates learning strength
        # (Higher importance = stronger input = faster encoder adaptation)
        scaled_vector = sparse_vector.astype(np.float32) * importance
        self._input_value = scaled_vector.copy()

        # Ensure learning gate is enabled for remember (0.0 = normal learning)
        self._learning_gate_value = 0.0

        steps = max(1, int(duration_ms / (self.dt * 1000)))
        self._simulator.run_steps(steps)

        # Store the SCALED vector in index (matches what network learned)
        self._memory_index[context_id] = MemoryEntry(
            context_id=context_id,
            sparse_vector=scaled_vector.copy(),
            importance=importance,
            stored_at=self._simulator.time,
        )

        # Clear input
        self._input_value = np.zeros(self.dimensions, dtype=np.float32)

        logger.info(
            "Memory stored",
            extra={
                "context_id": context_id,
                "importance": importance,
                "memory_count": len(self._memory_index),
            },
        )

        return True

    def recall(
        self,
        query_vector: NDArray[np.floating],
        threshold: float = 0.7,
        max_results: int = 5,
        duration_ms: int = DEFAULT_RECALL_DURATION_MS,
        bypass_snn: bool = False,
    ) -> list[RecallResult]:
        """
        Recall memories via pattern completion.

        Args:
            query_vector: FlyHash-encoded query.
            threshold: Minimum similarity threshold.
            max_results: Maximum number of results.
            duration_ms: Simulation duration in milliseconds.
            bypass_snn: If True, skip SNN and use direct cosine similarity.
                       Useful for benchmarking baseline before attractor dynamics.

        Returns:
            List of RecallResult with context_id and confidence.

        Raises:
            ValueError: If query_vector has wrong shape.
        """
        # Validate input
        if query_vector.shape != (self.dimensions,):
            raise ValueError(
                f"Expected shape ({self.dimensions},), got {query_vector.shape}"
            )

        query = query_vector.astype(np.float32).copy()

        if bypass_snn:
            # Direct cosine similarity (no SNN dynamics)
            # This is the baseline before attractor dynamics are implemented
            comparison_vector = query
        else:
            # Full SNN path
            self._ensure_simulator()
            assert self._simulator is not None

            # Disable learning during recall via gate (-1.0 = no learning)
            self._learning_gate_value = -1.0

            # Inject query
            self._input_value = query

            steps = max(1, int(duration_ms / (self.dt * 1000)))
            self._simulator.run_steps(steps)

            # Re-enable learning (0.0 = normal learning)
            self._learning_gate_value = 0.0

            # Read output probe (attractor state)
            output_data = self._simulator.data[self.output_probe]
            comparison_vector = output_data[-1]  # Last timestep

            # Clear input
            self._input_value = np.zeros(self.dimensions, dtype=np.float32)

        # Match against stored memories
        results: list[RecallResult] = []
        for entry in self._memory_index.values():
            similarity = self._compute_similarity(comparison_vector, entry.sparse_vector)
            if similarity >= threshold:
                results.append(RecallResult(entry.context_id, similarity))

        # Sort by confidence descending
        results.sort(key=lambda x: x.confidence, reverse=True)
        final_results = results[:max_results]

        logger.info(
            "Recall completed",
            extra={
                "matches": len(final_results),
                "threshold": threshold,
                "memory_count": len(self._memory_index),
                "bypass_snn": bypass_snn,
            },
        )

        return final_results

    def consolidate(
        self,
        noise_scale: float = 0.05,
        max_steps: int = 50,
        convergence_threshold: float = 1e-4,
        prune_weak: bool = False,
        prune_threshold: float = 0.1,
    ) -> tuple[int, int]:
        """
        Run stochastic consolidation phase (attractor dynamics).

        Injects Gaussian white noise into the network state, then iterates
        recurrent dynamics until the system settles into an attractor state.
        This mimics biological hippocampal consolidation during sleep.

        Args:
            noise_scale: Standard deviation of Gaussian noise injected.
                        Essential for escaping local minima.
            max_steps: Maximum recurrent cycles allowed for settling.
            convergence_threshold: State difference magnitude to consider 'settled'.
            prune_weak: Whether to prune weak associations after settling.
            prune_threshold: Importance threshold for pruning.

        Returns:
            Tuple of (steps_to_converge, pruned_count).
            steps_to_converge is -1 if max_steps reached without convergence.
        """
        if self._simulator is None:
            self._ensure_simulator()

        assert self._simulator is not None

        # Get current state from output probe
        output_data = self._simulator.data[self.output_probe]
        if len(output_data) == 0:
            # No prior simulation, start with zeros
            current_state = np.zeros(self.dimensions, dtype=np.float32)
        else:
            current_state = output_data[-1].astype(np.float32)

        # Inject Gaussian white noise (the "kick" to escape local minima)
        # Instance RNG ensures different noise each call (seeded for reproducibility)
        noise = self._rng.normal(0, noise_scale, current_state.shape).astype(np.float32)
        perturbed_state = current_state + noise

        # Set perturbed state as input and disable learning during consolidation
        self._input_value = perturbed_state
        self._learning_gate_value = -1.0  # Disable learning (Voja: -1 = no learn)

        steps_to_converge = -1

        # Iterative settling (attractor dynamics)
        for step in range(max_steps):
            # Run one simulation step
            self._simulator.run_steps(1)

            # Get new state
            new_output = self._simulator.data[self.output_probe]
            new_state = new_output[-1].astype(np.float32)

            # Check for convergence (did we find the attractor?)
            diff = float(np.linalg.norm(new_state - perturbed_state))
            if diff < convergence_threshold:
                steps_to_converge = step + 1
                break

            # Update for next iteration (no external input, only recurrent)
            perturbed_state = new_state
            self._input_value = np.zeros(self.dimensions, dtype=np.float32)

        # Re-enable learning and clear input
        self._learning_gate_value = 0.0
        self._input_value = np.zeros(self.dimensions, dtype=np.float32)

        # Prune weak memories if requested
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

        logger.info(
            "Consolidation completed",
            extra={
                "steps_to_converge": steps_to_converge,
                "pruned_count": pruned_count,
                "noise_scale": noise_scale,
                "memory_count": len(self._memory_index),
            },
        )

        return (steps_to_converge, pruned_count)

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
        return int(total_spikes * self.dimensions * CONNECTIVITY_FACTOR)

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
        eps = SIMILARITY_EPS
        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))
        if norm1 < eps or norm2 < eps:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def reset(self) -> None:
        """Reset the memory system and release resources."""
        if self._simulator is not None:
            self._simulator.close()
            self._simulator = None
        self._memory_index.clear()
        self._input_value = np.zeros(self.dimensions, dtype=np.float32)
        self._learning_gate_value = 0.0  # Reset to normal learning

    def __del__(self) -> None:
        """Clean up simulator resources on garbage collection."""
        if hasattr(self, "_simulator") and self._simulator is not None:
            try:
                self._simulator.close()
            except Exception:
                pass  # Ignore errors during cleanup

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
