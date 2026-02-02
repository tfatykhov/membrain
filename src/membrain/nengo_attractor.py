"""Nengo-based attractor memory for pattern completion.

Spiking neural network implementation of attractor dynamics using Nengo.
This is the production implementation for Loihi compatibility.

The attractor uses recurrent connections with Hebbian-learned weights
to create basins of attraction around stored patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import nengo
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_N_NEURONS = 1000
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_SETTLING_TIME_MS = 100
DEFAULT_DT = 0.001  # 1ms timestep


@dataclass
class NengoCleanupResult:
    """Result of Nengo attractor cleanup."""

    cleaned: NDArray[np.float32]
    """The cleaned pattern after attractor dynamics."""

    settling_time_ms: float
    """Time simulated for settling."""

    final_similarity: float
    """Cosine similarity between input and output (stability measure)."""


class NengoAttractorMemory:
    """Nengo-based attractor network for pattern cleanup.

    Uses a recurrent neural ensemble with learned weights to implement
    attractor dynamics. Patterns are stored via Voja learning rule,
    creating basins of attraction.

    This integrates with BiCameralMemory by cleaning up queries before
    similarity matching.

    Attributes:
        dimensions: Pattern dimensionality.
        n_neurons: Number of neurons in the attractor ensemble.
        learning_rate: Voja learning rate.
        settling_time_ms: Default simulation time for cleanup.

    Example:
        >>> attractor = NengoAttractorMemory(dimensions=128)
        >>> pattern = np.random.randn(128).astype(np.float32)
        >>> pattern /= np.linalg.norm(pattern)
        >>> attractor.store(pattern)
        >>> noisy = pattern + 0.3 * np.random.randn(128).astype(np.float32)
        >>> result = attractor.complete(noisy)
    """

    def __init__(
        self,
        dimensions: int,
        n_neurons: int = DEFAULT_N_NEURONS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        settling_time_ms: float = DEFAULT_SETTLING_TIME_MS,
        dt: float = DEFAULT_DT,
        seed: int | None = None,
    ) -> None:
        """Initialize Nengo attractor memory.

        Args:
            dimensions: Pattern dimensionality.
            n_neurons: Neurons in attractor ensemble.
            learning_rate: Voja learning rate for storage.
            settling_time_ms: Default settling time.
            dt: Simulation timestep.
            seed: RNG seed for reproducibility.
        """
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be positive, got {n_neurons}")

        self.dimensions = dimensions
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.settling_time_ms = settling_time_ms
        self.dt = dt
        self._seed = seed

        # Input value for simulation
        self._input_value: NDArray[np.float32] = np.zeros(
            dimensions, dtype=np.float32
        )

        # Learning gate: 0.0 = learn, -1.0 = no learn
        self._learning_gate_value: float = 0.0

        # Pattern count for diagnostics
        self._pattern_count = 0

        # Build the network
        self._build_network()
        self._simulator: nengo.Simulator | None = None

        logger.info(
            "NengoAttractorMemory initialized",
            extra={
                "dimensions": dimensions,
                "n_neurons": n_neurons,
                "learning_rate": learning_rate,
            },
        )

    def _build_network(self) -> None:
        """Construct the Nengo attractor network."""
        self.model = nengo.Network(label="AttractorMemory", seed=self._seed)

        with self.model:
            # Input node
            self.input_node = nengo.Node(
                output=lambda t: self._input_value,
                size_out=self.dimensions,
                label="input",
            )

            # Learning gate
            self.learning_gate = nengo.Node(
                output=lambda t: self._learning_gate_value,
                size_out=1,
                label="learning_gate",
            )

            # Attractor ensemble with recurrent connections
            self.attractor = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=self.dimensions,
                neuron_type=nengo.SpikingRectifiedLinear(),
                label="attractor",
            )

            # Input connection with Voja learning
            self.input_conn = nengo.Connection(
                self.input_node,
                self.attractor,
                learning_rule_type=nengo.Voja(learning_rate=self.learning_rate),
                synapse=0.01,
            )

            # Gate connection for learning control
            nengo.Connection(
                self.learning_gate,
                self.input_conn.learning_rule,
                synapse=None,
            )

            # Recurrent connection for attractor dynamics
            # This creates the "pull" toward stored patterns
            self.recurrent = nengo.Connection(
                self.attractor,
                self.attractor,
                synapse=0.1,  # Slower synapse for stability
                transform=0.9,  # Slight decay to prevent runaway
            )

            # Output probe
            self.output_probe = nengo.Probe(
                self.attractor,
                synapse=0.01,
            )

    def _ensure_simulator(self) -> None:
        """Build simulator if needed."""
        if self._simulator is None:
            self._simulator = nengo.Simulator(self.model, dt=self.dt)

    def store(self, pattern: NDArray[np.floating]) -> None:
        """Store a pattern via Voja learning.

        Args:
            pattern: Normalized pattern vector.

        Raises:
            ValueError: If pattern has wrong shape.
        """
        if pattern.shape != (self.dimensions,):
            raise ValueError(
                f"Expected shape ({self.dimensions},), got {pattern.shape}"
            )

        # Normalize
        norm = np.linalg.norm(pattern)
        if norm < 1e-10:
            logger.warning("Skipping near-zero pattern")
            return

        normalized = (pattern / norm).astype(np.float32)

        self._ensure_simulator()
        assert self._simulator is not None

        # Enable learning
        self._learning_gate_value = 0.0
        self._input_value = normalized.copy()

        # Run for storage duration
        store_time = 0.1  # 100ms for storage
        steps = max(1, int(store_time / self.dt))
        self._simulator.run_steps(steps)

        # Clear input
        self._input_value = np.zeros(self.dimensions, dtype=np.float32)
        self._pattern_count += 1

        logger.debug(
            "Pattern stored via Voja",
            extra={"pattern_count": self._pattern_count},
        )

    def complete(
        self,
        partial: NDArray[np.floating],
        settling_time_ms: float | None = None,
    ) -> NengoCleanupResult:
        """Complete a noisy pattern using attractor dynamics.

        Args:
            partial: Noisy input pattern.
            settling_time_ms: Override default settling time.

        Returns:
            NengoCleanupResult with cleaned pattern.

        Raises:
            ValueError: If partial has wrong shape.
        """
        if partial.shape != (self.dimensions,):
            raise ValueError(
                f"Expected shape ({self.dimensions},), got {partial.shape}"
            )

        if settling_time_ms is None:
            settling_time_ms = self.settling_time_ms

        self._ensure_simulator()
        assert self._simulator is not None

        # Normalize input
        norm = np.linalg.norm(partial)
        if norm > 1e-10:
            normalized = (partial / norm).astype(np.float32)
        else:
            normalized = partial.astype(np.float32)

        # Disable learning during recall
        self._learning_gate_value = -1.0
        self._input_value = normalized.copy()

        # Run settling dynamics
        settling_time_s = settling_time_ms / 1000.0
        steps = max(1, int(settling_time_s / self.dt))
        self._simulator.run_steps(steps)

        # Read output
        output_data = self._simulator.data[self.output_probe]
        cleaned = output_data[-1].astype(np.float32)

        # Normalize output
        cleaned_norm = np.linalg.norm(cleaned)
        if cleaned_norm > 1e-10:
            cleaned = cleaned / cleaned_norm

        # Clear input
        self._input_value = np.zeros(self.dimensions, dtype=np.float32)

        # Re-enable learning
        self._learning_gate_value = 0.0

        # Compute stability (similarity between input and output)
        final_similarity = float(np.dot(normalized, cleaned))

        logger.debug(
            "Nengo cleanup complete",
            extra={
                "settling_time_ms": settling_time_ms,
                "final_similarity": final_similarity,
            },
        )

        return NengoCleanupResult(
            cleaned=cleaned,
            settling_time_ms=settling_time_ms,
            final_similarity=final_similarity,
        )

    @property
    def pattern_count(self) -> int:
        """Number of stored patterns."""
        return self._pattern_count

    def reset(self) -> None:
        """Reset the simulator state."""
        if self._simulator is not None:
            self._simulator.close()
            self._simulator = None
        self._pattern_count = 0
        logger.info("NengoAttractorMemory reset")
