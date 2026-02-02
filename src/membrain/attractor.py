"""Attractor memory for pattern completion.

Hopfield-style auto-associative memory that cleans up noisy patterns
by converging to stored attractors through recurrent dynamics.

This is a numpy prototype for algorithm validation. The production
implementation will use Nengo for Loihi compatibility.

References:
    - Hopfield (1982): Neural networks and physical systems
    - Minsky, Society of Mind Ch. 4: The Self (conservative self = attractor basins)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_MAX_STEPS = 50
DEFAULT_CONVERGENCE_THRESHOLD = 1e-5
DEFAULT_SPARSITY_PERCENTILE = 90


@dataclass
class CleanupResult:
    """Result of pattern cleanup operation."""

    cleaned: NDArray[np.float32]
    """The cleaned pattern after attractor dynamics."""

    steps: int
    """Number of steps to convergence."""

    converged: bool
    """Whether dynamics converged within max_steps."""

    input_norm: float
    """L2 norm of input pattern."""

    output_norm: float
    """L2 norm of output pattern."""


@dataclass
class CleanupMetrics:
    """Metrics for evaluating cleanup effectiveness."""

    input_similarity: float
    """Cosine similarity between original and noisy input."""

    output_similarity: float
    """Cosine similarity between original and cleaned output."""

    cleanup_gain: float
    """Improvement from cleanup: output_similarity - input_similarity."""

    steps: int
    """Steps taken during cleanup."""


class AttractorMemory:
    """Hopfield-style attractor network for pattern cleanup.

    Stores patterns via Hebbian learning (outer product rule) and
    retrieves them through iterative dynamics that converge to
    the nearest attractor basin.

    This implements Minsky's "conservative self" concept: stored patterns
    create basins that resist perturbation and pull noisy inputs toward
    stable states.

    Attributes:
        dimensions: Dimensionality of patterns.
        learning_rate: Strength of Hebbian updates.
        max_steps: Maximum dynamics iterations.
        convergence_threshold: State change threshold for convergence.
        sparsity_percentile: Percentile for lateral inhibition.

    Example:
        >>> attractor = AttractorMemory(dimensions=128)
        >>> pattern = np.random.randn(128).astype(np.float32)
        >>> pattern /= np.linalg.norm(pattern)
        >>> attractor.store(pattern)
        >>> noisy = pattern + 0.3 * np.random.randn(128).astype(np.float32)
        >>> result = attractor.complete(noisy)
        >>> # result.cleaned should be closer to pattern than noisy was
    """

    def __init__(
        self,
        dimensions: int,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_steps: int = DEFAULT_MAX_STEPS,
        convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
        sparsity_percentile: float = DEFAULT_SPARSITY_PERCENTILE,
        seed: int | None = None,
    ) -> None:
        """Initialize attractor memory.

        Args:
            dimensions: Pattern dimensionality.
            learning_rate: Hebbian learning rate (basin depth).
            max_steps: Max iterations for dynamics.
            convergence_threshold: When to consider converged.
            sparsity_percentile: Lateral inhibition threshold.
            seed: RNG seed for reproducibility.
        """
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}")
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        self.dimensions = dimensions
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold
        self.sparsity_percentile = sparsity_percentile

        self._rng = np.random.default_rng(seed)

        # Recurrent weight matrix (Hebbian connections)
        self._weights: NDArray[np.float32] = np.zeros(
            (dimensions, dimensions), dtype=np.float32
        )

        # Track stored patterns for diagnostics
        self._pattern_count = 0

        logger.info(
            "AttractorMemory initialized",
            extra={
                "dimensions": dimensions,
                "learning_rate": learning_rate,
                "max_steps": max_steps,
            },
        )

    @property
    def pattern_count(self) -> int:
        """Number of patterns stored."""
        return self._pattern_count

    def store(self, pattern: NDArray[np.floating]) -> None:
        """Store a pattern as an attractor via Hebbian learning.

        Uses the outer product rule to create a basin of attraction
        around the pattern. Multiple stores of the same pattern
        deepen its basin (rehearsal effect).

        Args:
            pattern: Normalized pattern vector (L2 norm should be ~1).

        Raises:
            ValueError: If pattern has wrong shape.
        """
        if pattern.shape != (self.dimensions,):
            raise ValueError(
                f"Expected shape ({self.dimensions},), got {pattern.shape}"
            )

        # Normalize input
        norm = np.linalg.norm(pattern)
        if norm < 1e-10:
            logger.warning("Skipping near-zero pattern")
            return

        normalized = (pattern / norm).astype(np.float32)

        # Hebbian update: outer product
        # This creates a fixed point at the pattern
        update = self.learning_rate * np.outer(normalized, normalized)

        # Zero diagonal to prevent self-reinforcement runaway
        np.fill_diagonal(update, 0)

        self._weights += update
        self._pattern_count += 1

        logger.debug(
            "Pattern stored",
            extra={
                "pattern_count": self._pattern_count,
                "weight_norm": float(np.linalg.norm(self._weights)),
            },
        )

    def complete(self, partial: NDArray[np.floating]) -> CleanupResult:
        """Complete a partial/noisy pattern using attractor dynamics.

        Runs iterative dynamics until convergence or max_steps.
        The pattern evolves through state space, pulled toward
        the nearest stored attractor.

        Args:
            partial: Noisy or partial input pattern.

        Returns:
            CleanupResult with cleaned pattern and metadata.

        Raises:
            ValueError: If partial has wrong shape.
        """
        if partial.shape != (self.dimensions,):
            raise ValueError(
                f"Expected shape ({self.dimensions},), got {partial.shape}"
            )

        input_norm = float(np.linalg.norm(partial))

        # Initialize state
        state = partial.astype(np.float32).copy()
        if input_norm > 1e-10:
            state = state / input_norm  # Normalize

        converged = False
        steps = 0

        for step in range(self.max_steps):
            prev_state = state.copy()

            # Dynamics: weighted sum through recurrent connections
            activation = self._weights @ state

            # Nonlinearity: tanh for bounded output
            state = np.tanh(activation).astype(np.float32)

            # Re-normalize (inhibition applied after loop to avoid oscillation)
            state_norm = np.linalg.norm(state)
            if state_norm > 1e-10:
                state = state / state_norm

            steps = step + 1

            # Check convergence
            delta = np.linalg.norm(state - prev_state)
            if delta < self.convergence_threshold:
                converged = True
                break

        # Apply lateral inhibition to final state (not during dynamics)
        # This prevents oscillation from hard thresholding
        state = self._apply_inhibition(state)
        state_norm = np.linalg.norm(state)
        if state_norm > 1e-10:
            state = state / state_norm

        output_norm = float(np.linalg.norm(state))

        logger.debug(
            "Pattern completion finished",
            extra={
                "steps": steps,
                "converged": converged,
                "output_norm": output_norm,
            },
        )

        return CleanupResult(
            cleaned=state,
            steps=steps,
            converged=converged,
            input_norm=input_norm,
            output_norm=output_norm,
        )

    def _apply_inhibition(self, state: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply lateral inhibition to prevent blend states.

        Keeps only the top activations, zeroing out weak ones.
        This implements winner-take-all dynamics that ensure
        the network settles to a single attractor.

        Uses np.partition for O(N) complexity instead of O(N log N) sorting.

        Args:
            state: Current state vector.

        Returns:
            Sparsified state vector.
        """
        if self.sparsity_percentile >= 100:
            return state  # No inhibition

        # Calculate k: number of elements to zero out
        k = int((self.sparsity_percentile / 100.0) * len(state))
        if k <= 0 or k >= len(state):
            return state

        # Use partition for O(N) threshold finding
        abs_state = np.abs(state)
        threshold = np.partition(abs_state, k)[k]

        return np.where(abs_state > threshold, state, 0).astype(np.float32)

    def measure_cleanup(
        self,
        original: NDArray[np.floating],
        noisy: NDArray[np.floating],
    ) -> CleanupMetrics:
        """Measure cleanup effectiveness.

        Compares similarity before and after cleanup to quantify
        how much the attractor dynamics helped.

        Args:
            original: The ground truth pattern.
            noisy: The corrupted version to clean up.

        Returns:
            CleanupMetrics with input/output similarity and gain.
        """
        # Normalize inputs
        orig_norm = np.linalg.norm(original)
        noisy_norm = np.linalg.norm(noisy)

        if orig_norm < 1e-10 or noisy_norm < 1e-10:
            return CleanupMetrics(
                input_similarity=0.0,
                output_similarity=0.0,
                cleanup_gain=0.0,
                steps=0,
            )

        orig_normalized = original / orig_norm
        noisy_normalized = noisy / noisy_norm

        # Input similarity (before cleanup)
        input_sim = float(np.dot(orig_normalized, noisy_normalized))

        # Run cleanup
        result = self.complete(noisy)

        # Output similarity (after cleanup)
        output_sim = float(np.dot(orig_normalized, result.cleaned))

        return CleanupMetrics(
            input_similarity=input_sim,
            output_similarity=output_sim,
            cleanup_gain=output_sim - input_sim,
            steps=result.steps,
        )

    def clear(self) -> None:
        """Clear all stored patterns."""
        self._weights.fill(0)
        self._pattern_count = 0
        logger.info("AttractorMemory cleared")
