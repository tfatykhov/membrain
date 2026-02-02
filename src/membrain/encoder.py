"""
FlyHash Encoder - Locality-Sensitive Hashing for Neuromorphic Memory.

Converts dense LLM embeddings (e.g., 1536-d from OpenAI) into sparse binary
codes suitable for efficient Spiking Neural Network processing.

Inspired by the fruit fly olfactory system. Uses random projections followed
by Winner-Take-All (WTA) inhibition to create sparse, locality-sensitive hash codes.

Reference: https://arxiv.org/abs/1711.03127
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from membrain.logging import get_logger

logger = get_logger(__name__)


class FlyHash:
    """
    Locality-Sensitive Hashing using the FlyHash algorithm.

    Converts dense embedding vectors into sparse binary codes
    suitable for Spiking Neural Network processing.

    Uses int8 {-1, +1} projection matrix for 8x memory reduction
    compared to float64 Gaussian projections.

    Attributes:
        input_dim: Dimension of input vectors.
        output_dim: Dimension of output sparse codes.
        active_bits: Number of active bits (k) in output.
        _projection: Random binary projection matrix (int8).

    Example:
        >>> encoder = FlyHash(input_dim=1536, seed=42)
        >>> vector = np.random.randn(1536).astype(np.float32)
        >>> sparse_code = encoder.encode(vector)
        >>> assert sparse_code.shape == (encoder.output_dim,)
        >>> assert np.sum(sparse_code) == encoder.active_bits
    """

    def __init__(
        self,
        input_dim: int = 1536,
        expansion_ratio: float = 13.0,
        active_bits: int = 50,
        seed: int | None = None,
    ) -> None:
        """
        Initialize FlyHash encoder.

        Args:
            input_dim: Dimension of input vectors (default: 1536 for OpenAI Ada-002).
            expansion_ratio: Output/input dimension ratio (default: 13.0).
            active_bits: Number of active bits in output (default: 50).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If active_bits >= output_dim.
        """
        self.input_dim = input_dim
        self.output_dim = int(input_dim * expansion_ratio)
        self.active_bits = active_bits
        self._seed = seed

        if self.active_bits >= self.output_dim:
            raise ValueError(
                f"active_bits ({active_bits}) must be less than output_dim ({self.output_dim})"
            )

        # Initialize random projection matrix (int8 for memory efficiency)
        rng = np.random.default_rng(seed)
        self._projection = self._build_projection_matrix(rng)

    def _build_projection_matrix(
        self, rng: np.random.Generator
    ) -> NDArray[np.int8]:
        """
        Build random binary projection matrix with int8 storage.

        Uses {-1, +1} values for better projection quality compared
        to sparse {0, 1}. Memory: ~30 MB for default config vs ~245 MB
        for float64 Gaussian.

        Args:
            rng: NumPy random generator.

        Returns:
            Binary projection matrix of shape (output_dim, input_dim).
        """
        # Generate random bits (0 or 1), then map to {-1, +1}
        bits = rng.integers(0, 2, size=(self.output_dim, self.input_dim), dtype=np.uint8)
        projection = (bits.astype(np.int8) * 2 - 1)
        return projection

    def encode(self, vector: NDArray[np.floating]) -> NDArray[np.float32]:
        """
        Encode a dense vector into a sparse binary code.

        Args:
            vector: Dense input vector of shape (input_dim,).

        Returns:
            Binary sparse vector of shape (output_dim,) with exactly
            `active_bits` elements set to 1.0.

        Raises:
            ValueError: If input vector has wrong shape.
        """
        # Validate input
        if vector.ndim != 1:
            raise ValueError(f"Expected 1-D vector, got shape {vector.shape}")
        if vector.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected vector of shape ({self.input_dim},), got {vector.shape}"
            )

        # Ensure float32 for computation
        vector = vector.astype(np.float32)

        # Project to high-dimensional space: y = P @ x
        # Cast int8 to float32 for matmul
        projection = self._projection.astype(np.float32) @ vector

        # Winner-Take-All: find indices of top-k values
        # Using argpartition for O(n) instead of O(n log n) sort
        top_k_indices = np.argpartition(projection, -self.active_bits)[
            -self.active_bits :
        ]

        # Create sparse binary output
        output = np.zeros(self.output_dim, dtype=np.float32)
        output[top_k_indices] = 1.0

        # Log at debug level - only sparsity varies per call
        logger.debug("FlyHash encoded", extra={"sparsity": 1.0 - (self.active_bits / self.output_dim)})

        return output

    def encode_batch(self, vectors: NDArray[np.floating]) -> NDArray[np.float32]:
        """
        Encode a batch of dense vectors.

        Args:
            vectors: Dense input vectors of shape (batch_size, input_dim).

        Returns:
            Binary sparse vectors of shape (batch_size, output_dim).

        Raises:
            ValueError: If input has wrong shape.
        """
        if vectors.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {vectors.shape}")
        if vectors.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected vectors of shape (batch, {self.input_dim}), "
                f"got shape {vectors.shape}"
            )

        batch_size = vectors.shape[0]
        outputs = np.zeros((batch_size, self.output_dim), dtype=np.float32)

        for i in range(batch_size):
            outputs[i] = self.encode(vectors[i])

        return outputs

    def get_sparsity(self) -> float:
        """
        Return the sparsity ratio (fraction of zeros).

        Returns:
            Sparsity as a float between 0.0 and 1.0.
        """
        return 1.0 - (self.active_bits / self.output_dim)

    def hamming_distance(
        self, hash1: NDArray[np.floating], hash2: NDArray[np.floating]
    ) -> int:
        """
        Compute Hamming distance between two FlyHash codes.

        Args:
            hash1, hash2: Binary hash vectors.

        Returns:
            Number of differing bits.
        """
        return int(np.sum(np.abs(hash1 - hash2)))

    def jaccard_similarity(
        self, hash1: NDArray[np.floating], hash2: NDArray[np.floating]
    ) -> float:
        """
        Compute Jaccard similarity between two FlyHash codes.

        Args:
            hash1, hash2: Binary hash vectors.

        Returns:
            Jaccard similarity (0.0 to 1.0).
        """
        intersection: int = int(np.sum(np.logical_and(hash1, hash2)))
        union: int = int(np.sum(np.logical_or(hash1, hash2)))
        return float(intersection / union) if union > 0 else 0.0

    @property
    def memory_bytes(self) -> int:
        """Return memory usage of projection matrix in bytes."""
        return self._projection.nbytes

    def __repr__(self) -> str:
        return (
            f"FlyHash(input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"active_bits={self.active_bits}, sparsity={self.get_sparsity():.4f}, "
            f"memory={self.memory_bytes / 1_000_000:.1f}MB)"
        )
