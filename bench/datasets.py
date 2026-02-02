"""Synthetic and real datasets for benchmarking.

Provides deterministic, reproducible data generation with configurable noise.
All vectors are normalized to unit length for cosine similarity.
"""

from dataclasses import dataclass
from typing import Iterator
import numpy as np
from numpy.typing import NDArray


@dataclass
class SyntheticDataset:
    """A collection of vectors with keys for benchmarking.
    
    Vectors are normalized to unit length.
    Keys are formatted as 'vec_00000', 'vec_00001', etc.
    
    Attributes:
        vectors: Array of shape (n, dim), normalized rows
        keys: List of string keys
        seed: RNG seed used for reproducibility
    """
    
    vectors: NDArray[np.floating]
    keys: list[str]
    seed: int
    
    @classmethod
    def gaussian(
        cls, 
        n: int, 
        dim: int, 
        seed: int = 42
    ) -> "SyntheticDataset":
        """Generate n random Gaussian vectors of given dimension.
        
        Vectors are sampled from N(0, 1) and normalized to unit length.
        Deterministic given the seed.
        
        Args:
            n: Number of vectors
            dim: Vector dimensionality
            seed: Random seed for reproducibility
            
        Returns:
            SyntheticDataset with n normalized vectors
        """
        rng = np.random.default_rng(seed)
        vectors = rng.standard_normal((n, dim)).astype(np.float64)
        # Normalize to unit length
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        keys = [f"vec_{i:05d}" for i in range(n)]
        return cls(vectors=vectors, keys=keys, seed=seed)
    
    @classmethod
    def clustered(
        cls,
        n: int,
        dim: int,
        n_clusters: int = 10,
        cluster_std: float = 0.1,
        seed: int = 42
    ) -> "SyntheticDataset":
        """Generate vectors clustered around random centroids.
        
        More realistic than pure Gaussian — simulates semantic clusters.
        
        Args:
            n: Number of vectors
            dim: Vector dimensionality
            n_clusters: Number of cluster centers
            cluster_std: Standard deviation within clusters
            seed: Random seed for reproducibility
            
        Returns:
            SyntheticDataset with clustered vectors
        """
        rng = np.random.default_rng(seed)
        
        # Generate cluster centers
        centers = rng.standard_normal((n_clusters, dim))
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        
        # Assign each vector to a random cluster
        cluster_ids = rng.integers(0, n_clusters, size=n)
        
        # Generate vectors around centers
        vectors = np.zeros((n, dim), dtype=np.float64)
        for i in range(n):
            center = centers[cluster_ids[i]]
            noise = rng.normal(0, cluster_std, dim)
            vectors[i] = center + noise
        
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
        
        keys = [f"vec_{i:05d}" for i in range(n)]
        return cls(vectors=vectors, keys=keys, seed=seed)
    
    def __len__(self) -> int:
        return len(self.keys)
    
    def __iter__(self) -> Iterator[tuple[str, NDArray[np.floating]]]:
        """Iterate over (key, vector) pairs."""
        for key, vec in zip(self.keys, self.vectors):
            yield key, vec
    
    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self.vectors.shape[1]
    
    def subset(self, n: int) -> "SyntheticDataset":
        """Return a subset with the first n vectors."""
        return SyntheticDataset(
            vectors=self.vectors[:n],
            keys=self.keys[:n],
            seed=self.seed,
        )


def add_noise(
    vector: NDArray[np.floating],
    level: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Add Gaussian noise to a vector and re-normalize.
    
    The noise level is relative to the vector's magnitude.
    Result is always normalized to unit length.
    
    Args:
        vector: Input vector (assumed unit length)
        level: Noise standard deviation (0.0 = no noise, 0.3 = heavy noise)
        rng: NumPy random generator for reproducibility
        
    Returns:
        Noisy vector, normalized to unit length
        
    Example:
        >>> rng = np.random.default_rng(42)
        >>> v = np.array([1.0, 0.0, 0.0])
        >>> noisy = add_noise(v, 0.1, rng)
        >>> np.linalg.norm(noisy)  # Always 1.0
        1.0
    """
    if level <= 0:
        return vector.copy()
    
    noise = rng.normal(0, level, vector.shape).astype(vector.dtype)
    noisy = vector + noise
    norm = np.linalg.norm(noisy)
    
    if norm < 1e-10:
        # Extremely unlikely: noise cancelled the vector
        return vector.copy()
    
    return noisy / norm


def batch_add_noise(
    vectors: NDArray[np.floating],
    level: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Add noise to multiple vectors efficiently.
    
    Args:
        vectors: Array of shape (n, dim), each row unit length
        level: Noise standard deviation
        rng: NumPy random generator
        
    Returns:
        Array of noisy vectors, each row normalized
    """
    if level <= 0:
        return vectors.copy()
    
    noise = rng.normal(0, level, vectors.shape).astype(vectors.dtype)
    noisy = vectors + noise
    norms = np.linalg.norm(noisy, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)  # Avoid division by zero
    return noisy / norms
