"""Cosine similarity baseline — naive exact search.

The simplest possible vector search: store vectors, compute cosine similarity
against all stored vectors on each query. No indexing, no optimization.

This is the floor — any system should beat this on latency for large N.
But it should match on accuracy (100% hit@1 for exact queries).
"""

import sys
import numpy as np
from numpy.typing import NDArray


class CosineBaseline:
    """Naive exact cosine similarity search.
    
    Stores vectors in a list, builds a matrix on first query.
    Computes dot product against all vectors (brute force).
    
    Good for:
    - Baseline accuracy (should be 100% on exact queries)
    - Small datasets (< 10K vectors)
    
    Bad for:
    - Large datasets (O(n) per query)
    - Low latency requirements
    """
    
    def __init__(self) -> None:
        self._keys: list[str] = []
        self._key_set: set[str] = set()  # O(1) duplicate check
        self._vectors: list[NDArray[np.floating]] = []
        self._matrix: NDArray[np.floating] | None = None
        self._dim: int = 0
    
    def store(self, key: str, vector: NDArray[np.floating]) -> None:
        """Store a normalized vector with a key."""
        if key in self._key_set:
            raise ValueError(f"Duplicate key: {key}")
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            raise ValueError("Cannot store zero vector")
        normalized = vector / norm
        
        self._keys.append(key)
        self._key_set.add(key)
        self._vectors.append(normalized.astype(np.float64))
        self._matrix = None  # Invalidate cache
        
        if self._dim == 0:
            self._dim = len(vector)
    
    def store_batch(self, items: list[tuple[str, NDArray[np.floating]]]) -> None:
        """Bulk insert multiple vectors."""
        for key, vector in items:
            self.store(key, vector)
    
    def query(
        self, 
        vector: NDArray[np.floating], 
        k: int = 1
    ) -> list[tuple[str, float]]:
        """Find top-k most similar vectors."""
        if not self._vectors:
            return []
        
        # Build matrix cache on first query
        if self._matrix is None:
            self._matrix = np.vstack(self._vectors)
        
        # Normalize query
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return []
        query_norm = vector / norm
        
        # Compute all cosine similarities
        scores = self._matrix @ query_norm
        
        # Get top-k indices
        if k >= len(scores):
            top_k_idx = np.argsort(scores)[::-1]
        else:
            # Partial sort for efficiency
            top_k_idx = np.argpartition(scores, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        
        return [(self._keys[i], float(scores[i])) for i in top_k_idx[:k]]
    
    def clear(self) -> None:
        """Reset to empty state."""
        self._keys.clear()
        self._key_set.clear()
        self._vectors.clear()
        self._matrix = None
        self._dim = 0
    
    @property
    def count(self) -> int:
        """Number of stored vectors."""
        return len(self._keys)
    
    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._dim
    
    @property
    def memory_mb(self) -> float:
        """Approximate memory usage in MB."""
        if not self._vectors:
            return 0.0
        
        # Vector storage
        vector_bytes = sum(v.nbytes for v in self._vectors)
        
        # Matrix cache (if built)
        matrix_bytes = self._matrix.nbytes if self._matrix is not None else 0
        
        # Key storage (rough estimate)
        key_bytes = sum(sys.getsizeof(k) for k in self._keys)
        
        # Set storage
        set_bytes = sys.getsizeof(self._key_set)
        
        total_bytes = vector_bytes + matrix_bytes + key_bytes + set_bytes
        return total_bytes / (1024 * 1024)
