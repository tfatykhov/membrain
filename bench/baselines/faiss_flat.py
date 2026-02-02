"""FAISS baseline — exact and approximate nearest neighbor search.

Uses Facebook's FAISS library for production-grade vector search.
Provides two variants:
- FAISSFlatBaseline: Exact search (IndexFlatIP) — for accuracy comparison
- FAISSIVFBaseline: Approximate search (IndexIVFFlat) — for production comparison

Requires: pip install faiss-cpu (or faiss-gpu for GPU support)
"""

from typing import TYPE_CHECKING
import sys

import numpy as np
from numpy.typing import NDArray

# Conditional import for type checking
if TYPE_CHECKING:
    import faiss


class FAISSFlatBaseline:
    """FAISS exact search using IndexFlatIP (inner product).
    
    This is a highly optimized brute-force search. Should match
    CosineBaseline accuracy but with better latency for large datasets.
    
    Uses inner product (IP) on normalized vectors, which equals cosine similarity.
    """
    
    def __init__(self, dim: int) -> None:
        """Initialize FAISS index.
        
        Args:
            dim: Vector dimensionality (must be known upfront for FAISS)
        """
        import faiss as _faiss
        self._faiss = _faiss
        
        self._dim = dim
        self._index = _faiss.IndexFlatIP(dim)
        self._keys: list[str] = []
        self._key_set: set[str] = set()
    
    def store(self, key: str, vector: NDArray[np.floating]) -> None:
        """Store a normalized vector."""
        if key in self._key_set:
            raise ValueError(f"Duplicate key: {key}")
        
        # Normalize and convert to float32 (FAISS requirement)
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            raise ValueError("Cannot store zero vector")
        normalized = (vector / norm).astype(np.float32)
        
        # FAISS expects 2D array
        self._index.add(normalized.reshape(1, -1))
        self._keys.append(key)
        self._key_set.add(key)
    
    def store_batch(self, items: list[tuple[str, NDArray[np.floating]]]) -> None:
        """Bulk insert — more efficient for FAISS."""
        if not items:
            return
        
        # Check for duplicates
        new_keys = [k for k, _ in items]
        for key in new_keys:
            if key in self._key_set:
                raise ValueError(f"Duplicate key: {key}")
        
        # Normalize all vectors
        vectors = np.array([v for _, v in items], dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = (vectors / norms).astype(np.float32)
        
        # Bulk add
        self._index.add(normalized)
        self._keys.extend(new_keys)
        self._key_set.update(new_keys)
    
    def query(
        self, 
        vector: NDArray[np.floating], 
        k: int = 1
    ) -> list[tuple[str, float]]:
        """Find top-k nearest neighbors."""
        if self._index.ntotal == 0:
            return []
        
        # Normalize query
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return []
        query_norm = (vector / norm).astype(np.float32).reshape(1, -1)
        
        # Search
        actual_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_norm, actual_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((self._keys[idx], float(scores[0][i])))
        
        return results
    
    def clear(self) -> None:
        """Reset index."""
        self._index.reset()
        self._keys.clear()
        self._key_set.clear()
    
    @property
    def count(self) -> int:
        """Number of stored vectors."""
        return self._index.ntotal
    
    @property
    def dim(self) -> int:
        """Vector dimensionality."""
        return self._dim
    
    @property
    def memory_mb(self) -> float:
        """Approximate memory usage."""
        # FAISS IndexFlatIP: 4 bytes per float32 × dim × count
        index_bytes = 4 * self._dim * self._index.ntotal
        key_bytes = sum(sys.getsizeof(k) for k in self._keys)
        set_bytes = sys.getsizeof(self._key_set)
        return (index_bytes + key_bytes + set_bytes) / (1024 * 1024)


class FAISSIVFBaseline:
    """FAISS approximate search using IVF (Inverted File Index).
    
    Faster than exact search for large datasets, but may miss some results.
    Use nprobe to trade speed vs accuracy.
    
    Requires training before use — call train() with representative vectors.
    """
    
    def __init__(
        self, 
        dim: int, 
        nlist: int = 100,
        nprobe: int = 10
    ) -> None:
        """Initialize IVF index.
        
        Args:
            dim: Vector dimensionality
            nlist: Number of clusters (more = better accuracy, slower build)
            nprobe: Number of clusters to search (more = better accuracy, slower query)
        """
        import faiss as _faiss
        self._faiss = _faiss
        
        self._dim = dim
        self._nlist = nlist
        self._nprobe = nprobe
        
        # IVF requires a quantizer
        quantizer = _faiss.IndexFlatIP(dim)
        self._index = _faiss.IndexIVFFlat(quantizer, dim, nlist)
        self._index.nprobe = nprobe
        
        self._keys: list[str] = []
        self._key_set: set[str] = set()
        self._is_trained = False
    
    def train(self, vectors: NDArray[np.floating]) -> None:
        """Train the IVF index on representative vectors.
        
        Should be called with a sample of vectors before adding data.
        Typically need 30-50× nlist vectors for good training.
        
        Args:
            vectors: Training vectors, shape (n, dim)
        """
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = (vectors / norms).astype(np.float32)
        
        self._index.train(normalized)
        self._is_trained = True
    
    def store(self, key: str, vector: NDArray[np.floating]) -> None:
        """Store a normalized vector."""
        if not self._is_trained:
            raise RuntimeError("IVF index must be trained before adding vectors")
        
        if key in self._key_set:
            raise ValueError(f"Duplicate key: {key}")
        
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            raise ValueError("Cannot store zero vector")
        normalized = (vector / norm).astype(np.float32)
        
        self._index.add(normalized.reshape(1, -1))
        self._keys.append(key)
        self._key_set.add(key)
    
    def store_batch(self, items: list[tuple[str, NDArray[np.floating]]]) -> None:
        """Bulk insert."""
        if not self._is_trained:
            raise RuntimeError("IVF index must be trained before adding vectors")
        
        if not items:
            return
        
        new_keys = [k for k, _ in items]
        for key in new_keys:
            if key in self._key_set:
                raise ValueError(f"Duplicate key: {key}")
        
        vectors = np.array([v for _, v in items], dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = (vectors / norms).astype(np.float32)
        
        self._index.add(normalized)
        self._keys.extend(new_keys)
        self._key_set.update(new_keys)
    
    def query(
        self, 
        vector: NDArray[np.floating], 
        k: int = 1
    ) -> list[tuple[str, float]]:
        """Find approximate top-k nearest neighbors."""
        if self._index.ntotal == 0:
            return []
        
        norm = np.linalg.norm(vector)
        if norm < 1e-10:
            return []
        query_norm = (vector / norm).astype(np.float32).reshape(1, -1)
        
        actual_k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_norm, actual_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                results.append((self._keys[idx], float(scores[0][i])))
        
        return results
    
    def clear(self) -> None:
        """Reset index (keeps training)."""
        self._index.reset()
        self._keys.clear()
        self._key_set.clear()
        # Note: index remains trained
    
    @property
    def count(self) -> int:
        return self._index.ntotal
    
    @property
    def dim(self) -> int:
        return self._dim
    
    @property
    def memory_mb(self) -> float:
        # IVF uses more memory due to cluster info
        index_bytes = 4 * self._dim * self._index.ntotal
        overhead = self._nlist * self._dim * 4  # Centroids
        key_bytes = sum(sys.getsizeof(k) for k in self._keys)
        set_bytes = sys.getsizeof(self._key_set)
        return (index_bytes + overhead + key_bytes + set_bytes) / (1024 * 1024)
    
    @property
    def is_trained(self) -> bool:
        """Whether the index has been trained."""
        return self._is_trained
