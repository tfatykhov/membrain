"""VectorStore protocol — interface contract for all benchmark targets.

All baseline implementations and Membrain adapter must implement this protocol.
This ensures fair, apples-to-apples comparison across different systems.
"""

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@runtime_checkable
class VectorStore(Protocol):
    """Interface for vector storage systems under benchmark.
    
    Implementations must support:
    - Storing vectors with string keys
    - Querying for top-k nearest neighbors
    - Reporting memory usage for fair comparison
    
    All vectors are assumed to be normalized (unit length) for cosine similarity.
    Implementations should handle normalization internally if needed.
    """
    
    def store(self, key: str, vector: NDArray[np.floating]) -> None:
        """Store a vector with an associated key.
        
        Args:
            key: Unique identifier for retrieval
            vector: Dense vector (will be normalized internally)
        
        Raises:
            ValueError: If key already exists (no duplicates allowed)
        """
        ...
    
    def store_batch(self, items: list[tuple[str, NDArray[np.floating]]]) -> None:
        """Bulk insert multiple vectors.
        
        Default implementation calls store() in a loop.
        Implementations may override for efficiency.
        
        Args:
            items: List of (key, vector) tuples
        """
        ...
    
    def query(
        self, 
        vector: NDArray[np.floating], 
        k: int = 1
    ) -> list[tuple[str, float]]:
        """Find top-k nearest neighbors.
        
        Args:
            vector: Query vector (will be normalized internally)
            k: Number of results to return
            
        Returns:
            List of (key, score) tuples, sorted by descending similarity.
            Score is cosine similarity in range [-1, 1].
        """
        ...
    
    def clear(self) -> None:
        """Remove all stored vectors. Reset to empty state."""
        ...
    
    @property
    def count(self) -> int:
        """Number of vectors currently stored."""
        ...
    
    @property
    def dim(self) -> int:
        """Dimensionality of stored vectors. 0 if empty."""
        ...
    
    @property
    def memory_mb(self) -> float:
        """Approximate memory usage in megabytes.
        
        Should include:
        - Vector storage
        - Index structures
        - Key storage
        
        Used for fair comparison across implementations.
        """
        ...


def check_protocol_compliance(store: VectorStore, dim: int = 64) -> list[str]:
    """Verify a VectorStore implementation meets the protocol contract.
    
    Args:
        store: Implementation to check
        dim: Vector dimensionality to use for tests
        
    Returns:
        List of error messages (empty if compliant)
    """
    errors: list[str] = []
    rng = np.random.default_rng(42)
    
    # Check initial state
    if store.count != 0:
        errors.append(f"Expected count=0 initially, got {store.count}")
    
    # Store a vector
    vec1 = rng.standard_normal(dim).astype(np.float64)
    try:
        store.store("test_1", vec1)
    except Exception as e:
        errors.append(f"store() failed: {e}")
        return errors  # Can't continue
    
    if store.count != 1:
        errors.append(f"Expected count=1 after store, got {store.count}")
    
    if store.dim != dim:
        errors.append(f"Expected dim={dim}, got {store.dim}")
    
    # Query for it
    try:
        results = store.query(vec1, k=1)
        if not results:
            errors.append("query() returned empty results")
        elif results[0][0] != "test_1":
            errors.append(f"Expected key='test_1', got '{results[0][0]}'")
        elif not (0.99 <= results[0][1] <= 1.01):
            errors.append(f"Expected score≈1.0 for exact match, got {results[0][1]}")
    except Exception as e:
        errors.append(f"query() failed: {e}")
    
    # Check memory reporting
    if store.memory_mb <= 0:
        errors.append(f"memory_mb should be positive, got {store.memory_mb}")
    
    # Clear
    try:
        store.clear()
    except Exception as e:
        errors.append(f"clear() failed: {e}")
    
    if store.count != 0:
        errors.append(f"Expected count=0 after clear, got {store.count}")
    
    return errors
