# Feature 02: FlyHash Encoder

**Status:** Not Started  
**Priority:** P0 - Critical Path  
**Target File:** `src/membrain/encoder.py`  
**Depends On:** None  
**Required By:** Feature 03 (Neuromorphic Core)

---

## Objective

Convert dense LLM embeddings (1536-d floats) into sparse binary spike trains (20,000-d binary) suitable for efficient SNN processing. This encoding is critical for energy-efficient neuromorphic computing.

---

## Algorithm: FlyHash

FlyHash is inspired by the fruit fly olfactory system. It uses random projections followed by Winner-Take-All (WTA) inhibition to create sparse, locality-sensitive hash codes.

### Mathematical Definition

Given:

- Input vector `x` ∈ ℝ^d (d = 1536)
- Random projection matrix `M` ∈ {0,1}^(d × m) (m = d × r, typically r ≈ 13)
- Sparsity parameter `k` (number of active bits, typically k = 50)

Algorithm:

1. **Project:** `y = M^T · x` → y ∈ ℝ^m
2. **Inhibit (WTA):** Find indices of top-k values in y
3. **Binarize:** Output binary vector h ∈ {0,1}^m where h[i] = 1 iff i ∈ top-k

### Key Properties

| Property | Value | Significance |
|----------|-------|--------------|
| Input Dimension | 1536 | OpenAI Ada-002 embeddings |
| Output Dimension | ~20,000 | Expanded sparse space |
| Active Bits (k) | 50 | Fixed sparsity = k/m ≈ 0.25% |
| Sparsity Rate | >99.7% | Enables efficient SNN |

---

## Implementation Specification

### Class: FlyHash

```python
# src/membrain/encoder.py

import numpy as np
from typing import Optional

class FlyHash:
    """
    Locality-Sensitive Hashing using the FlyHash algorithm.
    
    Converts dense embedding vectors into sparse binary codes
    suitable for Spiking Neural Network processing.
    """
    
    def __init__(
        self,
        input_dim: int = 1536,
        expansion_ratio: float = 13.0,
        active_bits: int = 50,
        seed: Optional[int] = None
    ):
        """
        Initialize FlyHash encoder.
        
        Args:
            input_dim: Dimension of input vectors (default: 1536 for OpenAI)
            expansion_ratio: Output/input dimension ratio (default: 13.0)
            active_bits: Number of active bits in output (default: 50)
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.output_dim = int(input_dim * expansion_ratio)
        self.active_bits = active_bits
        
        # Initialize random projection matrix
        rng = np.random.default_rng(seed)
        # Binary random projection (sparse connections)
        self.projection_matrix = rng.choice(
            [0, 1], 
            size=(input_dim, self.output_dim),
            p=[0.9, 0.1]  # 10% connection probability
        ).astype(np.float32)
    
    def encode(self, vector: np.ndarray) -> np.ndarray:
        """
        Encode a dense vector into a sparse binary code.
        
        Args:
            vector: Dense input vector of shape (input_dim,)
            
        Returns:
            Binary sparse vector of shape (output_dim,)
        """
        # Validate input
        if vector.shape != (self.input_dim,):
            raise ValueError(f"Expected vector of shape ({self.input_dim},), got {vector.shape}")
        
        # Project to high-dimensional space
        projection = self.projection_matrix.T @ vector
        
        # Winner-Take-All: keep top-k values
        top_k_indices = np.argpartition(projection, -self.active_bits)[-self.active_bits:]
        
        # Create sparse binary output
        output = np.zeros(self.output_dim, dtype=np.float32)
        output[top_k_indices] = 1.0
        
        return output
    
    def encode_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode a batch of dense vectors.
        
        Args:
            vectors: Dense input vectors of shape (batch_size, input_dim)
            
        Returns:
            Binary sparse vectors of shape (batch_size, output_dim)
        """
        batch_size = vectors.shape[0]
        outputs = np.zeros((batch_size, self.output_dim), dtype=np.float32)
        
        for i, vector in enumerate(vectors):
            outputs[i] = self.encode(vector)
        
        return outputs
    
    def get_sparsity(self) -> float:
        """Return the sparsity ratio (fraction of zeros)."""
        return 1.0 - (self.active_bits / self.output_dim)
    
    def similarity(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """
        Compute similarity between two FlyHash codes using Jaccard index.
        
        Args:
            hash1, hash2: Binary hash vectors
            
        Returns:
            Jaccard similarity (0.0 to 1.0)
        """
        intersection = np.sum(np.logical_and(hash1, hash2))
        union = np.sum(np.logical_or(hash1, hash2))
        return float(intersection / union) if union > 0 else 0.0
```

---

## Configuration Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `input_dim` | 1536 | 128-4096 | Input embedding dimension |
| `expansion_ratio` | 13.0 | 5.0-20.0 | Output/input dimension ratio |
| `active_bits` | 50 | 10-200 | Number of active bits (k) |
| `seed` | None | int | Random seed for reproducibility |

---

## Acceptance Criteria

- [ ] `FlyHash` class initializes with configurable parameters
- [ ] `encode()` produces output of correct dimension
- [ ] Output sparsity is >90% (configurable via `active_bits`)
- [ ] Exactly `k` bits are set to 1 in output
- [ ] Similar input vectors produce similar hash codes
- [ ] Dissimilar input vectors produce different hash codes
- [ ] `encode_batch()` handles multiple vectors efficiently
- [ ] Reproducible with `seed` parameter

---

## Testing

### Unit Tests (`tests/test_flyhash.py`)

```python
import numpy as np
import pytest
from membrain.encoder import FlyHash

class TestFlyHash:
    @pytest.fixture
    def encoder(self):
        return FlyHash(input_dim=1536, seed=42)
    
    def test_output_dimension(self, encoder):
        """Output should have correct expanded dimension."""
        vector = np.random.randn(1536).astype(np.float32)
        output = encoder.encode(vector)
        assert output.shape == (encoder.output_dim,)
    
    def test_output_sparsity(self, encoder):
        """Output should be >90% sparse."""
        vector = np.random.randn(1536).astype(np.float32)
        output = encoder.encode(vector)
        sparsity = 1.0 - (np.sum(output) / len(output))
        assert sparsity > 0.90
    
    def test_exact_k_active_bits(self, encoder):
        """Exactly k bits should be active."""
        vector = np.random.randn(1536).astype(np.float32)
        output = encoder.encode(vector)
        assert np.sum(output) == encoder.active_bits
    
    def test_similar_vectors_similar_hashes(self, encoder):
        """Similar vectors should produce similar hash codes."""
        v1 = np.random.randn(1536).astype(np.float32)
        v2 = v1 + 0.1 * np.random.randn(1536).astype(np.float32)  # Small perturbation
        
        h1 = encoder.encode(v1)
        h2 = encoder.encode(v2)
        
        similarity = encoder.similarity(h1, h2)
        assert similarity > 0.3  # Expect significant overlap
    
    def test_dissimilar_vectors_different_hashes(self, encoder):
        """Dissimilar vectors should produce different hash codes."""
        v1 = np.random.randn(1536).astype(np.float32)
        v2 = np.random.randn(1536).astype(np.float32)  # Unrelated vector
        
        h1 = encoder.encode(v1)
        h2 = encoder.encode(v2)
        
        similarity = encoder.similarity(h1, h2)
        assert similarity < 0.3  # Expect low overlap
    
    def test_reproducibility_with_seed(self):
        """Same seed should produce same projection matrix."""
        enc1 = FlyHash(seed=42)
        enc2 = FlyHash(seed=42)
        
        vector = np.random.randn(1536).astype(np.float32)
        h1 = enc1.encode(vector)
        h2 = enc2.encode(vector)
        
        assert np.array_equal(h1, h2)
    
    def test_invalid_input_dimension(self, encoder):
        """Should raise error for wrong input dimension."""
        vector = np.random.randn(512).astype(np.float32)
        with pytest.raises(ValueError):
            encoder.encode(vector)
    
    def test_batch_encoding(self, encoder):
        """Batch encoding should work correctly."""
        vectors = np.random.randn(10, 1536).astype(np.float32)
        outputs = encoder.encode_batch(vectors)
        
        assert outputs.shape == (10, encoder.output_dim)
        for output in outputs:
            assert np.sum(output) == encoder.active_bits
```

### Semantic Similarity Test

```python
def test_semantic_preservation():
    """
    FlyHash should preserve semantic relationships.
    
    Test case from PRD:
    - "King" and "Queen" embeddings should have lower Hamming distance
    - "King" and "Apple" embeddings should have higher Hamming distance
    """
    encoder = FlyHash(seed=42)
    
    # Simulate embeddings (in real use, get from OpenAI)
    np.random.seed(100)
    king = np.random.randn(1536).astype(np.float32)
    queen = king + 0.2 * np.random.randn(1536)  # Similar
    apple = np.random.randn(1536).astype(np.float32)  # Different
    
    h_king = encoder.encode(king)
    h_queen = encoder.encode(queen)
    h_apple = encoder.encode(apple)
    
    # Hamming distance (number of differing bits)
    dist_king_queen = np.sum(np.abs(h_king - h_queen))
    dist_king_apple = np.sum(np.abs(h_king - h_apple))
    
    assert dist_king_queen < dist_king_apple
```

---

## Performance Considerations

- **Projection Matrix Storage:** ~100MB for 1536×20000 float32 matrix
- **Encoding Latency:** O(d × m) matrix multiplication
- **Optimization:** Use sparse matrix representation for projection matrix

---

## References

- [FlyHash Paper (arXiv:1711.03127)](https://arxiv.org/abs/1711.03127) - "A neural algorithm for a fundamental computing problem"
- [Fruit Fly Olfactory System](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5731833/) - Biological inspiration
