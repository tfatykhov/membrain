# PR-005 — FlyHash Memory Footprint Optimization

## Status: 🔴 Not Started

## Current State Analysis

### What Exists

`src/membrain/encoder.py` implements FlyHash with:
```python
class FlyHash:
    def __init__(self, input_dim: int, expansion_ratio: float = 13.0, ...):
        self.output_dim = int(input_dim * expansion_ratio)
        # Projection matrix: float64, shape (output_dim, input_dim)
        self._projection = rng.standard_normal((self.output_dim, input_dim))
```

### Problem

For typical LLM embeddings:
- `input_dim = 1536` (Ada-002) or `3072` (text-embedding-3-large)
- `expansion_ratio = 13.0`
- `output_dim = 1536 * 13 = 19,968`

**Memory calculation:**
- Projection matrix: `19,968 × 1,536 × 8 bytes = ~245 MB` (float64)
- Even float32: `~122 MB`

This is excessive for a sparse random projection that only needs sign information.

---

## Objective

Reduce encoder memory usage while keeping determinism and correctness.

---

## Detailed Requirements

### Option A: Compact Dtype (Quick Win)

Store projection as `int8` (+1/-1) instead of float64:

```python
class FlyHash:
    def __init__(self, input_dim: int, expansion_ratio: float = 13.0, ...):
        self.output_dim = int(input_dim * expansion_ratio)
        rng = np.random.default_rng(seed)
        
        # Generate random signs: +1 or -1
        random_bits = rng.integers(0, 2, size=(self.output_dim, input_dim), dtype=np.uint8)
        self._projection = (random_bits * 2 - 1).astype(np.int8)  # {-1, +1}
    
    def encode(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        # Cast to float32 for matmul, then apply WTA
        pre_activation = self._projection.astype(np.float32) @ x
        # ... WTA selection ...
```

**Memory:** `19,968 × 1,536 × 1 byte = ~30 MB` (8x reduction)

### Option B: Sparse Index Representation (Better)

Store only the indices of non-zero elements per output feature:

```python
class FlyHash:
    def __init__(self, input_dim: int, expansion_ratio: float = 13.0, 
                 connectivity: float = 0.1, ...):
        """
        Args:
            connectivity: Fraction of input dimensions each output connects to.
        """
        self.output_dim = int(input_dim * expansion_ratio)
        k = int(input_dim * connectivity)  # connections per output
        
        rng = np.random.default_rng(seed)
        
        # For each output, store k random input indices and their signs
        self._indices = np.zeros((self.output_dim, k), dtype=np.uint16)
        self._signs = np.zeros((self.output_dim, k), dtype=np.int8)
        
        for i in range(self.output_dim):
            self._indices[i] = rng.choice(input_dim, size=k, replace=False)
            self._signs[i] = rng.choice([-1, 1], size=k)
    
    def encode(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        pre_activation = np.zeros(self.output_dim, dtype=np.float32)
        for i in range(self.output_dim):
            pre_activation[i] = np.dot(self._signs[i], x[self._indices[i]])
        # ... WTA selection ...
```

**Memory with 10% connectivity:**
- Indices: `19,968 × 154 × 2 bytes = ~6 MB`
- Signs: `19,968 × 154 × 1 byte = ~3 MB`
- **Total: ~9 MB** (27x reduction)

### Option C: On-the-fly Generation (Best Memory, Slower)

Don't store projection at all — regenerate from seed during encode:

```python
class FlyHash:
    def __init__(self, input_dim: int, expansion_ratio: float = 13.0, seed: int = 42):
        self.output_dim = int(input_dim * expansion_ratio)
        self.input_dim = input_dim
        self.seed = seed
    
    def encode(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        rng = np.random.default_rng(self.seed)
        pre_activation = np.zeros(self.output_dim, dtype=np.float32)
        
        # Stream through projection rows
        for i in range(self.output_dim):
            row = rng.standard_normal(self.input_dim).astype(np.float32)
            pre_activation[i] = np.dot(row, x)
        
        # ... WTA selection ...
```

**Memory:** Near zero, but ~10x slower encode.

---

## Recommended Approach

**Implement Option A first** (int8 projection) as it's:
- Simple to implement
- Backward compatible
- 8x memory reduction
- No performance penalty

**Consider Option B later** if memory is still a concern at scale.

---

## Files / Modules

| File | Action |
|------|--------|
| `src/membrain/encoder.py` | **Update** — Use int8 projection |
| `tests/test_encoder.py` | **Update** — Memory usage test |

---

## Detailed Implementation (Option A)

```python
# encoder.py changes

class FlyHash:
    """FlyHash locality-sensitive sparse encoder."""
    
    def __init__(
        self,
        input_dim: int,
        expansion_ratio: float = 13.0,
        active_fraction: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.expansion_ratio = expansion_ratio
        self.output_dim = int(input_dim * expansion_ratio)
        self.active_bits = max(1, int(self.output_dim * active_fraction))
        
        rng = np.random.default_rng(seed)
        
        # Memory-efficient: store as int8 {-1, +1}
        random_bits = rng.integers(0, 2, size=(self.output_dim, input_dim), dtype=np.uint8)
        self._projection: NDArray[np.int8] = (random_bits * 2 - 1).astype(np.int8)
        
        self._seed = seed
    
    def encode(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Encode dense vector to sparse FlyHash representation."""
        if x.shape != (self.input_dim,):
            raise ValueError(f"Expected shape ({self.input_dim},), got {x.shape}")
        
        # Project: cast int8 to float32 for matmul
        pre_activation = self._projection.astype(np.float32) @ x.astype(np.float32)
        
        # Winner-take-all: keep top-k activations
        threshold_idx = np.argpartition(pre_activation, -self.active_bits)[-self.active_bits:]
        
        sparse = np.zeros(self.output_dim, dtype=np.float32)
        sparse[threshold_idx] = pre_activation[threshold_idx]
        
        return sparse
    
    @property
    def memory_bytes(self) -> int:
        """Return approximate memory usage in bytes."""
        return self._projection.nbytes
```

---

## Tests

### Unit Tests

```python
def test_flyhash_determinism():
    """Same seed produces identical encodings."""
    enc1 = FlyHash(input_dim=64, expansion_ratio=4.0, seed=42)
    enc2 = FlyHash(input_dim=64, expansion_ratio=4.0, seed=42)
    
    x = np.random.randn(64).astype(np.float32)
    
    assert np.allclose(enc1.encode(x), enc2.encode(x))

def test_flyhash_memory_efficient():
    """Projection uses int8, not float64."""
    enc = FlyHash(input_dim=1536, expansion_ratio=13.0, seed=42)
    
    # Should be ~30 MB, not ~245 MB
    assert enc.memory_bytes < 50_000_000  # 50 MB
    assert enc._projection.dtype == np.int8

def test_flyhash_sparsity():
    """Output has expected number of active bits."""
    enc = FlyHash(input_dim=64, expansion_ratio=4.0, active_fraction=0.05, seed=42)
    x = np.random.randn(64).astype(np.float32)
    
    sparse = enc.encode(x)
    n_active = np.count_nonzero(sparse)
    
    assert n_active == enc.active_bits
```

---

## Acceptance Criteria

- [ ] Projection stored as int8 (not float64)
- [ ] Memory usage < 50 MB for default config
- [ ] Encoding determinism preserved (seeded)
- [ ] WTA behavior unchanged
- [ ] All existing tests pass
- [ ] Memory usage test added

---

## Risks / Notes

- **Numerical differences**: int8 {-1,+1} vs float64 Gaussian will produce different sparse codes. This is a **breaking change** if any persisted state depends on the old encoding. Document as v0.2.0 breaking change.
- **Alternative**: Could keep float32 with sign() function for gradual migration.

---

## Definition of Done

- [ ] Tests added for memory usage and determinism
- [ ] README notes on memory optimization
- [ ] Determinism preserved with seed
- [ ] Breaking change documented if applicable
