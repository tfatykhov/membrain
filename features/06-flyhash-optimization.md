# PR-005 — FlyHash Memory Footprint Optimization

## Status: 🔴 Not Started

## Current State Analysis

### Problem
Current `encoder.py` uses float64 projection matrix:
- `input_dim = 1536`, `expansion_ratio = 13.0` → `output_dim = 19,968`
- Memory: `19,968 × 1,536 × 8 bytes = ~245 MB`

This is excessive for a sparse random projection.

---

## Objective

Reduce encoder memory usage while keeping determinism and correctness.

---

## Recommended Approach: int8 Projection

Store projection as `int8` (+1/-1) instead of float64:

```python
class FlyHash:
    def __init__(self, input_dim: int, expansion_ratio: float = 13.0, seed: int = None):
        self.output_dim = int(input_dim * expansion_ratio)
        rng = np.random.default_rng(seed)
        
        # Memory-efficient: store as int8 {-1, +1}
        random_bits = rng.integers(0, 2, size=(self.output_dim, input_dim), dtype=np.uint8)
        self._projection = (random_bits * 2 - 1).astype(np.int8)
    
    def encode(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        # Cast to float32 for matmul
        pre_activation = self._projection.astype(np.float32) @ x
        # ... WTA selection ...
```

**Memory:** `19,968 × 1,536 × 1 byte = ~30 MB` (8x reduction)

---

## Files / Modules

| File | Action |
|------|--------|
| `src/membrain/encoder.py` | **Update** — Use int8 projection |
| `tests/test_encoder.py` | **Update** — Memory usage test |

---

## Tests

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
    assert enc._projection.nbytes < 50_000_000  # 50 MB
    assert enc._projection.dtype == np.int8
```

---

## Acceptance Criteria

- [ ] Projection stored as int8 (not float64)
- [ ] Memory usage < 50 MB for default config
- [ ] Encoding determinism preserved (seeded)
- [ ] All existing tests pass

---

## Risks

**Breaking change**: int8 {-1,+1} produces different sparse codes than float64 Gaussian. Document as v0.2.0 breaking change if any persisted state exists.
