# PR-011 — Attractor Dynamics (True Pattern Completion)

## Status: 🔴 Not Started — P1 Priority

## Problem

Current recall is similarity lookup, not true pattern completion. The SNN doesn't demonstrate attractor dynamics or cleanup behavior.

---

## Objective

Make pattern completion depend on actual network dynamics, demonstrating measurable improvement from the SNN.

---

## Approach: Attractor Network

Add `src/membrain/attractor.py`:

```python
class AttractorMemory:
    """
    Attractor network for pattern cleanup.
    
    Uses auto-associative memory principles:
    - Recurrent connections encode stored patterns
    - Lateral inhibition provides competition
    - Dynamics converge to nearest attractor
    """
    
    def store(self, pattern: NDArray[np.float32]) -> None:
        """Store a pattern as an attractor."""
        # Hebbian update to recurrent weights
        pass
    
    def complete(self, partial: NDArray[np.float32]) -> NDArray[np.float32]:
        """Complete a partial/noisy pattern using attractor dynamics."""
        # Run network for settling time
        # Return cleaned up pattern
        pass
    
    def measure_cleanup(self, original, noisy) -> dict:
        """Measure cleanup effectiveness."""
        return {
            "input_similarity": ...,
            "output_similarity": ...,
            "improvement": ...,
        }
```

---

## Integration with BiCameralMemory

```python
class BiCameralMemory:
    def __init__(self, ..., use_attractor: bool = True):
        if use_attractor:
            self.attractor = AttractorMemory(dimensions=dimensions)
    
    def recall(self, query_vector, ...):
        if self.use_attractor:
            cleaned_query = self.attractor.complete(query_vector)
            # Use cleaned query for matching
```

---

## Key Test

```python
def test_cleanup_improves_similarity():
    """Pattern completion should improve similarity to target."""
    attractor = AttractorMemory(dimensions=128)
    
    # Store patterns
    patterns = [np.random.randn(128) for _ in range(10)]
    for p in patterns:
        attractor.store(p / np.linalg.norm(p))
    
    # Test on noisy version
    original = patterns[0] / np.linalg.norm(patterns[0])
    noisy = original + 0.3 * np.random.randn(128)
    
    result = attractor.measure_cleanup(original, noisy)
    assert result["improvement"] > 0  # Cleanup helps!
```

---

## Files / Modules

| File | Action |
|------|--------|
| `src/membrain/attractor.py` | **Create** |
| `src/membrain/core.py` | **Update** |
| `tests/test_attractor.py` | **Create** |
| `bench/bench_cleanup.py` | **Create** |

---

## Acceptance Criteria

- [ ] Demonstrable pattern completion improvement
- [ ] Recall remains read-only (learning gate enforced)
- [ ] Benchmark shows improvement over no-attractor baseline

---

## Why This Matters

This is the **key differentiator** for "synthetic hippocampus" claim. Without demonstrable attractor dynamics, Membrain is just similarity search with extra steps.
