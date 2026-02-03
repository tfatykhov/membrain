# PR-015 — Noise-Robust Recall via Attractor Denoising

## Status: 🟡 In Progress — P0 Priority

## Problem

Current recall is pure vector similarity — the SNN runs during `remember()` but doesn't participate in `recall()`. Benchmarks show Membrain performs **worse** than cosine/FAISS baselines at higher noise levels (0.2-0.3), while being ~5000x slower.

**Benchmark results (PR #39 baseline):**

| Method | 0.00 | 0.10 | 0.20 | 0.30 |
|--------|------|------|------|------|
| CosineBaseline | 1.00 | 1.00 | 0.55 | 0.45 |
| FAISSFlatBaseline | 1.00 | 1.00 | 0.55 | 0.45 |
| MembrainStore | 1.00 | 0.95 | 0.35 | 0.25 |

We're paying SNN latency without getting noise robustness benefits.

---

## Objective

Make Membrain **exceed** baseline methods at noisy recall by leveraging attractor dynamics for query denoising.

**Target:**

| Method | 0.00 | 0.10 | 0.20 | 0.30 |
|--------|------|------|------|------|
| MembrainStore | 1.00 | 1.00 | 0.70+ | 0.55+ |

---

## Root Cause Analysis

1. **Attractor not used in recall path** — Attractor runs during consolidation, but queries bypass it entirely
2. **Insufficient training density** — Voja learning with few patterns doesn't create selective neuron responses
3. **Neuron-space comparison abandoned** — PR #39 fell back to vector comparison due to (2)

---

## Phased Approach

### Phase 1: Attractor Query Denoising (This PR)

Run incoming queries through the attractor network to denoise before matching.

```python
def recall(self, query_text: str, ...) -> list[RecallResult]:
    query_vector = self._encode(query_text)
    
    # NEW: Denoise query via attractor dynamics
    if self.use_attractor and len(self._memories) > 0:
        cleaned_vector = self._denoise_query(query_vector)
    else:
        cleaned_vector = query_vector
    
    # Compare cleaned vector to stored patterns
    return self._similarity_search(cleaned_vector, top_k)
```

**Implementation:**

```python
def _denoise_query(self, query: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Run query through attractor for N steps.
    Similar inputs should converge to same basin.
    """
    # Build temporary probe network with query input
    # Run for attractor_max_steps
    # Return settled state
    ...
```

**Expected improvement:** +10-20% Hit@1 at noise levels 0.2-0.3

### Phase 2: Training Density

Pre-seed the network with synthetic patterns during initialization.

```python
def __init__(self, ..., preseed_patterns: int = 100):
    if preseed_patterns > 0:
        self._preseed_network(preseed_patterns)

def _preseed_network(self, n: int):
    """Generate synthetic patterns to establish attractor basins."""
    for _ in range(n):
        synthetic = self._generate_synthetic_pattern()
        self._train_voja(synthetic)
```

**Goal:** Give Voja enough training to form meaningful encoder selectivity.

### Phase 3: Neuron-Space Comparison (Revisit)

Once Phase 1+2 establish attractor basins, revisit comparing neuron spike patterns instead of vectors.

```python
def recall(self, ...):
    # Run query through SNN
    query_response = self._get_neuron_response(cleaned_vector)
    
    # Compare to stored neuron responses (not vectors)
    for entry in self._memories:
        similarity = cosine_similarity(query_response, entry.neuron_response)
        ...
```

**Expected improvement:** Match or exceed all baselines at all noise levels.

---

## Files / Modules

| File | Action | Phase |
|------|--------|-------|
| `src/membrain/core.py` | **Update** — add `_denoise_query()` | 1 |
| `tests/test_denoising.py` | **Create** — test cleanup improves recall | 1 |
| `bench/bench_noise.py` | **Update** — add denoising metrics | 1 |
| `src/membrain/core.py` | **Update** — add `_preseed_network()` | 2 |
| `src/membrain/core.py` | **Update** — neuron-space comparison | 3 |

---

## Key Tests

### Phase 1 Tests

```python
def test_denoising_improves_recall():
    """Attractor cleanup should improve Hit@1 on noisy queries."""
    mem = BiCameralMemory(use_attractor=True)
    
    # Store patterns
    for i in range(20):
        mem.remember(f"pattern_{i}", {"id": i})
    
    # Noisy recall
    noisy_query = add_noise("pattern_5", level=0.2)
    
    # With denoising
    result_denoised = mem.recall(noisy_query, top_k=1)
    
    # Without denoising (bypass)
    mem.use_attractor = False
    result_raw = mem.recall(noisy_query, top_k=1)
    
    assert result_denoised[0].metadata["id"] == 5
    # Denoised should be more confident
    assert result_denoised[0].similarity > result_raw[0].similarity
```

### Benchmark Validation

```bash
# Before (baseline)
python -m bench.bench_noise --dataset clustered -n 100 -v

# After Phase 1
python -m bench.bench_noise --dataset clustered -n 100 -v --denoising

# Target: Membrain matches or exceeds FAISS at 0.2+ noise
```

---

## Invariants

1. **Recall remains read-only** — No Voja learning during recall (gate disabled)
2. **Denoising is optional** — `use_attractor=False` bypasses denoising
3. **Determinism preserved** — Same seed + query = same result
4. **Latency budget** — Denoising adds ≤50ms to recall latency

---

## Acceptance Criteria

### Phase 1
- [ ] `_denoise_query()` implemented in core.py
- [ ] Unit tests for denoising improvement
- [ ] Benchmark shows improvement at 0.2-0.3 noise
- [ ] Latency remains acceptable (<200ms per recall)

### Phase 2
- [ ] Pre-seeding option available
- [ ] Benchmark with 500+ patterns shows improved baselines

### Phase 3
- [ ] Neuron-space comparison re-enabled
- [ ] Membrain exceeds FAISS at all noise levels

---

## Dependencies

- PR #39 (merged) — neuron_response storage, vector comparison baseline
- Feature 07 (done) — Stochastic consolidation
- Feature 11 — Attractor dynamics (this feature implements the recall-side of 11)

---

## Why This Matters

This is the **core value proposition** of Membrain. Without noise-robust recall that exceeds baselines, we're just slow vector search. Phase 1 is the quick win; Phase 3 is the vision.

---

## References

- Minsky, *Society of Mind* — K-lines and attractor memory
- Hopfield networks — Auto-associative memory, basin of attraction
- PR #39 analysis — Why neuron-space comparison failed initially
