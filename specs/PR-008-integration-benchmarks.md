# PR-008 — Integration Tests + Benchmark Harness (with Baselines)

## Status: 🔴 Not Started

## Current State Analysis

### What Exists
- Unit tests for individual components
- Basic round-trip test (remember → recall)
- No noise robustness tests
- No baseline comparisons
- No benchmark harness

### What's Missing
1. **Noise robustness tests** — How well does recall work with corrupted queries?
2. **Baseline comparisons** — Is SNN better than simple cosine/Jaccard?
3. **Benchmark harness** — Reproducible performance measurements
4. **Hit@k metrics** — Standard retrieval metrics

---

## Objective

Provide measurable evidence (noise robustness, hit@k) and a baseline comparator to demonstrate Membrain's value.

---

## Detailed Requirements

### A. Integration Test Suite

Create `tests/integration/test_noise_robustness.py`:

```python
"""Integration tests for noise robustness."""

import numpy as np
import pytest
from membrain.encoder import FlyHash
from membrain.core import BiCameralMemory

class TestNoiseRobustness:
    """Test recall accuracy under various noise levels."""
    
    @pytest.fixture
    def memory_system(self):
        """Create a seeded memory system."""
        encoder = FlyHash(input_dim=128, expansion_ratio=8.0, seed=42)
        memory = BiCameralMemory(n_neurons=200, dimensions=encoder.output_dim)
        memory._ensure_simulator()
        return encoder, memory
    
    @pytest.fixture
    def stored_memories(self, memory_system):
        """Store N random memories."""
        encoder, memory = memory_system
        n_items = 100
        
        vectors = {}
        rng = np.random.default_rng(42)
        
        for i in range(n_items):
            vec = rng.standard_normal(128).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize
            
            sparse = encoder.encode(vec)
            memory.remember(f"item-{i:03d}", sparse, importance=1.0)
            vectors[f"item-{i:03d}"] = vec
        
        return encoder, memory, vectors
    
    @pytest.mark.parametrize("noise_level", [0.0, 0.05, 0.10, 0.20, 0.30])
    def test_hit_at_1(self, stored_memories, noise_level):
        """Test hit@1 accuracy at various noise levels."""
        encoder, memory, vectors = stored_memories
        rng = np.random.default_rng(123)
        
        hits = 0
        total = len(vectors)
        
        for context_id, original_vec in vectors.items():
            # Add noise
            noise = rng.standard_normal(128).astype(np.float32)
            noisy_vec = original_vec + noise_level * noise
            noisy_vec = noisy_vec / np.linalg.norm(noisy_vec)
            
            # Query
            sparse_query = encoder.encode(noisy_vec)
            results = memory.recall(sparse_query, threshold=0.1, max_results=1)
            
            if results and results[0].context_id == context_id:
                hits += 1
        
        accuracy = hits / total
        
        # Expected: accuracy degrades with noise
        if noise_level == 0.0:
            assert accuracy >= 0.95, f"Perfect recall should be >=95%, got {accuracy:.2%}"
        elif noise_level <= 0.10:
            assert accuracy >= 0.80, f"Low noise should be >=80%, got {accuracy:.2%}"
        elif noise_level <= 0.20:
            assert accuracy >= 0.50, f"Medium noise should be >=50%, got {accuracy:.2%}"
        # High noise (0.30) - no assertion, just measure
```

### B. Baseline Comparators

Create `bench/baselines.py`:

```python
"""Baseline retrieval methods for comparison."""

import numpy as np
from numpy.typing import NDArray
from typing import Protocol

class RetrievalMethod(Protocol):
    """Protocol for retrieval methods."""
    def store(self, context_id: str, vector: NDArray[np.float32]) -> None: ...
    def query(self, vector: NDArray[np.float32], k: int) -> list[tuple[str, float]]: ...

class CosineBaseline:
    """Simple cosine similarity baseline."""
    
    def __init__(self):
        self.vectors: dict[str, NDArray[np.float32]] = {}
    
    def store(self, context_id: str, vector: NDArray[np.float32]) -> None:
        self.vectors[context_id] = vector / np.linalg.norm(vector)
    
    def query(self, vector: NDArray[np.float32], k: int = 5) -> list[tuple[str, float]]:
        query_norm = vector / np.linalg.norm(vector)
        
        scores = []
        for ctx_id, stored in self.vectors.items():
            similarity = float(np.dot(query_norm, stored))
            scores.append((ctx_id, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

class FlyHashJaccardBaseline:
    """FlyHash encoding + Jaccard similarity (no SNN)."""
    
    def __init__(self, encoder):
        self.encoder = encoder
        self.sparse_vectors: dict[str, NDArray[np.float32]] = {}
    
    def store(self, context_id: str, vector: NDArray[np.float32]) -> None:
        sparse = self.encoder.encode(vector)
        self.sparse_vectors[context_id] = sparse
    
    def query(self, vector: NDArray[np.float32], k: int = 5) -> list[tuple[str, float]]:
        query_sparse = self.encoder.encode(vector)
        query_nonzero = set(np.nonzero(query_sparse)[0])
        
        scores = []
        for ctx_id, stored in self.sparse_vectors.items():
            stored_nonzero = set(np.nonzero(stored)[0])
            
            intersection = len(query_nonzero & stored_nonzero)
            union = len(query_nonzero | stored_nonzero)
            jaccard = intersection / union if union > 0 else 0.0
            
            scores.append((ctx_id, jaccard))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
```

### C. Benchmark Harness

Create `bench/bench_noise.py`:

```python
#!/usr/bin/env python3
"""
Benchmark noise robustness across methods.

Usage:
    python bench/bench_noise.py --output results.csv
"""

import argparse
import csv
import time
import numpy as np
from typing import Callable

from membrain.encoder import FlyHash
from membrain.core import BiCameralMemory
from bench.baselines import CosineBaseline, FlyHashJaccardBaseline


def run_benchmark(
    n_items: int = 500,
    input_dim: int = 128,
    noise_levels: list[float] = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    seed: int = 42,
) -> list[dict]:
    """Run benchmark across all methods and noise levels."""
    
    results = []
    rng = np.random.default_rng(seed)
    
    # Generate test vectors
    vectors = {}
    for i in range(n_items):
        vec = rng.standard_normal(input_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vectors[f"item-{i:04d}"] = vec
    
    # Initialize methods
    encoder = FlyHash(input_dim=input_dim, expansion_ratio=8.0, seed=seed)
    
    methods = {
        "cosine": CosineBaseline(),
        "flyhash_jaccard": FlyHashJaccardBaseline(encoder),
        "membrain_snn": create_membrain_method(encoder),
    }
    
    # Store in all methods
    for ctx_id, vec in vectors.items():
        for method in methods.values():
            method.store(ctx_id, vec)
    
    # Benchmark each noise level
    for noise_level in noise_levels:
        for method_name, method in methods.items():
            hits_at_1 = 0
            hits_at_5 = 0
            total_time = 0.0
            
            for ctx_id, original_vec in vectors.items():
                # Add noise
                noise = rng.standard_normal(input_dim).astype(np.float32)
                noisy_vec = original_vec + noise_level * noise
                noisy_vec = noisy_vec / np.linalg.norm(noisy_vec)
                
                # Query
                start = time.perf_counter()
                retrieved = method.query(noisy_vec, k=5)
                total_time += time.perf_counter() - start
                
                # Check hits
                retrieved_ids = [r[0] for r in retrieved]
                if retrieved_ids and retrieved_ids[0] == ctx_id:
                    hits_at_1 += 1
                if ctx_id in retrieved_ids:
                    hits_at_5 += 1
            
            results.append({
                "method": method_name,
                "noise_level": noise_level,
                "hit_at_1": hits_at_1 / n_items,
                "hit_at_5": hits_at_5 / n_items,
                "avg_latency_ms": (total_time / n_items) * 1000,
            })
            
            print(f"{method_name} @ noise={noise_level:.2f}: "
                  f"hit@1={hits_at_1/n_items:.2%}, "
                  f"hit@5={hits_at_5/n_items:.2%}, "
                  f"latency={total_time/n_items*1000:.2f}ms")
    
    return results


def create_membrain_method(encoder):
    """Create Membrain SNN method wrapper."""
    memory = BiCameralMemory(n_neurons=500, dimensions=encoder.output_dim)
    memory._ensure_simulator()
    
    class MembrainMethod:
        def store(self, context_id: str, vector):
            sparse = encoder.encode(vector)
            memory.remember(context_id, sparse, importance=1.0)
        
        def query(self, vector, k: int = 5):
            sparse = encoder.encode(vector)
            results = memory.recall(sparse, threshold=0.1, max_results=k)
            return [(r.context_id, r.confidence) for r in results]
    
    return MembrainMethod()


def main():
    parser = argparse.ArgumentParser(description="Benchmark noise robustness")
    parser.add_argument("--output", "-o", default="bench_results.csv", help="Output CSV file")
    parser.add_argument("--n-items", type=int, default=500, help="Number of items to store")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print(f"Running benchmark with {args.n_items} items, seed={args.seed}")
    results = run_benchmark(n_items=args.n_items, seed=args.seed)
    
    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
```

---

## Files / Modules

| File | Action |
|------|--------|
| `tests/integration/__init__.py` | **Create** |
| `tests/integration/test_noise_robustness.py` | **Create** — Noise tests |
| `bench/__init__.py` | **Create** |
| `bench/baselines.py` | **Create** — Baseline methods |
| `bench/bench_noise.py` | **Create** — Benchmark harness |
| `bench/README.md` | **Create** — Benchmark documentation |

---

## Expected Results

Based on FlyHash + SNN properties, expected outcomes:

| Method | Hit@1 (0% noise) | Hit@1 (10% noise) | Hit@1 (20% noise) |
|--------|------------------|-------------------|-------------------|
| Cosine | 100% | ~95% | ~85% |
| FlyHash+Jaccard | ~98% | ~85% | ~65% |
| Membrain SNN | ~95% | ~90% | ~75% |

**Key differentiators to demonstrate:**
1. SNN pattern completion improves partial match recovery
2. Attractor dynamics clean up noisy representations
3. Trade-off: SNN may have higher latency

---

## Acceptance Criteria

- [ ] Results are reproducible with seeded randomness
- [ ] Benchmark outputs CSV with all metrics
- [ ] At least one measurable advantage shown vs baseline
- [ ] Integration tests pass in CI
- [ ] Benchmark can run in reasonable time (<5 min)

---

## Risks / Notes

- **SNN may not beat baselines initially**: If results show no advantage, this reveals a real problem to fix in P1 PRs
- **Latency**: SNN recall is slower — document the trade-off
- **Nengo CI**: May need to skip integration tests in fast CI mode

---

## Definition of Done

- [ ] Integration tests added and pass
- [ ] Benchmark harness produces reproducible CSV
- [ ] Results documented in bench/README.md
- [ ] Comparison shows at least one metric where SNN wins
- [ ] If SNN underperforms, documented as P1 priority
