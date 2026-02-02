# Feature 11: Attractor Dynamics — Minsky Mapping

*Connecting Society of Mind Chapter 4 to implementation*

---

## The Core Insight

> "One function of the Self is to keep us from changing too rapidly."
> — Minsky, 4.4 The Conservative Self

Attractor dynamics implement **the conservative self** at the memory level. Each stored pattern creates a basin of attraction — a region of state space that resists perturbation and pulls nearby patterns toward a stable state.

---

## Concept Mapping

| Minsky (Ch. 4) | Attractor Implementation |
|----------------|--------------------------|
| **Conservative Self** | Attractor basins resist rapid change |
| **Self-ideals as constraints** | Learned weights define basin shapes |
| **"Chains we forge"** | Hebbian connections lock in patterns |
| **Roundabout exploitation** | Dynamics find paths to stored states |
| **Slow-changing agencies** | Low learning rate, high stability |
| **Fast-changing agencies** | Input space (queries) can be noisy |

---

## Design Principles from Minsky

### 1. Stability Through Structure (4.4)

> "We also have to find some ways to constrain the changes we might later make — to prevent ourselves from turning those plan-agents off again!"

**Implementation:**
- Stored patterns should be **hard to dislodge**
- High energy barriers between attractors
- Recurrent weights create self-reinforcing states

```python
class AttractorMemory:
    def store(self, pattern: NDArray) -> None:
        """Hebbian storage creates deep basin."""
        # Outer product reinforces pattern as stable state
        # Basin depth ∝ storage count (rehearsal deepens)
        self.weights += self.learning_rate * np.outer(pattern, pattern)
```

### 2. Indirect Paths (4.5)

> "All direct connections must have been removed in the course of our evolution."

**Implementation:**
- No direct lookup (that's just similarity search)
- Pattern must **evolve through dynamics** to reach attractor
- Multiple timesteps of settling create the "roundabout path"

```python
def complete(self, partial: NDArray, max_steps: int = 50) -> NDArray:
    """Pattern evolves through dynamics — no shortcuts."""
    state = partial.copy()
    for _ in range(max_steps):
        # State update: move toward attractors
        activation = self.weights @ state
        state = np.tanh(activation)  # Bounded nonlinearity
        
        # Check convergence
        if self._converged(state, prev_state):
            break
    return state
```

### 3. Multiple Selves (4.2)

> "Sometimes we regard ourselves as single, self-coherent entities. Other times we feel decentralized or dispersed."

**Implementation:**
- Multiple patterns can coexist as distinct attractors
- **Lateral inhibition** prevents blend states
- Competition ensures clean separation

```python
def _apply_inhibition(self, state: NDArray) -> NDArray:
    """Winners suppress losers — no mushy blends."""
    # Top-k sparsification or soft-WTA
    threshold = np.percentile(np.abs(state), 90)
    return np.where(np.abs(state) > threshold, state, 0)
```

### 4. Value in the Crust (4.3)

> "The value of a human self lies not in some small, precious core, but in its vast, constructed crust."

**Implementation:**
- Memory quality comes from **the network of relationships**
- Not a lookup table but an interconnected dynamical system
- The crust = learned weight structure

---

## Metrics Inspired by Minsky

### 1. Stability (Conservative Self)

How well does the network resist perturbation?

```python
def measure_stability(self, pattern: NDArray, noise_levels: list[float]) -> dict:
    """Conservative self: resist change."""
    results = {}
    for noise in noise_levels:
        noisy = pattern + noise * np.random.randn(*pattern.shape)
        recovered = self.complete(noisy)
        results[noise] = cosine_similarity(pattern, recovered)
    return results
```

### 2. Basin Separation (Multiple Selves)

Are different memories distinguishable?

```python
def measure_separation(self, patterns: list[NDArray]) -> float:
    """Multiple selves: distinct identities."""
    overlaps = []
    for i, p1 in enumerate(patterns):
        for p2 in patterns[i+1:]:
            recovered1 = self.complete(p1 + 0.3 * np.random.randn(*p1.shape))
            recovered2 = self.complete(p2 + 0.3 * np.random.randn(*p2.shape))
            # Check: did they converge to different attractors?
            overlaps.append(cosine_similarity(recovered1, recovered2))
    return 1.0 - np.mean(overlaps)  # Higher = better separation
```

### 3. Cleanup Gain (Indirect Paths)

Does the roundabout path help?

```python
def measure_cleanup_gain(self, original: NDArray, noisy: NDArray) -> float:
    """Indirect path should improve over direct."""
    direct_sim = cosine_similarity(original, noisy)
    recovered = self.complete(noisy)
    recovered_sim = cosine_similarity(original, recovered)
    return recovered_sim - direct_sim  # Positive = dynamics helped
```

---

## Benchmark Target

**Baseline (cosine similarity, no dynamics):** 59% hit@1 at 30% noise

**Target:** Beat this with attractor cleanup. If dynamics help:
- Noisy query → complete() → cleaned query
- Cleaned query should match stored pattern better than raw noisy query

**Success metric:**
```
cleanup_gain = hit@1_with_attractor - hit@1_without > 0.10
```

At 30% noise, we want: **>69% hit@1** (10+ percentage point improvement)

---

## Implementation Plan

### Phase 1: Basic Attractor (Hopfield-style)
1. Outer-product storage (Hebbian)
2. Synchronous update dynamics
3. Convergence detection

### Phase 2: Nengo Integration
1. Implement in spiking neurons
2. Learning gate integration (recall = read-only)
3. Sparse representation (FlyHash compatible)

### Phase 3: Benchmark
1. Run bench_noise.py with attractor
2. Compare hit@1 at all noise levels
3. Measure cleanup gain distribution

---

## Connection to Consolidation (Feature 10)

Attractor dynamics and consolidation work together:

| Feature | Minsky Parallel | Function |
|---------|-----------------|----------|
| **Attractor (11)** | Conservative self | Resist perturbation during recall |
| **Consolidation (10)** | Slow-changing agencies | Strengthen patterns over time |

Consolidation makes basins **deeper** (more stable).
Attractor dynamics make basins **useful** (enable cleanup).

---

## References

- Minsky, Society of Mind Ch. 4: The Self
- Hopfield (1982): Neural networks and physical systems with emergent collective computational abilities
- Membrain PRD Feature 11: Attractor Dynamics
- Benchmark baseline: `benchmarks/2026-02-02-baseline-clustered.json`
