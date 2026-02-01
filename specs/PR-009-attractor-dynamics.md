# PR-009 — True Cleanup / Attractor Dynamics (Defensible Pattern Completion)

## Status: 🔴 Not Started — P1 Priority

## Current State Analysis

### What Exists

The current `BiCameralMemory.recall()` in `core.py`:
1. Encodes query via FlyHash (sparse representation)
2. Runs Nengo simulation briefly to collect activations
3. Returns matches based on activity similarity

### What's Missing

**True pattern completion**: Currently, recall is essentially a similarity lookup. There's no evidence that:
1. The SNN is actually "cleaning up" noisy patterns
2. Attractor dynamics are settling to stored patterns
3. Pattern completion is happening beyond what FlyHash provides

This is critical for the "synthetic hippocampus" claim.

---

## Objective

Make pattern completion depend on actual network dynamics (cleanup/attractor behavior), demonstrating measurable improvement from the SNN beyond simple similarity matching.

---

## Background: Attractor Networks

**Hopfield-style attractors**: Networks can store patterns as stable states. When given a partial/noisy input, the network dynamics should converge to the nearest stored pattern.

**Key mechanisms:**
1. **Recurrent connections** — Neurons reinforce each other when co-activated
2. **Inhibition** — Competition between patterns prevents blending
3. **Settling time** — Network needs time to converge to attractor

---

## Detailed Requirements

### A. Attractor/Cleanup Network Module

Create `src/membrain/attractor.py`:

```python
"""
Attractor network for pattern completion.

Implements cleanup memory using recurrent dynamics to
converge noisy inputs to stored patterns.
"""

import numpy as np
from numpy.typing import NDArray
import nengo

class AttractorMemory:
    """
    Attractor network for pattern cleanup.
    
    Uses auto-associative memory principles:
    - Recurrent connections encode stored patterns
    - Lateral inhibition provides competition
    - Dynamics converge to nearest attractor
    """
    
    def __init__(
        self,
        dimensions: int,
        n_neurons: int = 500,
        learning_rate: float = 1e-4,
        recall_duration_ms: int = 50,
        dt: float = 0.001,
    ):
        self.dimensions = dimensions
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.recall_duration_ms = recall_duration_ms
        self.dt = dt
        
        # Stored patterns (for weight initialization)
        self._patterns: list[NDArray[np.float32]] = []
        
        # Build network
        self._network = None
        self._simulator = None
        self._build_network()
    
    def _build_network(self) -> None:
        """Build the Nengo attractor network."""
        self._network = nengo.Network(label="AttractorMemory")
        
        with self._network:
            # Input node (will be set during recall)
            self.input_node = nengo.Node(output=np.zeros(self.dimensions))
            
            # Main ensemble with recurrent connections
            self.memory = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=self.dimensions,
                label="memory",
            )
            
            # Input connection (transient)
            self.input_conn = nengo.Connection(
                self.input_node,
                self.memory,
                synapse=0.01,
            )
            
            # Recurrent connection (auto-associative)
            # Initialize as identity; will be modified by learning
            self.recurrent_conn = nengo.Connection(
                self.memory,
                self.memory,
                transform=np.eye(self.dimensions) * 0.1,  # Weak initial
                synapse=0.1,
                learning_rule_type=nengo.PES(learning_rate=self.learning_rate),
            )
            
            # Output probe
            self.output_probe = nengo.Probe(
                self.memory,
                synapse=0.01,
            )
    
    def store(self, pattern: NDArray[np.float32]) -> None:
        """
        Store a pattern as an attractor.
        
        Updates recurrent weights using Hebbian-like learning.
        """
        self._patterns.append(pattern.copy())
        
        # Update recurrent weights: W += lr * pattern @ pattern.T
        # This creates an attractor basin around the pattern
        if hasattr(self, 'recurrent_conn'):
            with self._network:
                # Hebbian update to transform
                outer = np.outer(pattern, pattern)
                current = self.recurrent_conn.transform.init
                if isinstance(current, np.ndarray):
                    new_transform = current + self.learning_rate * outer
                    self.recurrent_conn.transform = new_transform
    
    def complete(self, partial: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Complete a partial/noisy pattern using attractor dynamics.
        
        Args:
            partial: Noisy or partial input pattern.
        
        Returns:
            Cleaned up pattern after network settling.
        """
        # Set input
        self.input_node.output = partial
        
        # Build simulator if needed
        if self._simulator is None:
            self._simulator = nengo.Simulator(
                self._network,
                dt=self.dt,
                progress_bar=False,
            )
        else:
            self._simulator.reset()
        
        # Run for settling time
        duration = self.recall_duration_ms / 1000.0
        self._simulator.run(duration)
        
        # Get final state (average of last few samples)
        output = self._simulator.data[self.output_probe]
        completed = np.mean(output[-10:], axis=0).astype(np.float32)
        
        return completed
    
    def measure_cleanup(
        self,
        original: NDArray[np.float32],
        noisy: NDArray[np.float32],
    ) -> dict:
        """
        Measure cleanup effectiveness.
        
        Returns:
            Dict with input_similarity, output_similarity, improvement
        """
        input_sim = float(np.dot(noisy, original) / (
            np.linalg.norm(noisy) * np.linalg.norm(original) + 1e-9
        ))
        
        completed = self.complete(noisy)
        output_sim = float(np.dot(completed, original) / (
            np.linalg.norm(completed) * np.linalg.norm(original) + 1e-9
        ))
        
        return {
            "input_similarity": input_sim,
            "output_similarity": output_sim,
            "improvement": output_sim - input_sim,
        }
```

### B. Integration with BiCameralMemory

Update `core.py` to use attractor cleanup:

```python
class BiCameralMemory:
    def __init__(self, ..., use_attractor: bool = True):
        self.use_attractor = use_attractor
        if use_attractor:
            self.attractor = AttractorMemory(
                dimensions=dimensions,
                n_neurons=n_neurons // 2,  # Split neurons
            )
    
    def recall(self, query_vector, ...):
        if self.use_attractor:
            # Clean up query first
            cleaned_query = self.attractor.complete(query_vector)
            # Use cleaned query for matching
            # ...
```

### C. Cleanup Effectiveness Test

```python
def test_cleanup_improves_similarity():
    """Pattern completion should improve similarity to target."""
    attractor = AttractorMemory(dimensions=128, n_neurons=500)
    
    # Store some patterns
    rng = np.random.default_rng(42)
    patterns = [rng.standard_normal(128).astype(np.float32) for _ in range(10)]
    for p in patterns:
        attractor.store(p / np.linalg.norm(p))
    
    # Test cleanup on noisy version
    original = patterns[0] / np.linalg.norm(patterns[0])
    noise = rng.standard_normal(128).astype(np.float32)
    noisy = original + 0.3 * noise
    noisy = noisy / np.linalg.norm(noisy)
    
    result = attractor.measure_cleanup(original, noisy)
    
    # Improvement should be positive
    assert result["improvement"] > 0, \
        f"Cleanup should improve similarity, got {result['improvement']:.4f}"
    assert result["output_similarity"] > result["input_similarity"]
```

---

## Files / Modules

| File | Action |
|------|--------|
| `src/membrain/attractor.py` | **Create** — Attractor network module |
| `src/membrain/core.py` | **Update** — Integrate attractor cleanup |
| `tests/test_attractor.py` | **Create** — Attractor tests |
| `bench/bench_cleanup.py` | **Create** — Cleanup effectiveness benchmark |

---

## Metrics to Measure

1. **Cleanup improvement**: `sim(output, target) - sim(input, target)`
2. **Convergence time**: How quickly does network settle?
3. **Basin width**: How much noise can be tolerated?
4. **False attraction**: Rate of converging to wrong pattern

---

## Acceptance Criteria

- [ ] Demonstrable pattern completion improvement on synthetic suite
- [ ] Recall remains read-only (learning gate enforced)
- [ ] Benchmark shows improvement over no-attractor baseline
- [ ] Cleanup doesn't significantly increase latency (or trade-off documented)

---

## Risks / Notes

- **Nengo complexity**: Attractor networks are non-trivial to tune
- **Capacity limits**: Hopfield networks have limited pattern capacity (~0.14N)
- **Training requirements**: May need to run learning during Remember, not just Recall
- **Alternative**: Consider sparse distributed memory (SDM) as simpler alternative

---

## Research References

1. Hopfield, J.J. (1982). Neural networks and physical systems with emergent collective computational abilities.
2. Eliasmith, C. (2013). How to Build a Brain. Oxford University Press.
3. Nengo documentation on learning rules and recurrent connections.

---

## Definition of Done

- [ ] Tests prove cleanup improves similarity to target
- [ ] Benchmark shows measurable advantage
- [ ] Recall learning gate preserved
- [ ] Documentation explains attractor mechanism
