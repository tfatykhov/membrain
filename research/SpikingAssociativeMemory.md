# Spiking Representation Learning for Associative Memories

**Paper:** [arXiv:2406.03054](https://arxiv.org/abs/2406.03054)
**Published:** Frontiers in Neuroscience, 2024
**Authors:** Naresh Balaji Ravichandran et al.

## Summary

A novel SNN architecture that performs **unsupervised representation learning** combined with **associative memory operations** using biologically-plausible mechanisms.

## Key Features

### Architecture
- Based on **neocortical columnar organization**
- **Feedforward projections:** Learn hidden representations
- **Recurrent projections:** Form associative memories
- Neuron-units: Poisson spike generators (~1 Hz mean, ~100 Hz max firing rate)

### Learning Mechanisms
- **Hebbian synaptic plasticity** — "Neurons that fire together, wire together"
- **Activity-dependent structural plasticity** — Connections grow/prune based on activity
- **Sparse firing** — Biologically realistic rates

### Evaluated Properties
- Pattern completion ✅
- Perceptual rivalry ✅
- Distortion resistance ✅
- Prototype extraction ✅

## Relevance to Membrain

### Direct Applications

| Paper Concept | Membrain Component |
|---------------|-------------------|
| Unsupervised representation learning | FlyHash encoding layer |
| Associative memory formation | BiCameralMemory recall |
| Pattern completion | Attractor dynamics (H1) |
| Distortion resistance | Noise-robust retrieval |
| Columnar organization | Potential architecture enhancement |

### Key Insights

1. **Hebbian + Structural Plasticity:** The paper combines two plasticity types:
   - Synaptic (weight changes)
   - Structural (connection growth/pruning)
   
   This maps to our H3 (forgetting as feature) — unused connections decay.

2. **Sparse Firing:** ~1 Hz mean rate is extremely sparse, similar to our FlyHash sparse codes. This validates our approach.

3. **Feedforward + Recurrent:** Separating representation learning from memory formation aligns with our split-brain architecture (Payload Store + Associative Index).

4. **Attractor Properties:** Demonstrated pattern completion and distortion resistance — exactly what we need for H1 (Associative Recall > Similarity Search).

### Implementation Ideas

```python
class HebbianAssociativeMemory:
    """Hebbian learning for associative memory formation."""
    
    def __init__(self, input_dim: int, memory_dim: int):
        self.weights = np.zeros((input_dim, memory_dim))
        self.learning_rate = 0.01
        self.decay_rate = 0.001  # Structural plasticity
    
    def learn(self, pre: np.ndarray, post: np.ndarray):
        """Hebbian update: strengthen co-active connections."""
        # Outer product: Δw = η * pre * post^T
        self.weights += self.learning_rate * np.outer(pre, post)
        
        # Structural plasticity: decay weak connections
        self.weights *= (1 - self.decay_rate)
        self.weights[np.abs(self.weights) < 0.01] = 0
    
    def recall(self, cue: np.ndarray, iterations: int = 10) -> np.ndarray:
        """Pattern completion via attractor dynamics."""
        state = cue.copy()
        for _ in range(iterations):
            state = np.tanh(self.weights.T @ state)
        return state
```

### Integration Path

1. **Phase 3 (SNN MVP):** Use this paper's approach for associative memory instead of/alongside Nengo's built-in
2. **Phase 4 (Dynamics):** Their structural plasticity mechanism for implementing forgetting
3. **Validation:** Use their evaluation metrics (pattern completion, distortion resistance)

## Limitations

- Tested on relatively small scale
- Poisson spike generators may be slower than rate-coded alternatives
- No hardware acceleration path discussed

## Citation

```bibtex
@article{ravichandran2024spiking,
  title={Spiking representation learning for associative memories},
  author={Ravichandran, Naresh Balaji and others},
  journal={Frontiers in Neuroscience},
  volume={18},
  year={2024},
  doi={10.3389/fnins.2024.1439414}
}
```
