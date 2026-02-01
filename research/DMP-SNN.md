# DMP-SNN: Dual Memory Pathway for Spiking Neural Networks

**Paper:** [arXiv:2512.07602](https://arxiv.org/abs/2512.07602)
**Code:** [github.com/sunpengfei1122/Dual_memory_pathways](https://github.com/sunpengfei1122/Dual_memory_pathways)
**Published:** December 2025
**Authors:** Pengfei Sun*, Zhe Su*, Jascha Achterberg* (Imperial College London, ETH Zurich, Oxford, Cambridge)

## Summary

DMP-SNN addresses the core challenge of maintaining long-range temporal context in SNNs without sacrificing event-driven efficiency.

### Problem
Standard LIF neurons only capture instantaneous evidence — they either remember *what* happened or *when*, but not both over long timescales.

Existing solutions (dense recurrence, learnable delays) are computationally expensive:
- Dense recurrence scales quadratically with layer width
- Long delays require deep on-chip buffers

### Solution: Dual Memory Pathway

Inspired by cortical fast-slow organization:

```
┌─────────────────────────────────────────┐
│              DMP-SNN Layer              │
├─────────────────────────────────────────┤
│  Fast Pathway: LIF spiking neurons (N)  │
│         ↑                               │
│         │ modulates                     │
│         ↓                               │
│  Slow Pathway: memory state m ∈ ℝᵈ     │
│         (d << N, compact context)       │
└─────────────────────────────────────────┘
```

Each layer maintains a **low-dimensional state vector** `m ∈ ℝᵈ` where `d << N`:
1. Evolves under slow dynamics
2. Summarizes recent activity
3. Modulates fast spiking neurons

### Key Results

| Metric | Value |
|--------|-------|
| Parameter reduction | 40-60% fewer than SOTA SNNs |
| Throughput | 4x increase |
| Energy efficiency | 5x improvement |
| Benchmarks | PS-MNIST, S-MNIST, SHD, SSC |

## Code Repository

**GitHub:** https://github.com/sunpengfei1122/Dual_memory_pathways

### Requirements
- Python 3
- PyTorch 1.12.1 (CUDA 11.3)
- spikingjelly 0.0.0.0.14

### Example Usage
```bash
cd shd/src
python train_spiking.py -d 5 -t 40
# -d: memory state dimension
# -t: state buffer length
```

### Datasets
- SHD/SSC: https://zenkelab.org/datasets/

## Relevance to Membrain

### Architectural Alignment

| DMP-SNN Concept | Membrain Component |
|-----------------|-------------------|
| Fast pathway (LIF spikes) | FlyHash encoding (sparse, immediate) |
| Slow pathway (memory state) | BiCameralMemory SNN state |
| Low-dimensional state `m` | Compressed episode representation |
| Temporal modulation | Attractor dynamics influence |

### Direct Applications

1. **BiCameralMemory Architecture**
   - Current: FlyHash → SNN (flat)
   - With DMP: FlyHash → Fast pathway + Slow memory pathway
   - Benefit: Explicit separation of immediate vs. contextual processing

2. **Temporal Binding (PR-011)**
   - DMP's slow pathway explicitly handles long-range dependencies
   - Could replace or augment our planned temporal binding mechanism

3. **Memory Efficiency**
   - Their `d << N` design matches our constraints
   - FlyHash: 2048-dim sparse → SNN state: could be much smaller

4. **Hardware Co-design**
   - Open-source hardware architecture
   - Near-memory-compute design
   - Relevant for Docker/Loihi integration

### Implementation Ideas

```python
class BiCameralDMP(BiCameralMemory):
    """BiCameralMemory with DMP-inspired dual pathway."""
    
    def __init__(self, input_dim: int, memory_dim: int, state_dim: int = 8):
        super().__init__(input_dim, memory_dim)
        self.state_dim = state_dim  # d << memory_dim
        
        # Slow memory pathway
        self.slow_state = np.zeros(state_dim)
        self.slow_decay = 0.95  # Slow dynamics
        
        # State-to-memory projection
        self.state_proj = np.random.randn(state_dim, memory_dim) * 0.01
    
    def update_slow_state(self, sparse_input: np.ndarray):
        """Compress activity into slow state."""
        # Simple exponential moving average for now
        compressed = sparse_input[:self.state_dim]  # Project to low-dim
        self.slow_state = self.slow_decay * self.slow_state + (1 - self.slow_decay) * compressed
    
    def modulated_recall(self, query: np.ndarray) -> np.ndarray:
        """Recall modulated by slow state."""
        # Standard recall
        base_recall = self.recall(query)
        
        # Modulate by slow state
        modulation = self.slow_state @ self.state_proj
        return base_recall + 0.1 * modulation  # Weighted combination
```

### Research Questions

1. **Optimal state dimension `d`**: Paper uses small values (5-40). What's optimal for Membrain's use case?

2. **State update rule**: Paper uses specific dynamics. Should we adopt theirs or design custom?

3. **Integration with FlyHash**: Does FlyHash output naturally decompose into fast/slow components?

4. **Attractor interaction**: How does slow state interact with attractor dynamics?

## Related Work by Same Authors

Also relevant from Pengfei Sun's repos:
- **DMU (Delayed Memory Units)**: RNN implementation with delays
- **Adaptive_axonal_delay**: Learnable delays for SNNs
- **LMUFormer**: Legendre Memory Units for efficient temporal modeling

## Citation

```bibtex
@article{sun2025dmp,
  title={Algorithm-hardware co-design of neuromorphic networks with dual memory pathways},
  author={Sun, Pengfei and Su, Zhe and Achterberg, Jascha and Indiveri, Giacomo and Goodman, Dan FM and Akarca, Danyal},
  journal={arXiv preprint arXiv:2512.07602},
  year={2025}
}
```
