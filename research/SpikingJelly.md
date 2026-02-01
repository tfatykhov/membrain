# SpikingJelly: SNN Framework Analysis

**Repository:** [github.com/fangwei123456/spikingjelly](https://github.com/fangwei123456/spikingjelly)
**Documentation:** [spikingjelly.readthedocs.io](https://spikingjelly.readthedocs.io)
**License:** Mozilla Public License 2.0
**Stars:** ~3,000

## Overview

SpikingJelly is an open-source deep learning framework for Spiking Neural Networks (SNNs) based on PyTorch. Developed by researchers at Peking University, it's one of the fastest SNN frameworks available.

## Key Features

### 1. Multiple Backends
- **Pure PyTorch**: Flexible, CPU/GPU portable
- **CuPy/CUDA**: Custom kernels, fastest option (~0.26s for 16k neurons)
- **Triton**: New backend, JIT-compiled GPU kernels

### 2. Neuron Models
- Integrate-and-Fire (IF)
- Leaky Integrate-and-Fire (LIF)
- Parametric LIF (learnable time constants)
- Custom neurons via surrogate gradients

### 3. Training Methods
- Surrogate gradient descent (backprop through time)
- ANN-to-SNN conversion
- CUDA-accelerated temporal batching

### 4. Dataset Support
- DVS-CIFAR10, N-MNIST (vision)
- SHD, SSC (audio) — same as DMP-SNN paper

## Benchmark Performance

From [Open Neuromorphic benchmarks](https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/):

| Framework | 16k neurons (fwd+bwd) | Backend |
|-----------|----------------------|---------|
| **SpikingJelly (CuPy)** | **0.26s** | CUDA |
| Spyx | 0.30s | JAX |
| Lava DL (SLAYER) | 0.40s | CUDA |
| Norse (compiled) | 0.50s | torch.compile |
| snnTorch | 1.50s | PyTorch |
| Norse | 2.00s | PyTorch |

**SpikingJelly is 5-10x faster than pure PyTorch implementations.**

## SpikingJelly vs Nengo

| Aspect | SpikingJelly | Nengo (current) |
|--------|-------------|-----------------|
| **Primary use** | Deep learning SNNs | Biological modeling |
| **Paradigm** | Surrogate gradients | Neural Engineering Framework (NEF) |
| **Speed (GPU)** | Very fast | Moderate |
| **Hardware** | GPU-optimized | Loihi, analog chips |
| **Learning** | Backprop through time | PES, Voja, BCM |
| **Flexibility** | PyTorch ecosystem | Nengo DSL |
| **Loihi support** | ❌ No | ✅ First-class |
| **Community** | Growing | Established |

## Relevance to Membrain

### Why Consider SpikingJelly

1. **DMP-SNN uses it**: The dual memory pathway paper uses SpikingJelly
2. **Speed**: Faster training iteration during development
3. **PyTorch native**: Easier integration with modern ML tooling
4. **Active development**: Triton backend, spiking attention layers

### Why Keep Nengo

1. **Loihi deployment**: Nengo has first-class Intel Loihi support
2. **NEF theory**: Principled approach to neural representation
3. **Voja learning**: Our BiCameralMemory uses Nengo-specific learning rules
4. **Existing code**: BiCameralMemory already implemented in Nengo

## Hybrid Architecture Proposal

```
┌─────────────────────────────────────────────────────────┐
│                    Membrain Hybrid                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────┐     ┌─────────────────┐            │
│  │   Development   │     │   Production    │            │
│  │   (SpikingJelly)│     │   (Nengo)       │            │
│  ├─────────────────┤     ├─────────────────┤            │
│  │ • Fast training │     │ • Loihi deploy  │            │
│  │ • GPU-optimized │     │ • NEF-based     │            │
│  │ • Prototyping   │     │ • Voja learning │            │
│  └────────┬────────┘     └────────┬────────┘            │
│           │                       │                      │
│           └───────────┬───────────┘                      │
│                       │                                  │
│              ┌────────▼────────┐                        │
│              │       NIR       │                        │
│              │ (Neuromorphic   │                        │
│              │  Intermediate   │                        │
│              │  Representation)│                        │
│              └─────────────────┘                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### NIR (Neuromorphic Intermediate Representation)

Both SpikingJelly and Nengo support NIR, allowing model exchange:
- Train in SpikingJelly (fast GPU)
- Export to NIR
- Import to Nengo for Loihi deployment

**Repository:** [github.com/neuromorphs/NIR](https://github.com/neuromorphs/NIR)

## Component Mapping

| Membrain Component | Current (Nengo) | Hybrid Option |
|-------------------|-----------------|---------------|
| FlyHash encoder | numpy | numpy (unchanged) |
| BiCameralMemory | Nengo SNN | Keep Nengo (Voja learning) |
| Attractor dynamics | Nengo (planned) | SpikingJelly (faster dev) |
| DMP slow pathway | Not implemented | SpikingJelly (matches paper) |
| Loihi deployment | Nengo-Loihi | Nengo-Loihi (unchanged) |

## Implementation Phases

### Phase 1: Evaluation (No code changes)
- [x] Research SpikingJelly capabilities
- [ ] Test NIR conversion between frameworks
- [ ] Benchmark Membrain-like workload in both

### Phase 2: Parallel Development
- [ ] Implement attractor dynamics in SpikingJelly
- [ ] Compare performance with Nengo equivalent
- [ ] Evaluate accuracy/speed tradeoffs

### Phase 3: Integration Decision
- [ ] Choose framework per component based on benchmarks
- [ ] Implement NIR-based conversion if needed
- [ ] Document hybrid architecture

## Research Questions

1. **NIR fidelity**: How much accuracy is lost in NIR conversion?
2. **Voja in SpikingJelly**: Can we implement Voja-like learning?
3. **Memory footprint**: How do they compare for our embedding sizes?
4. **Temporal binding**: Which handles long sequences better?

## Related Work

Other SNN frameworks to monitor:
- **Lava** (Intel): Official Loihi framework
- **snnTorch**: Best tutorials, CUDA support
- **Norse**: Functional design, torch.compile friendly
- **Spyx**: JAX-based, TPU support

## Citations

```bibtex
@article{fang2023spikingjelly,
  title={SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence},
  author={Fang, Wei and Chen, Yanqi and Ding, Jianhao and Yu, Zhaofei and Masquelier, Timothée and Chen, Ding and Huang, Liwei and Zhou, Huihui and Li, Guoqi and Tian, Yonghong},
  journal={Science Advances},
  volume={9},
  number={40},
  year={2023}
}
```
