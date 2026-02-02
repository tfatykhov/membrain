# Membrain Benchmark Suite

Benchmark infrastructure for measuring Membrain's noise robustness against baseline vector stores.

## Overview

This suite compares Membrain's neuromorphic memory against traditional vector search methods:

| Baseline | Description | Use Case |
|----------|-------------|----------|
| `CosineBaseline` | Naive brute-force cosine similarity | Accuracy floor |
| `FAISSFlatBaseline` | FAISS exact search (optimized) | Production reference |
| `MembrainStore` | Membrain SNN via gRPC | Our system |

## Quick Start

```bash
# Install benchmark dependencies
pip install -e ".[bench]"

# Run smoke test
python -m bench.smoke_test

# Run benchmark
python -m bench.bench_noise --output results/benchmark.csv
```

## CLI Reference

The benchmark runner `bench_noise.py` supports the following arguments:

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--output` | `-o` | None | Output CSV file path |
| `--samples` | `-n` | 100 | Number of vectors to benchmark |
| `--dim` | `-d` | 128 | Vector dimensionality |
| `--noise-levels` | | 0.0 0.05... | List of noise levels to test |
| `--seed` | `-s` | 42 | Random seed for reproducibility |
| `--methods` | `-m` | All | Specific methods (e.g., `cosine`, `membrain`) |
| `--skip-membrain` | | False | Skip Membrain server benchmark |
| `--verbose` | `-v` | False | Enable verbose logging |

## Structure

```
bench/
├── __init__.py          # Package exports
├── protocol.py          # VectorStore interface
├── metrics.py           # BenchmarkResult, Timer
├── datasets.py          # SyntheticDataset, noise functions
├── baselines/
│   ├── cosine.py        # CosineBaseline
│   ├── faiss_flat.py    # FAISSFlatBaseline (optional)
│   └── membrain_client.py  # Membrain gRPC adapter
├── bench_noise.py       # Main benchmark runner
└── README.md            # This file
```

## Protocol

All vector stores implement the `VectorStore` protocol:

```python
class VectorStore(Protocol):
    def store(self, key: str, vector: NDArray) -> None: ...
    def query(self, vector: NDArray, k: int) -> list[tuple[str, float]]: ...
    def clear(self) -> None: ...
    @property
    def count(self) -> int: ...
    @property
    def memory_mb(self) -> float: ...
```

## Datasets

### Synthetic Gaussian
```python
from bench.datasets import SyntheticDataset

ds = SyntheticDataset.gaussian(n=10000, dim=64, seed=42)
```

### Clustered (more realistic)
```python
ds = SyntheticDataset.clustered(n=10000, dim=64, n_clusters=20, seed=42)
```

### Adding Noise
```python
from bench.datasets import add_noise
import numpy as np

rng = np.random.default_rng(42)
noisy_vec = add_noise(original_vec, level=0.2, rng=rng)
```

## Metrics

| Metric | Description |
|--------|-------------|
| `hit_at_1` | Fraction of queries where correct answer is top result |
| `hit_at_5` | Fraction where correct answer is in top 5 |
| `mrr` | Mean Reciprocal Rank |
| `avg_latency_ms` | Average query time |
| `p99_latency_ms` | 99th percentile latency |
| `memory_mb` | Memory usage |
| `throughput_qps` | Queries per second |

## Running Tests

```bash
# Run all benchmark tests
pytest tests/bench/ -v

# Run with coverage
pytest tests/bench/ -v --cov=bench
```

## Expected Results

At 20% noise level, we expect:

| Method | Hit@1 | Latency |
|--------|-------|---------|
| Cosine | ~85% | 0.05ms |
| FAISS | ~85% | 0.02ms |
| Membrain | ~87%+ | 2-5ms |

The thesis: **Membrain's attractor dynamics should outperform at high noise levels**, despite higher latency.
