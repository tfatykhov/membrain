<p align="center">
  <img src="docs/media/logo-v2.png" alt="Membrain - Neuromorphic Bio-Memory System" width="600">
</p>

**Neuromorphic Memory Bridge for LLM Agents**

A Spiking Neural Network (SNN) based memory system that provides associative recall and continuous learning for AI agents — like a synthetic hippocampus.

[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-PoC-orange)](https://github.com/tfatykhov/membrain)

[![CI](https://github.com/tfatykhov/membrain/actions/workflows/ci.yml/badge.svg)](https://github.com/tfatykhov/membrain/actions/workflows/ci.yml)

## Why Membrain?

> *"All competitors treat memory as Information Retrieval. We treat memory as State Reconstruction."*
> — [Product Vision](docs/PRODUCT_VISION.md)

Traditional RAG (Retrieval Augmented Generation) uses vector databases for static retrieval. Membrain is different:

| Feature | Vector DB (RAG) | Membrain (SNN) |
|---------|-----------------|----------------|
| **Retrieval** | Static similarity search | Dynamic associative recall |
| **Learning** | Requires reindexing | Continuous, plastic updates |
| **Associations** | None (just nearest neighbors) | Forms semantic links between concepts |
| **Energy** | O(n) comparisons | O(active neurons) — sparse |
| **Pattern Completion** | No | Yes — recalls from partial/noisy input |

## Architecture

![Membrain Architecture](docs/media/architecture.png)

## Key Concepts

### FlyHash Encoding
Converts dense LLM embeddings (1536-d floats) into sparse binary spike trains (20,000-d binary) using `int8` random projection + Winner-Take-All inhibition. This enables efficient SNN processing with minimal memory footprint (~30 MB).

### Associative Memory
Uses Nengo's neural populations with Voja learning rule to form dynamic associations. Unlike hash tables, memories naturally cluster by semantic similarity.

### Pattern Completion
Query with noisy or partial input → network settles into learned attractor state → returns complete memory. Works with up to 20% noise. Now integrated with `AttractorMemory` for enhanced cleanup (requires `MEMBRAIN_USE_ATTRACTOR=true`).

> **Warning**: Enabling attractor dynamics creates an $O(N^2)$ weight matrix. For default 20,000 dimensions, this requires ~1.6 GB RAM. Use with caution on high-dimensional inputs.

### Stochastic Consolidation
Mimics biological memory consolidation by injecting noise during sleep phases. This drives the network into robust attractor states, pruning weak transient memories while strengthening important patterns (Feature 08).

## Installation

```bash
# Clone the repository
git clone https://github.com/tfatykhov/membrain.git
cd membrain

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"
```

## Quick Start

### Docker (Recommended)

```bash
# Clone and run with one command
git clone https://github.com/tfatykhov/membrain.git
cd membrain
docker compose -f docker/docker-compose.yml up -d

# Check health (should show "healthy" after ~15s)
docker compose -f docker/docker-compose.yml ps
```

### Custom Configuration

```bash
# Copy example config
cp docker/.env.example docker/.env

# Edit settings (auth token, dimensions, etc.)
nano docker/.env

# Start with custom config
docker compose -f docker/docker-compose.yml --env-file docker/.env up -d
```

### Run Locally (Development)

```bash
# Install dependencies
pip install -e ".[dev]"

# Start server
python -m membrain.server
```

### Connect from Your Agent

```python
import grpc
from membrain.proto import memory_a2a_pb2, memory_a2a_pb2_grpc

# Connect to Membrain
channel = grpc.insecure_channel('localhost:50051')
memory = memory_a2a_pb2_grpc.MemoryUnitStub(channel)

# Store a memory
response = memory.Remember(memory_a2a_pb2.MemoryPacket(
    context_id="doc-001",
    vector=embedding,  # Your 1536-d embedding
    importance=0.8
))

# Recall associated memories
result = memory.Recall(memory_a2a_pb2.QueryPacket(
    vector=query_embedding,
    threshold=0.7
))
print(f"Recalled: {result.context_ids}")
```

## Configuration

Membrain is configured via environment variables. All have sensible defaults.

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MEMBRAIN_PORT` | int | 50051 | gRPC server port |
| `MEMBRAIN_HOST` | string | localhost | Host for healthcheck connection |
| `MEMBRAIN_MAX_WORKERS` | int | 10 | Thread pool size |
| `MEMBRAIN_INPUT_DIM` | int | 1536 | Input embedding dimension |
| `MEMBRAIN_EXPANSION_RATIO` | float | 13.0 | FlyHash expansion ratio |
| `MEMBRAIN_ACTIVE_BITS` | int | auto | Number of active bits in WTA |
| `MEMBRAIN_N_NEURONS` | int | 1000 | SNN neuron count |
| `MEMBRAIN_DT` | float | 0.001 | Simulation timestep (seconds) |
| `MEMBRAIN_SYNAPSE` | float | 0.01 | Synapse time constant (seconds) |
| `MEMBRAIN_SEED` | int | None | Random seed for reproducibility |
| `MEMBRAIN_AUTH_TOKEN` | string | None | Single bearer token |
| `MEMBRAIN_AUTH_TOKENS` | string | None | Comma-separated tokens for multi-client |
| `MEMBRAIN_PRUNE_THRESHOLD` | float | 0.1 | Importance threshold for pruning |
| `MEMBRAIN_NOISE_SCALE` | float | 0.05 | Gaussian noise std for consolidation |
| `MEMBRAIN_MAX_CONSOLIDATION_STEPS` | int | 50 | Max iterations for attractor settling |
| `MEMBRAIN_CONVERGENCE_THRESHOLD` | float | 1e-4 | State diff to consider settled |
| `MEMBRAIN_HEALTH_TIMEOUT` | float | 5.0 | Healthcheck timeout in seconds |
| `MEMBRAIN_LOG_LEVEL` | string | INFO | Minimum log level (DEBUG, INFO, etc.) |
| `MEMBRAIN_LOG_FORMAT` | string | json | Log format (json or text) |
| `MEMBRAIN_LOG_FILE` | string | None | Optional file path for logs |
| `MEMBRAIN_LOG_INCLUDE_TRACE` | bool | false | Include stack traces in logs |

### Example Docker Configuration

```bash
docker run -d \
  -e MEMBRAIN_PORT=50051 \
  -e MEMBRAIN_INPUT_DIM=768 \
  -e MEMBRAIN_AUTH_TOKEN=your-secret-token-here \
  -p 50051:50051 \
  membrain:latest
```

### Programmatic Configuration

```python
from membrain.config import MembrainConfig
from membrain.server import MembrainServer

# Load from environment
config = MembrainConfig.from_env()

# Or create directly
config = MembrainConfig(
    port=50051,
    input_dim=768,  # For smaller embeddings
    n_neurons=500,
    seed=42,        # Reproducible
)

# Validate and start
config.validate()
server = MembrainServer(config=config)
server.start()
```

## API Reference

### gRPC Methods

| Method | Description |
|--------|-------------|
| `Remember(MemoryPacket)` | Store a context vector with learning |
| `Recall(QueryPacket)` | Retrieve associated context IDs |
| `Consolidate(SleepSignal)` | Trigger memory consolidation phase |

### Message Types

```protobuf
message MemoryPacket {
  string context_id = 1;      // UUID of the text chunk
  repeated float vector = 2;  // Dense embedding (1536-d)
  float importance = 3;       // 0.0-1.0, modulates learning rate
}

message QueryPacket {
  repeated float vector = 1;  // Query embedding
  float threshold = 2;        // Similarity threshold
}

message ContextResponse {
  repeated string context_ids = 1;
  float confidence = 2;
}
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run linting
ruff check src/

# Run type checking
mypy src/

# Format code
ruff format src/
```

## Benchmarking

Membrain includes a benchmark suite to measure noise robustness against baseline vector stores.

### Quick Start

```bash
# Install benchmark dependencies
pip install -e ".[bench]"

# Run benchmark
python -m bench.bench_noise --output results/benchmark.csv
```

### Included Baselines

| Baseline | Description |
|----------|-------------|
| `CosineBaseline` | Naive brute-force cosine similarity |
| `FAISSFlatBaseline` | FAISS exact search (IndexFlatIP) |
| `FAISSIVFBaseline` | FAISS approximate search (IVF) |
| `MembrainStore` | Membrain SNN via gRPC |

### Usage

```python
from bench.baselines import CosineBaseline, MembrainStore
from bench.datasets import SyntheticDataset, add_noise

# Generate test data
dataset = SyntheticDataset.gaussian(n=1000, dim=768, seed=42)

# Compare baselines
for Store in [CosineBaseline, MembrainStore]:
    store = Store() if Store != MembrainStore else Store(host="localhost")
    for key, vec in dataset:
        store.store(key, vec)
    
    # Query with noise
    rng = np.random.default_rng(42)
    noisy = add_noise(dataset.vectors[0], level=0.2, rng=rng)
    results = store.query(noisy, k=5)
```

See [bench/README.md](bench/README.md) for full documentation.

## Project Structure

```
membrain/
├── src/
│   └── membrain/
│       ├── __init__.py
│       ├── server.py         # gRPC server
│       ├── encoder.py        # FlyHash implementation
│       ├── core.py           # Nengo SNN network
│       └── proto/            # Generated gRPC stubs
├── tests/
│   ├── test_flyhash.py
│   ├── test_memory.py
│   └── test_integration.py
├── docker/
│   └── Dockerfile
├── protos/
│   └── memory_a2a.proto
├── pyproject.toml
└── README.md
```

## Success Metrics (PoC)

| Metric | Target | Description |
|--------|--------|-------------|
| **SynOp Count** | Linear scaling | Operations scale with active neurons, not total |
| **Sparsity** | >90% | Less than 10% of neurons fire per timestep |
| **Pattern Completion** | 100% @ 20% noise | Full retrieval with noisy queries |

## Roadmap

- [x] PRD and architecture design
- [x] Project scaffolding and CI/CD
- [x] FlyHash encoder implementation
- [x] Nengo SNN core (BiCameralMemory)
- [x] Learning gate for read-only recall
- [x] gRPC server (Feature 01)
- [x] FlyHash int8 optimization (Feature 06)
- [x] gRPC healthcheck (Feature 07)
- [x] Stochastic consolidation (Feature 08)
- [x] Docker containerization (Feature 09)
- [ ] Integration tests
- [ ] Lava process integration (Feature 04)
- [ ] Loihi hardware migration path

## Documentation

- **[Product Vision](docs/PRODUCT_VISION.md)** — Strategic direction, hypotheses, and roadmap
- **[Feature Specs](features/)** — Detailed implementation specifications
- **[Research Notes](research/)** — Related papers and analysis

## References

- [Intel Loihi 2](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html)
- [Nengo Documentation](https://www.nengo.ai/)
- [Lava Framework](https://github.com/lava-nc/lava)
- [FlyHash Paper](https://arxiv.org/abs/1711.03127) — Sparse binary codes for neural systems
- [Voja Learning Rule](https://www.nengo.ai/nengo/examples/learning/learn_associations.html)

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Membrain: Because your AI deserves a hippocampus.*
