# AGENTS.md - Agentic Development Guide

**Purpose:** This document provides essential context for AI agents (GPT, Claude, Gemini, etc.) working on the Membrain codebase.

---

## Project Overview

**Membrain** is a Neuromorphic Memory Bridge for LLM Agents — a Spiking Neural Network (SNN) based memory system providing associative recall and continuous learning for AI agents. Think of it as a synthetic hippocampus.

| Aspect | Details |
|--------|---------|
| **Language** | Python 3.9+ |
| **Status** | PoC (Proof of Concept) |
| **Core Tech** | Nengo, nengo-loihi, Lava, gRPC |
| **Target HW** | CPU (simulating Intel Loihi 2) |

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM Agent                              │
└─────────────────────┬───────────────────────────────────────┘
                      │ gRPC (A2A Protocol) - Port 50051
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Membrain Service                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   gRPC API   │──│   FlyHash    │──│  Nengo SNN Core  │  │
│  │  (A2A Proto) │  │   Encoder    │  │  (Loihi Emu)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input:** LLM agent sends 1536-d embedding vector via gRPC
2. **Transduction:** FlyHash converts to 20,000-d sparse binary spikes
3. **Processing:** Nengo SNN stores/retrieves via Voja learning
4. **Output:** Context IDs returned to agent

---

## Directory Structure

```
membrain/
├── src/membrain/           # Main source code
│   ├── __init__.py         # Package init, exports __version__
│   ├── server.py           # gRPC server (TODO: implement)
│   ├── cli.py              # CLI interface (serve, status, version)
│   ├── encoder.py          # FlyHash encoder (TODO: create)
│   ├── core.py             # Nengo SNN network (TODO: create)
│   └── proto/              # Generated gRPC stubs
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_core.py        # Placeholder tests (FlyHash, Memory)
├── protos/
│   └── memory_a2a.proto    # gRPC service definition
├── docker/
│   ├── Dockerfile          # Container build file
│   └── docker-compose.yml  # Docker Compose config
├── ProductRequiremets/
│   └── prd.md              # Detailed PRD with implementation specs
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI/CD pipeline
├── pyproject.toml          # Project config (hatchling build)
└── README.md               # User documentation
```

---

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `src/membrain/server.py` | gRPC MemoryUnit service | Stub only |
| `src/membrain/cli.py` | CLI entry points | Working |
| `src/membrain/encoder.py` | FlyHash implementation | **Not created** |
| `src/membrain/core.py` | Nengo SNN BiCameralMemory | **Not created** |
| `protos/memory_a2a.proto` | A2A protocol definition | Complete |
| `ProductRequiremets/prd.md` | Full implementation specs | Reference |

---

## gRPC API (A2A Protocol)

**Service:** `MemoryUnit` on port `50051`

| Method | Request | Response | Description |
|--------|---------|----------|-------------|
| `Remember` | `MemoryPacket` | `Ack` | Store context vector with learning |
| `Recall` | `QueryPacket` | `ContextResponse` | Retrieve via pattern completion |
| `Consolidate` | `SleepSignal` | `Ack` | Memory consolidation (sleep phase) |
| `Ping` | `Empty` | `Ack` | Health check |

### Key Message Types

```protobuf
MemoryPacket {
  string context_id;      // UUID of memory
  repeated float vector;  // 1536-d embedding
  float importance;       // 0.0-1.0 learning rate mod
  map<string, string> metadata;
}

QueryPacket {
  repeated float vector;  // Query embedding
  float threshold;        // Similarity threshold
  int32 max_results;
}
```

---

## Development Commands

```powershell
# Install dependencies (dev mode)
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Linting
ruff check src/

# Type checking
mypy src/ --ignore-missing-imports

# Format code
ruff format src/

# Generate protobuf stubs
python -m grpc_tools.protoc -I./protos --python_out=./src/membrain/proto --grpc_python_out=./src/membrain/proto ./protos/memory_a2a.proto

# Start server
python -m membrain.server

# Docker build
docker build -t membrain:latest -f docker/Dockerfile .

# Docker run
docker compose -f docker/docker-compose.yml up -d
```

---

## Implementation Roadmap

Based on `ProductRequiremets/prd.md`:

- [x] Project scaffolding and CI/CD
- [x] gRPC protocol definition (`memory_a2a.proto`)
- [ ] FlyHash encoder implementation (`encoder.py`)
- [ ] Nengo SNN core (`core.py`)
- [ ] gRPC server implementation (`server.py`)
- [ ] Integration tests
- [ ] Docker containerization testing

---

## Core Concepts for Agents

### 1. FlyHash Encoding

Converts dense 1536-d vectors into sparse 20,000-d binary codes:

- Random projection matrix `M` of shape `(d, d*r)`
- Winner-Take-All: keep top `k` active bits
- Guarantees fixed sparsity for SNN efficiency

### 2. Associative Memory (Nengo)

Uses `nengo.Ensemble` with Voja learning rule:

- **Write:** Run SNN 50ms with plasticity ON
- **Read:** Run SNN 20ms with plasticity OFF, network settles to attractor state

### 3. Pattern Completion

Query with noisy/partial input → network settles → returns complete memory. Target: 100% accuracy with 20% noise.

---

## Testing Strategy

Tests in `tests/test_core.py` (currently placeholders):

| Test Class | Focus |
|------------|-------|
| `TestFlyHash` | Sparsity, similar/dissimilar vector handling |
| `TestAssociativeMemory` | Remember/recall, pattern completion, sparsity rate |

**Success Metrics:**

- SynOp Count: Linear scaling with active neurons
- Sparsity Rate: >90% (less than 10% neurons fire)
- Pattern Completion: 100% at 20% noise

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOIHI_BACKEND` | `SIM` | Backend mode (`SIM` for CPU emulation) |
| `MEMBRAIN_PORT` | `50051` | gRPC server port |

---

## Dependencies

**Core:**

- `nengo>=3.2.0` - Neural simulation
- `nengo-loihi>=1.1.0` - Loihi emulator backend
- `lava-nc>=0.5.0` - Intel Lava framework
- `grpcio>=1.50.0` - gRPC runtime
- `numpy>=1.24.0` - Numerical operations

**Dev:**

- `pytest>=7.0.0` - Testing
- `ruff>=0.1.0` - Linting/formatting
- `mypy>=1.0.0` - Type checking

---

## Coding Conventions

1. **Type hints required** - All functions must have type annotations
2. **Docstrings** - Use Google-style docstrings for all public APIs
3. **Line length** - 100 characters max
4. **Imports** - Use `isort` ordering (ruff handles this)
5. **Testing** - All new features must have corresponding unit tests

---

## Common Tasks for Agents

### Adding New Functionality

1. Read `ProductRequiremets/prd.md` for implementation specs
2. Create/modify source in `src/membrain/`
3. Add corresponding tests in `tests/`
4. Run `ruff check` and `mypy` before committing
5. Ensure all tests pass: `python -m pytest tests/ -v`

### Implementing FlyHash Encoder

Reference: PRD Section "Feature 2: The Encoder (FlyHash)"

- Create `src/membrain/encoder.py`
- Class `FlyHash` with random projection + WTA
- Test semantic similarity preservation

### Implementing SNN Core

Reference: PRD Section "Feature 3: The Neuromorphic Core"

- Create `src/membrain/core.py`
- Class `BiCameralMemory` using `nengo.Ensemble`
- Voja learning rule for plasticity

---

## CI/CD Pipeline

GitHub Actions (`.github/workflows/ci.yml`):

- Runs on push/PR to `main`
- Matrix: Python 3.9, 3.10, 3.11
- Steps: lint → type-check → test → coverage → docker build

---

## Contact & Resources

- **Repository:** <https://github.com/tfatykhov/membrain>
- **Nengo Docs:** <https://www.nengo.ai/>
- **Intel Lava:** <https://github.com/lava-nc/lava>
- **FlyHash Paper:** <https://arxiv.org/abs/1711.03127>
