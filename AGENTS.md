# AGENTS.md - Agentic Development Guide

**Purpose:** Essential context for AI agents working on Membrain.

---

## Project Overview

**Membrain** is a Neuromorphic Memory Bridge for LLM Agents — a Spiking Neural Network (SNN) based memory system providing associative recall and continuous learning. Think of it as a synthetic hippocampus.

| Aspect | Details |
|--------|---------|
| **Language** | Python 3.11+ |
| **Version** | v0.4.0 |
| **Status** | Active Development |
| **Core Tech** | Nengo, gRPC, NumPy |
| **Target HW** | CPU (Loihi 2 planned for Phase 3) |

---

## Architecture

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
│  │   gRPC API   │──│   FlyHash    │──│  BiCameralMemory │  │
│  │  + Logging   │  │  (int8 proj) │  │  (Nengo SNN)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input:** LLM agent sends 1536-d embedding via gRPC
2. **Encoding:** FlyHash converts to ~20,000-d sparse binary (int8 projection, 8x memory efficient)
3. **Processing:** Nengo SNN stores/retrieves via Voja learning
4. **Consolidation:** Stochastic attractor dynamics with noise injection
5. **Output:** Context IDs + confidence scores returned

---

## Directory Structure

```
membrain/
├── src/membrain/           # Main source code
│   ├── __init__.py         # Package init
│   ├── server.py           # gRPC server + auth
│   ├── config.py           # MembrainConfig dataclass
│   ├── encoder.py          # FlyHash (int8 projection)
│   ├── core.py             # BiCameralMemory (Nengo SNN)
│   ├── logging.py          # Structured JSON logging
│   ├── interceptors.py     # gRPC LoggingInterceptor
│   ├── health_check.py     # Docker healthcheck
│   └── proto/              # Generated gRPC stubs
├── tests/                  # 141+ tests
├── protos/
│   └── memory_a2a.proto    # gRPC service definition
├── docker/
│   ├── Dockerfile          # Container build
│   ├── docker-compose.yml  # One-command run
│   └── .env.example        # Config template
├── features/               # Feature specifications (01-14)
├── docs/                   # Documentation
└── pyproject.toml          # Project config
```

---

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `server.py` | gRPC MemoryUnit service + auth | ✅ Complete |
| `config.py` | Centralized config from env vars | ✅ Complete |
| `encoder.py` | FlyHash (int8 projection, 8x memory savings) | ✅ Complete |
| `core.py` | BiCameralMemory with stochastic consolidation | ✅ Complete |
| `logging.py` | JSON structured logging + context vars | ✅ Complete |
| `interceptors.py` | gRPC request logging with timing | ✅ Complete |
| `health_check.py` | Docker HEALTHCHECK via Ping RPC | ✅ Complete |

---

## gRPC API

**Service:** `MemoryUnit` on port `50051`

| Method | Request | Response | Description |
|--------|---------|----------|-------------|
| `Remember` | `MemoryPacket` | `Ack` | Store memory with learning |
| `Recall` | `QueryPacket` | `ContextResponse` | Pattern completion recall |
| `Consolidate` | `SleepSignal` | `ConsolidateResponse` | Stochastic attractor settling |
| `Ping` | `Empty` | `Ack` | Health check (auth exempt) |

### Authentication

Token-based auth via `authorization: Bearer <token>` metadata.
- Set via `MEMBRAIN_AUTH_TOKEN` or `MEMBRAIN_AUTH_TOKENS`
- Ping is exempt for Docker healthcheck

---

## Configuration

All via environment variables. See `docs/config.md` for full reference.

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMBRAIN_PORT` | 50051 | gRPC port |
| `MEMBRAIN_INPUT_DIM` | 1536 | Embedding dimension |
| `MEMBRAIN_N_NEURONS` | 1000 | SNN neuron count |
| `MEMBRAIN_SEED` | None | Reproducibility seed |
| `MEMBRAIN_AUTH_TOKEN` | None | Bearer token |
| `MEMBRAIN_LOG_FORMAT` | json | `json` or `text` |
| `MEMBRAIN_NOISE_SCALE` | 0.05 | Consolidation noise |

---

## Development Commands

```bash
# Install
pip install -e ".[dev]"

# Test (141+ tests)
python -m pytest tests/ -v

# Type check
mypy src/membrain/ --ignore-missing-imports

# Lint
ruff check src/

# Start server
python -m membrain.server

# Docker one-command run
docker compose -f docker/docker-compose.yml up -d

# Regenerate proto stubs
python -m grpc_tools.protoc -I./protos \
  --python_out=./src/membrain/proto \
  --grpc_python_out=./src/membrain/proto \
  ./protos/memory_a2a.proto
```

---

## Feature Status

### Completed (v0.4.0)
- ✅ 01: gRPC A2A Interface
- ✅ 02: FlyHash Encoder
- ✅ 03: Neuromorphic Core
- ✅ 04: Config System
- ✅ 05: FlyHash int8 Optimization (8x memory reduction)
- ✅ 06: gRPC Healthcheck
- ✅ 07: Stochastic Consolidation (attractor dynamics)
- ✅ 08: Docker Compose
- ✅ 10: Structured Logging

### Phase 1 Remaining
- 🔴 09: Benchmarks

### Phase 2 (Synthetic Hippocampus)
- 🔴 11: Attractor Dynamics (advanced)
- 🔴 12: Temporal Binding
- 🔴 13: Persistence

### Phase 3 (Hardware)
- 🔴 14: Lava Process Integration (Intel Loihi 2)

---

## Core Concepts

### FlyHash Encoding
- **int8 {-1, +1} projection** — 8x memory reduction vs float64
- Random projection + Winner-Take-All
- ~30 MB for default config (was ~245 MB)

### Stochastic Consolidation
- Injects Gaussian white noise into network state
- Iterates until convergence (attractor settling)
- Mimics hippocampal consolidation during sleep
- Key for patent claim ("Attractor Dynamics")

### Structured Logging
- JSON format with `timestamp`, `level`, `logger`, `message`
- Request correlation via `request_id` context var
- RPC timing logged automatically

---

## Testing

141+ tests covering:
- FlyHash encoding (sparsity, similarity preservation)
- BiCameralMemory (remember, recall, consolidation)
- gRPC server (all RPCs, auth, edge cases)
- Config validation
- Logging (JSON format, context vars)
- Health check

**Run tests:**
```bash
python -m pytest tests/ -v
```

---

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`):
- Python 3.11, 3.12
- Steps: lint → type-check → test → docker build
- Must pass before merge

---

## Coding Conventions

1. **Type hints required** — All functions annotated
2. **Docstrings** — Google-style for public APIs
3. **Line length** — 88 characters (ruff default)
4. **Structured logging** — Use `get_logger(__name__)`
5. **Tests** — All features must have tests
6. **PR workflow** — Branch → Review → CI green → Merge

---

## Resources

- **Repo:** https://github.com/tfatykhov/membrain
- **Docs:** `docs/` folder
- **Features:** `features/` folder (numbered specs)
- **Nengo:** https://www.nengo.ai/
- **FlyHash Paper:** https://arxiv.org/abs/1711.03127
