# PR-004 — Config System (Fast CI Mode + Tunable Runtime)

## Status: ✅ Done (PR #14)

## Current State Analysis

### What Exists
The current `server.py` already reads these environment variables:
- `MEMBRAIN_PORT` (default: 50051)
- `MEMBRAIN_MAX_WORKERS` (default: 10)
- `MEMBRAIN_INPUT_DIM` (default: 1536)
- `MEMBRAIN_EXPANSION_RATIO` (default: 13.0)
- `MEMBRAIN_N_NEURONS` (default: 1000)
- `MEMBRAIN_AUTH_TOKEN` / `MEMBRAIN_AUTH_TOKENS`

### What's Missing
1. **No config dataclass** — values scattered across code
2. **No validation** — invalid values not caught early
3. **No config logging** — resolved config not logged at startup
4. **Missing parameters:**
   - `MEMBRAIN_ACTIVE_BITS` — number of active bits in FlyHash output
   - `MEMBRAIN_DT` — simulation timestep
   - `MEMBRAIN_SYNAPSE` — synapse time constant
   - `MEMBRAIN_SEED` — random seed for reproducibility
5. **Tests don't use small dims** — CI may be slower than needed

---

## Objective

Make all major dimensions and learning params configurable (env/flags) so CI runs fast and deployments tune behavior.

---

## Detailed Requirements

### A. Config Dataclass

Create `src/membrain/config.py`:

```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class MembrainConfig:
    """Central configuration for Membrain server."""
    
    # Server
    port: int = 50051
    max_workers: int = 10
    
    # Encoder (FlyHash)
    input_dim: int = 1536
    expansion_ratio: float = 13.0
    active_bits: Optional[int] = None  # None = use default
    
    # Neural Network
    n_neurons: int = 1000
    dt: float = 0.001  # Simulation timestep (seconds)
    synapse: float = 0.01  # Synapse time constant (seconds)
    
    # Reproducibility
    seed: Optional[int] = None
    
    @classmethod
    def from_env(cls) -> "MembrainConfig":
        """Load configuration from environment variables."""
        return cls(
            port=int(os.environ.get("MEMBRAIN_PORT", 50051)),
            max_workers=int(os.environ.get("MEMBRAIN_MAX_WORKERS", 10)),
            input_dim=int(os.environ.get("MEMBRAIN_INPUT_DIM", 1536)),
            expansion_ratio=float(os.environ.get("MEMBRAIN_EXPANSION_RATIO", 13.0)),
            active_bits=_parse_optional_int(os.environ.get("MEMBRAIN_ACTIVE_BITS")),
            n_neurons=int(os.environ.get("MEMBRAIN_N_NEURONS", 1000)),
            dt=float(os.environ.get("MEMBRAIN_DT", 0.001)),
            synapse=float(os.environ.get("MEMBRAIN_SYNAPSE", 0.01)),
            seed=_parse_optional_int(os.environ.get("MEMBRAIN_SEED")),
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Invalid port: {self.port}")
        if self.input_dim < 1:
            raise ValueError(f"input_dim must be positive")
        if self.expansion_ratio < 1.0:
            raise ValueError(f"expansion_ratio must be >= 1.0")
        if self.n_neurons < 1:
            raise ValueError(f"n_neurons must be positive")
```

### B. Server Integration

- Load config once at startup via `MembrainConfig.from_env()`
- Call `config.validate()` before starting
- Log resolved config (excluding sensitive values)

### C. Seed Support

- Pass seed to `FlyHash` for deterministic projection matrix
- Pass seed to Nengo network builder if applicable

---

## Files / Modules

| File | Action |
|------|--------|
| `src/membrain/config.py` | **Create** |
| `src/membrain/server.py` | **Update** |
| `src/membrain/encoder.py` | **Update** — Accept seed |
| `tests/conftest.py` | **Update** — Small dimension fixtures |
| `README.md` | **Update** — Document env variables |

---

## Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MEMBRAIN_PORT` | int | 50051 | gRPC server port |
| `MEMBRAIN_MAX_WORKERS` | int | 10 | Thread pool size |
| `MEMBRAIN_INPUT_DIM` | int | 1536 | Input embedding dimension |
| `MEMBRAIN_EXPANSION_RATIO` | float | 13.0 | FlyHash expansion ratio |
| `MEMBRAIN_ACTIVE_BITS` | int | auto | Number of active bits (WTA) |
| `MEMBRAIN_N_NEURONS` | int | 1000 | SNN neuron count |
| `MEMBRAIN_DT` | float | 0.001 | Simulation timestep (s) |
| `MEMBRAIN_SYNAPSE` | float | 0.01 | Synapse time constant (s) |
| `MEMBRAIN_SEED` | int | None | Random seed |

---

## Acceptance Criteria

- [ ] Config dataclass with validation exists
- [ ] Server logs resolved config at startup
- [ ] All tests run with small dimensions (CI fast)
- [ ] Seed parameter enables deterministic behavior
- [ ] README documents all env variables
