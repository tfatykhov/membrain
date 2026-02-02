# Configuration

Membrain is configured via environment variables or programmatically via the `MembrainConfig` class.

## Environment Variables

All configuration is loaded from environment variables with sensible defaults.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MEMBRAIN_PORT` | int | 50051 | gRPC server listening port. |
| `MEMBRAIN_MAX_WORKERS` | int | 10 | Number of thread pool workers. |
| `MEMBRAIN_INPUT_DIM` | int | 1536 | Input embedding dimension. |
| `MEMBRAIN_EXPANSION_RATIO` | float | 13.0 | FlyHash expansion ratio. |
| `MEMBRAIN_ACTIVE_BITS` | int | auto | Number of active bits in WTA. |
| `MEMBRAIN_N_NEURONS` | int | 1000 | Number of neurons in the SNN. |
| `MEMBRAIN_DT` | float | 0.001 | Simulation timestep in seconds. |
| `MEMBRAIN_SYNAPSE` | float | 0.01 | Synapse time constant in seconds. |
| `MEMBRAIN_SEED` | int | None | Random seed for reproducibility. |
| `MEMBRAIN_AUTH_TOKEN` | string | None | Single auth token (legacy). |
| `MEMBRAIN_AUTH_TOKENS` | string | None | Comma-separated auth tokens. |
| `MEMBRAIN_PRUNE_THRESHOLD` | float | 0.1 | Importance threshold for pruning weak memories. |
| `MEMBRAIN_NOISE_SCALE` | float | 0.05 | Gaussian noise std for stochastic consolidation. |
| `MEMBRAIN_MAX_CONSOLIDATION_STEPS` | int | 50 | Max iterations for attractor settling. |
| `MEMBRAIN_CONVERGENCE_THRESHOLD` | float | 1e-4 | State difference to consider settled. |

## Programmatic Configuration

You can also configure the server using the `MembrainConfig` class:

```python
from membrain.config import MembrainConfig

config = MembrainConfig(
    port=50051,
    noise_scale=0.1,  # Higher noise for exploration
    max_consolidation_steps=100
)
```

## Consolidation Tuning

The stochastic consolidation process is controlled by three key parameters:

- **`MEMBRAIN_NOISE_SCALE`**: Controls the magnitude of the random "kick" injected into the system during sleep. 
  - Higher values (>0.1) help escape deep local minima but might destabilize weak memories.
  - Lower values (<0.01) might not be enough to move the system state.

- **`MEMBRAIN_MAX_CONSOLIDATION_STEPS`**: The maximum number of recurrent cycles to run.
  - If the system doesn't settle within this limit, consolidation is aborted for that cycle.

- **`MEMBRAIN_CONVERGENCE_THRESHOLD`**: How stable the state must be to be considered "settled".
  - Smaller values (e.g., 1e-5) require stricter stability.
