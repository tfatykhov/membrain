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
| `MEMBRAIN_USE_ATTRACTOR` | bool | false | Enable attractor dynamics for cleanup. |
| `MEMBRAIN_ATTRACTOR_LEARNING_RATE` | float | 0.3 | Hebbian learning rate for attractors. |
| `MEMBRAIN_ATTRACTOR_MAX_STEPS` | int | 50 | Max dynamics iterations for cleanup. |
| `MEMBRAIN_MAX_CONSOLIDATION_STEPS` | int | 50 | Max iterations for attractor settling. |
| `MEMBRAIN_CONVERGENCE_THRESHOLD` | float | 1e-4 | State difference to consider settled. |
| `MEMBRAIN_USE_PES` | bool | true | Enable PES decoder learning. |
| `MEMBRAIN_PES_LEARNING_RATE` | float | 1e-4 | Learning rate for PES rule. |
| `MEMBRAIN_LOG_LEVEL` | string | INFO | Minimum log level. |
| `MEMBRAIN_LOG_FORMAT` | string | json | Log format (json or text). |
| `MEMBRAIN_LOG_FILE` | string | None | Optional file path for logs. |
| `MEMBRAIN_LOG_INCLUDE_TRACE` | bool | false | Include stack traces in logs. |

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

## Learning Configuration

Membrain uses multiple learning rules that can be tuned:

- **`MEMBRAIN_USE_PES`**: Enables Prescribed Error Sensitivity (PES) learning for the decoders.
  - When true, the system minimizes reconstruction error by adjusting decoder weights.
  - Default: `true`.

- **`MEMBRAIN_PES_LEARNING_RATE`**: Controls the speed of error correction.
  - Default: `1e-4`.

## Consolidation Tuning

The stochastic consolidation process is controlled by three key parameters:

- **`MEMBRAIN_NOISE_SCALE`**: Controls the magnitude of the random "kick" injected into the system during sleep. 
  - Higher values (>0.1) help escape deep local minima but might destabilize weak memories.
  - Lower values (<0.01) might not be enough to move the system state.

- **`MEMBRAIN_MAX_CONSOLIDATION_STEPS`**: The maximum number of recurrent cycles to run.
  - If the system doesn't settle within this limit, consolidation is aborted for that cycle.

- **`MEMBRAIN_CONVERGENCE_THRESHOLD`**: How stable the state must be to be considered "settled".
  - Smaller values (e.g., 1e-5) require stricter stability.

## Logging Configuration

Structured logging is enabled by default to facilitate observability and debugging.

- **`MEMBRAIN_LOG_LEVEL`**: Controls the verbosity of logs.
  - Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
  - Default: `INFO`.

- **`MEMBRAIN_LOG_FORMAT`**: Determines the output format.
  - `json`: Structured JSON output, ideal for production monitoring and log aggregation systems (e.g., Datadog, ELK).
  - `text`: Human-readable text format, suitable for local development and testing.

- **`MEMBRAIN_LOG_FILE`**:
  - If set, logs are written to the specified file path.
  - If unset (default), logs are written to standard output (stdout).

- **`MEMBRAIN_LOG_INCLUDE_TRACE`**:
  - If `true`, full stack traces are included in error logs for detailed debugging.
  - Default: `false`.
