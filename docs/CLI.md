# CLI & Deployment Guide

Membrain is primarily configured via environment variables.

## Running the Server

### Direct Execution
You can run the server module directly. It will load configuration from the environment.

```bash
# Basic run
python -m membrain.server

# With specific port and token
export MEMBRAIN_PORT=9090
export MEMBRAIN_AUTH_TOKEN="your-secret-token-min-16-chars"
python -m membrain.server
```

## Environment Variables

All configuration is controlled via `MEMBRAIN_` prefixed variables.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MEMBRAIN_PORT` | Integer | `50051` | Port to listen on. |
| `MEMBRAIN_MAX_WORKERS` | Integer | `10` | gRPC thread pool size. |
| `MEMBRAIN_INPUT_DIM` | Integer | `1536` | Input vector dimension (must match client). |
| `MEMBRAIN_EXPANSION_RATIO` | Float | `13.0` | Multiplier for sparse dimension. |
| `MEMBRAIN_ACTIVE_BITS` | Integer | *Auto* | Specific number of active bits (optional). |
| `MEMBRAIN_N_NEURONS` | Integer | `1000` | Size of the SNN population. |
| `MEMBRAIN_DT` | Float | `0.001` | Physics timestep (seconds). |
| `MEMBRAIN_SYNAPSE` | Float | `0.01` | Synaptic decay constant (seconds). |
| `MEMBRAIN_SEED` | Integer | `None` | Set for deterministic execution. |
| `MEMBRAIN_AUTH_TOKEN` | String | - | Single authentication token. |
| `MEMBRAIN_AUTH_TOKENS` | String | - | Comma-separated list of tokens. |
| `MEMBRAIN_PRUNE_THRESHOLD` | Float | `0.1` | Importance threshold for pruning. |
| `MEMBRAIN_HEALTH_TIMEOUT` | Float | `5.0` | Health check timeout (seconds). |

## Validation Errors

The server will fail to start if configuration is invalid.

**Common Errors:**
- `ValueError: Auth token too short (min 16 chars)`: Ensure your token is secure.
- `ValueError: Invalid port`: Port must be 1-65535.
