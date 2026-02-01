# API Reference

## Configuration System

The `membrain.config` module provides a centralized way to configure the server.

### `MembrainConfig`

The main configuration dataclass.

```python
from membrain.config import MembrainConfig
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port` | `int` | `50051` | gRPC server listening port. |
| `max_workers` | `int` | `10` | Number of thread pool workers. |
| `input_dim` | `int` | `1536` | Dimension of input embeddings (e.g., OpenAI ada-002). |
| `expansion_ratio` | `float` | `13.0` | FlyHash expansion factor. |
| `active_bits` | `int` | `None` | Number of active bits in sparse representation (None = auto). |
| `n_neurons` | `int` | `1000` | Number of neurons in the SNN. |
| `dt` | `float` | `0.001` | Simulation timestep in seconds. |
| `synapse` | `float` | `0.01` | Synapse time constant in seconds. |
| `seed` | `int` | `None` | Random seed for reproducibility. |
| `auth_tokens` | `list[str]` | `[]` | List of valid authentication tokens. |

#### Methods

**`from_env() -> MembrainConfig`**
Factory method to load configuration from environment variables.
```python
config = MembrainConfig.from_env()
```

**`for_testing() -> MembrainConfig`**
Factory method creating a lightweight configuration for fast CI/testing.
- Reduces dimensions (64 input, 50 neurons)
- Sets fixed seed (42)
- Includes test auth token
```python
test_config = MembrainConfig.for_testing()
```

**`validate() -> None`**
Checks that all configuration values are valid. Raises `ValueError` if any check fails.
- Ensures positive integers for dimensions
- Checks port range
- Validates token length (min 16 chars)
```python
try:
    config.validate()
except ValueError as e:
    print(f"Config error: {e}")
```

## Server API

### `MembrainServer`

Wrapper for the gRPC server.

```python
from membrain.server import MembrainServer

server = MembrainServer(config=config)
server.start()
server.wait_for_termination()
```
