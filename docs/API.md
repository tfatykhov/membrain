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

---

## Server API

### `MembrainServer`

Wrapper for the gRPC server.

```python
from membrain.server import MembrainServer

server = MembrainServer(config=config)
server.start()
server.wait_for_termination()
```

---

## Protocol Buffers (v1)

Defined in `protos/memory_a2a.proto`.

### Service: `MemoryUnit`

| RPC | Request | Response | Description |
| :--- | :--- | :--- | :--- |
| `Remember` | `MemoryPacket` | `Ack` | Store a vector with learning. |
| `Recall` | `QueryPacket` | `ContextResponse` | Query associative memory. |
| `Consolidate` | `SleepSignal` | `Ack` | Run sleep phase/cleanup. |
| `Ping` | `Empty` | `Ack` | Health check. |

### Messages

#### `MemoryPacket`
| Field | Type | Description |
| :--- | :--- | :--- |
| `context_id` | `string` | Unique identifier for the memory (1-256 chars). |
| `vector` | `repeated float` | Dense input embedding. |
| `importance` | `float` | Learning weight (0.0 to 1.0). |
| `metadata` | `map<string, string>` | Arbitrary key-value tags (optional). |

#### `QueryPacket`
| Field | Type | Description |
| :--- | :--- | :--- |
| `vector` | `repeated float` | Query embedding vector. |
| `threshold` | `float` | Minimum similarity (0.0 to 1.0). |
| `max_results` | `int32` | Max items to return. |

#### `ContextResponse`
| Field | Type | Description |
| :--- | :--- | :--- |
| `context_ids` | `repeated string` | Matched memory IDs. |
| `confidences` | `repeated float` | Similarity scores. |
| `overall_confidence` | `float` | Average confidence of results. |

#### `SleepSignal`
| Field | Type | Description |
| :--- | :--- | :--- |
| `duration_ms` | `int32` | Milliseconds to simulate sleep. |
| `prune_weak` | `bool` | If true, remove low-importance memories. |

---

## Python API (Internal)

### `BiCameralMemory`
`membrain.core.BiCameralMemory`

| Method | Parameters | Returns | Description |
| :--- | :--- | :--- | :--- |
| `__init__` | `n_neurons, dimensions, learning_rate` | `self` | Initialize SNN. |
| `remember` | `context_id, sparse_vector, importance` | `bool` | Learn pattern. |
| `recall` | `query_vector, threshold` | `list[RecallResult]` | Retrieve pattern. |
| `consolidate` | `duration_ms, prune_weak` | `int` | Maintenance. |

### `FlyHash`
`membrain.encoder.FlyHash`

| Method | Parameters | Returns | Description |
| :--- | :--- | :--- | :--- |
| `__init__` | `input_dim, expansion_ratio, active_bits` | `self` | Initialize encoder. |
| `encode` | `vector` | `ndarray` | Dense -> Sparse. |
| `encode_batch` | `vectors` | `ndarray` | Batch encoding. |
