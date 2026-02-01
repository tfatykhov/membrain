# API Reference

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
