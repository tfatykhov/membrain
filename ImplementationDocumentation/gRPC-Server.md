# gRPC Server & A2A Interface

`src/membrain/server.py`

## Purpose
The **Membrain Server** exposes the neuromorphic memory capabilities via a standard gRPC interface. This acts as the "Agent-to-Agent" (A2A) bridge, allowing any agent or service (Python, Go, Node.js, etc.) to offload memory tasks to the Membrain unit.

## Design Rationale
- **gRPC:** Chosen for high performance, strict type safety (Protobuf), and streaming support (future-proofing).
- **Thread Safety:** Nengo simulators are not inherently thread-safe for concurrent read/writes. The server implements a `threading.RLock` to serialize access to the core memory unit.
- **Security:** Implements Bearer token authentication to prevent unauthorized access.

## Class: `MemoryUnitServicer`

Implements the `memory_a2a_pb2_grpc.MemoryUnitServicer` interface.

### Initialization
```python
def __init__(
    self,
    input_dim: int = 1536,
    expansion_ratio: float = 13.0,
    n_neurons: int = 1000
)
```
Initializes the internal `FlyHash` encoder and `BiCameralMemory` core.

### RPC Methods

#### `Remember`
```protobuf
rpc Remember (MemoryPacket) returns (Ack)
```
- **Process:**
  1. Validates `context_id` (regex) and `importance`.
  2. Acquires lock.
  3. Encodes dense vector -> sparse.
  4. Calls `memory.remember()`.
- **Validation:** Enforces `context_id` format and `importance` range (0.0 - 1.0).

#### `Recall`
```protobuf
rpc Recall (QueryPacket) returns (ContextResponse)
```
- **Process:**
  1. Validates input vector shape.
  2. Acquires lock.
  3. Encodes query -> sparse.
  4. Calls `memory.recall()`.
  5. Computes overall confidence (mean of results).

#### `Consolidate`
```protobuf
rpc Consolidate (SleepSignal) returns (Ack)
```
- **Process:**
  1. Acquires lock.
  2. Calls `memory.consolidate()`.
  3. Returns count of pruned memories.

## Security: `TokenAuthInterceptor`

Middleware that intercepts every gRPC call.

- **Mechanism:** Checks `authorization` metadata header.
- **Format:** `Bearer <token>`
- **Validation:** Uses `hmac.compare_digest` for timing-safe string comparison.
- **Multi-tenant:** Supports a dictionary of `client_id -> token` mappings (loaded from `MEMBRAIN_AUTH_TOKENS` env var).

## Configuration (Environment Variables)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MEMBRAIN_PORT` | `50051` | Listening port. |
| `MEMBRAIN_MAX_WORKERS` | `10` | Thread pool size. |
| `MEMBRAIN_INPUT_DIM` | `1536` | Expected input vector dimension. |
| `MEMBRAIN_AUTH_TOKENS` | `None` | JSON dict `{"client": "token"}`. |
| `MEMBRAIN_AUTH_TOKEN` | `None` | Legacy single token. |

## Lifecycle Management (`MembrainServer`)
- Handles `SIGTERM` and `SIGINT` signals for graceful shutdown.
- Ensures the Nengo simulator resources are released properly via `servicer.shutdown()`.

## Usage Example (Client)

```python
import grpc
from membrain.proto import memory_a2a_pb2, memory_a2a_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = memory_a2a_pb2_grpc.MemoryUnitStub(channel)

# Store
stub.Remember(memory_a2a_pb2.MemoryPacket(
    context_id="doc-123",
    vector=[0.1, ...], # 1536 floats
    importance=0.9
), metadata=(('authorization', 'Bearer my-secret-token'),))

# Recall
response = stub.Recall(memory_a2a_pb2.QueryPacket(
    vector=[0.1, ...],
    threshold=0.8
), metadata=(('authorization', 'Bearer my-secret-token'),))
```

## Known Limitations / TODOs
- **TLS:** Currently uses `insecure_port`. Production deployment requires `ssl_server_credentials`.
- **Blocking:** The global lock `_lock` makes the server effectively single-threaded for memory operations. While `FlyHash` could be parallelized, `nengo.Simulator` is the bottleneck.
- **Error Handling:** Standard gRPC status codes are used, but detailed error payloads (via `grpc-status-details-bin`) could be added.
