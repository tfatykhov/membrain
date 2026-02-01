# Feature 01: A2A Interface (gRPC)

**Status:** Partial (Proto defined, server stub only)  
**Priority:** P0 - Critical Path  
**Target File:** `src/membrain/server.py`  
**Proto Definition:** `protos/memory_a2a.proto`

---

## Objective

Define and implement a strict contract for Agent-to-Memory communication using gRPC. This enables any LLM agent to connect to Membrain for associative memory operations.

---

## API Specification

### Service Definition

```protobuf
service MemoryUnit {
  rpc Remember (MemoryPacket) returns (Ack) {}
  rpc Recall (QueryPacket) returns (ContextResponse) {}
  rpc Consolidate (SleepSignal) returns (Ack) {}
  rpc Ping (Empty) returns (Ack) {}
}
```

### Message Types

#### MemoryPacket (Write Request)

| Field | Type | Description |
|-------|------|-------------|
| `context_id` | `string` | UUID of the text chunk/memory |
| `vector` | `repeated float` | Dense embedding (1536-d from OpenAI Ada-002) |
| `importance` | `float` | 0.0-1.0, modulates learning rate |
| `metadata` | `map<string, string>` | Optional key-value metadata |

#### QueryPacket (Read Request)

| Field | Type | Description |
|-------|------|-------------|
| `vector` | `repeated float` | Query embedding |
| `threshold` | `float` | Similarity threshold (0.0-1.0) |
| `max_results` | `int32` | Maximum results to return |

#### ContextResponse

| Field | Type | Description |
|-------|------|-------------|
| `context_ids` | `repeated string` | IDs of recalled memories |
| `confidences` | `repeated float` | Confidence score per result |
| `overall_confidence` | `float` | Aggregate confidence |

#### SleepSignal (Consolidation Request)

| Field | Type | Description |
|-------|------|-------------|
| `duration_ms` | `int32` | Duration of consolidation phase |
| `prune_weak` | `bool` | Whether to prune weak associations |

#### Ack (Response)

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Operation success |
| `message` | `string` | Status message or error |

---

## Implementation Steps

### Step 1.1: Generate Python Stubs ✅

```powershell
python -m grpc_tools.protoc -I./protos --python_out=./src/membrain/proto --grpc_python_out=./src/membrain/proto ./protos/memory_a2a.proto
```

### Step 1.2: Implement MemoryUnitServicer

Create the gRPC servicer class in `src/membrain/server.py`:

```python
import grpc
from concurrent import futures
from membrain.proto import memory_a2a_pb2, memory_a2a_pb2_grpc

class MemoryUnitServicer(memory_a2a_pb2_grpc.MemoryUnitServicer):
    def __init__(self, memory_core, encoder):
        self.memory = memory_core  # BiCameralMemory instance
        self.encoder = encoder     # FlyHash instance
    
    def Remember(self, request, context):
        # 1. Encode vector using FlyHash
        # 2. Inject into SNN with plasticity ON
        # 3. Run for 50ms
        # 4. Return Ack
        pass
    
    def Recall(self, request, context):
        # 1. Encode query vector using FlyHash
        # 2. Inject into SNN with plasticity OFF
        # 3. Run for 20ms
        # 4. Decode attractor state to context IDs
        # 5. Return ContextResponse
        pass
    
    def Consolidate(self, request, context):
        # 1. Trigger sleep phase
        # 2. Optionally prune weak associations
        # 3. Return Ack
        pass
    
    def Ping(self, request, context):
        return memory_a2a_pb2.Ack(success=True, message="pong")
```

### Step 1.3: Server Lifecycle

```python
def serve(port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    memory_a2a_pb2_grpc.add_MemoryUnitServicer_to_server(
        MemoryUnitServicer(memory_core, encoder), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    server.wait_for_termination()
```

---

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MEMBRAIN_PORT` | `50051` | gRPC server port |
| `MEMBRAIN_MAX_WORKERS` | `10` | Thread pool size |

---

## Dependencies

- `grpcio>=1.50.0` - gRPC runtime
- `grpcio-tools>=1.50.0` - Protobuf compiler
- `protobuf>=4.21.0` - Protocol buffers

---

## Acceptance Criteria

- [ ] Proto stubs generated successfully
- [ ] Server starts and listens on configured port
- [ ] `Ping` RPC returns successfully
- [ ] `Remember` stores memory and returns Ack
- [ ] `Recall` retrieves memories with confidence scores
- [ ] `Consolidate` triggers sleep phase
- [ ] Graceful shutdown on SIGTERM

---

## Testing

### Unit Tests

```python
# tests/test_grpc_server.py
class TestMemoryUnitServicer:
    def test_ping_returns_pong(self):
        """Ping should return success=True"""
        pass
    
    def test_remember_stores_memory(self):
        """Remember should store vector and return success"""
        pass
    
    def test_recall_returns_context_ids(self):
        """Recall should return matching context IDs"""
        pass
```

### Integration Tests

```python
# tests/test_integration.py
def test_remember_then_recall():
    """Store a memory, then recall it with the same vector"""
    pass

def test_recall_with_noisy_query():
    """Store a memory, recall with 20% noise added"""
    pass
```

### Manual Verification

1. Start server: `python -m membrain.server`
2. Use `grpcurl` to test endpoints:

   ```bash
   grpcurl -plaintext localhost:50051 memory_bridge.MemoryUnit/Ping
   ```

---

## Error Handling

| Error Case | gRPC Code | Message |
|------------|-----------|---------|
| Invalid vector dimensions | `INVALID_ARGUMENT` | "Vector must be 1536 dimensions" |
| Memory not found | `NOT_FOUND` | "No memories match query" |
| SNN simulation error | `INTERNAL` | "Memory simulation failed" |
| Server overloaded | `RESOURCE_EXHAUSTED` | "Server at capacity" |
