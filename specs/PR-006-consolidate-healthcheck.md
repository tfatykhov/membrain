# PR-006 — Consolidate RPC + Meaningful Healthcheck

## Status: 🟡 Partial

## Current State Analysis

### What Exists

**Consolidate RPC** — Already implemented in `server.py`:
```python
def Consolidate(self, request, context) -> memory_a2a_pb2.Ack:
    duration_ms = request.duration_ms if request.duration_ms > 0 else 1000
    with self._lock:
        pruned = self.memory.consolidate(
            duration_ms=duration_ms,
            prune_weak=request.prune_weak,
        )
    # Returns Ack with success + message
```

**Docker healthcheck** — Current `docker/Dockerfile`:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 50051)); s.close()" || exit 1
```

### What's Missing

1. **Healthcheck is socket-only** — Doesn't validate gRPC works
2. **No prune threshold config** — `prune_weak` behavior not configurable
3. **No consolidate params in config** — defaults hardcoded

---

## Objective

1. Make Docker healthcheck validate actual gRPC functionality (not just port open)
2. Add configurable prune threshold and consolidate parameters

---

## Detailed Requirements

### A. Meaningful gRPC Healthcheck

Create a lightweight health check script:

```python
# src/membrain/health_check.py
"""gRPC health check for Docker."""

import sys
import grpc
from membrain.proto import memory_a2a_pb2, memory_a2a_pb2_grpc

def main() -> int:
    try:
        channel = grpc.insecure_channel("localhost:50051")
        stub = memory_a2a_pb2_grpc.MemoryUnitStub(channel)
        
        response = stub.Ping(memory_a2a_pb2.Empty(), timeout=5)
        
        if response.success and response.message == "pong":
            return 0
        else:
            return 1
    except Exception:
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Update `Dockerfile`:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -m membrain.health_check || exit 1
```

### B. Prune Threshold Configuration

Add to config:
```python
# In config.py
@dataclass
class MembrainConfig:
    # ... existing fields ...
    
    # Consolidation
    prune_threshold: float = 0.1  # Memories below this importance get pruned
    default_consolidate_ms: int = 1000  # Default consolidation duration
```

Environment variable:
- `MEMBRAIN_PRUNE_THRESHOLD` — float, default 0.1

### C. Update BiCameralMemory.consolidate()

Ensure the prune threshold is actually used:

```python
# In core.py
def consolidate(
    self,
    duration_ms: int = 1000,
    prune_weak: bool = False,
    prune_threshold: float = 0.1,
) -> int:
    """
    Run consolidation phase.
    
    Args:
        duration_ms: Duration to run network without input.
        prune_weak: If True, remove memories below prune_threshold.
        prune_threshold: Importance threshold for pruning.
    
    Returns:
        Number of memories pruned.
    """
    pruned = 0
    if prune_weak:
        # Remove weak memories
        to_remove = [
            entry.context_id 
            for entry in self._entries.values() 
            if entry.importance < prune_threshold
        ]
        for context_id in to_remove:
            del self._entries[context_id]
            pruned += 1
    
    # Run network for settling (if needed)
    # ... existing consolidation logic ...
    
    return pruned
```

### D. Server Integration

Pass prune threshold from config:

```python
def Consolidate(self, request, context) -> memory_a2a_pb2.Ack:
    duration_ms = request.duration_ms if request.duration_ms > 0 else self.config.default_consolidate_ms
    
    with self._lock:
        pruned = self.memory.consolidate(
            duration_ms=duration_ms,
            prune_weak=request.prune_weak,
            prune_threshold=self.config.prune_threshold,
        )
    # ...
```

---

## Files / Modules

| File | Action |
|------|--------|
| `src/membrain/health_check.py` | **Create** — gRPC health check script |
| `docker/Dockerfile` | **Update** — Use gRPC health check |
| `src/membrain/config.py` | **Update** — Add prune_threshold, default_consolidate_ms |
| `src/membrain/core.py` | **Update** — Accept prune_threshold param |
| `src/membrain/server.py` | **Update** — Pass config to consolidate |

---

## Tests

### Unit Tests

```python
def test_consolidate_prunes_weak_memories(servicer):
    """Consolidate with prune_weak=True removes low-importance memories."""
    # Store a weak memory
    vector = np.random.randn(64).astype(np.float32).tolist()
    servicer.Remember(
        memory_a2a_pb2.MemoryPacket(
            context_id="weak-001",
            vector=vector,
            importance=0.05,  # Below default threshold of 0.1
        ),
        MagicMock(),
    )
    
    # Consolidate with pruning
    response = servicer.Consolidate(
        memory_a2a_pb2.SleepSignal(duration_ms=10, prune_weak=True),
        MagicMock(),
    )
    
    assert response.success
    assert "pruned 1" in response.message.lower()

def test_consolidate_respects_threshold(servicer):
    """Memory at threshold should NOT be pruned."""
    vector = np.random.randn(64).astype(np.float32).tolist()
    servicer.Remember(
        memory_a2a_pb2.MemoryPacket(
            context_id="threshold-001",
            vector=vector,
            importance=0.1,  # Exactly at threshold
        ),
        MagicMock(),
    )
    
    response = servicer.Consolidate(
        memory_a2a_pb2.SleepSignal(duration_ms=10, prune_weak=True),
        MagicMock(),
    )
    
    assert "pruned 0" in response.message.lower()
```

### Integration Tests

```python
def test_health_check_succeeds_when_server_running():
    """Health check returns 0 when server is running."""
    # Start server in background
    # Run health_check.py
    # Assert exit code 0

def test_health_check_fails_when_server_down():
    """Health check returns 1 when server is not running."""
    # Don't start server
    # Run health_check.py
    # Assert exit code 1
```

---

## Acceptance Criteria

- [ ] Health check validates Ping RPC (not just socket)
- [ ] `MEMBRAIN_PRUNE_THRESHOLD` configurable
- [ ] Consolidate doesn't deadlock under lock serialization
- [ ] Container health reflects actual gRPC availability
- [ ] Pruning respects threshold correctly

---

## Risks / Notes

- **Health check timeout**: Ping should be fast (<100ms), but set reasonable timeout
- **Lock contention**: Consolidate holds lock during pruning — should be quick
- **Prune during active use**: Consider whether to allow pruning during high load

---

## Definition of Done

- [ ] Tests added for healthcheck and pruning behavior
- [ ] README updated if config changes
- [ ] Logging for consolidate operations
- [ ] Docker healthcheck works end-to-end
