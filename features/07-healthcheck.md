# PR-007 — Consolidate RPC + Meaningful Healthcheck

## Status: ✅ Done (PR #18)

## Current State

**Consolidate RPC** — Already implemented in `server.py` ✅

**Docker healthcheck** — Current uses socket check only:
```dockerfile
HEALTHCHECK CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 50051)); s.close()"
```

---

## Objective

Fix Docker healthcheck to validate actual gRPC functionality.

---

## Requirements

### A. gRPC Health Check Script

Create `src/membrain/health_check.py`:

```python
"""gRPC health check for Docker."""
import sys
import grpc
from membrain.proto import memory_a2a_pb2, memory_a2a_pb2_grpc

def main() -> int:
    try:
        channel = grpc.insecure_channel("localhost:50051")
        stub = memory_a2a_pb2_grpc.MemoryUnitStub(channel)
        response = stub.Ping(memory_a2a_pb2.Empty(), timeout=5)
        return 0 if response.success else 1
    except Exception:
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### B. Update Dockerfile

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -m membrain.health_check || exit 1
```

### C. Prune Threshold Config

Add `MEMBRAIN_PRUNE_THRESHOLD` env var for consolidation.

---

## Files / Modules

| File | Action |
|------|--------|
| `src/membrain/health_check.py` | **Create** |
| `docker/Dockerfile` | **Update** |
| `src/membrain/config.py` | **Update** — Add prune_threshold |

---

## Acceptance Criteria

- [x] Health check validates Ping RPC
- [x] `MEMBRAIN_PRUNE_THRESHOLD` configurable
- [x] Container health reflects actual gRPC availability
