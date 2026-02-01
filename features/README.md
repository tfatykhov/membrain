# Feature Specifications

This folder contains detailed feature specifications for the Membrain Neuromorphic Memory Bridge.

## Features Overview

| Feature | File | Status | Priority |
|---------|------|--------|----------|
| A2A Interface (gRPC) | [01-grpc-a2a-interface.md](01-grpc-a2a-interface.md) | 🔜 Next Up | P0 |
| FlyHash Encoder | [02-flyhash-encoder.md](02-flyhash-encoder.md) | ✅ Complete | P0 |
| Neuromorphic Core | [03-neuromorphic-core.md](03-neuromorphic-core.md) | ✅ Complete | P0 |
| Lava Process Integration | [04-lava-process-integration.md](04-lava-process-integration.md) | Not Started | P1 |

## Implementation Order

1. **Feature 2: FlyHash Encoder** - ✅ Complete (PR #2)
2. **Feature 3: Neuromorphic Core** - ✅ Complete (PR #3, #4)
3. **Feature 1: A2A Interface** - 🔜 Next (server implementation)
4. **Feature 4: Lava Integration** - Future-proofing wrapper

## Success Metrics

All features must contribute to these PoC success metrics:

- **SynOp Count:** Linear scaling with active neurons
- **Sparsity Rate:** >90% (less than 10% neurons fire)
- **Pattern Completion:** 100% accuracy at 20% noise
