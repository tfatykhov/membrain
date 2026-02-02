# Membrain Feature Specifications

This folder contains detailed specifications for all Membrain features.

## Feature Status

### Completed (in codebase)

| # | Feature | Status |
|---|---------|--------|
| 01 | [gRPC A2A Interface](./01-grpc-a2a-interface.md) | ✅ Done |
| 02 | [FlyHash Encoder](./02-flyhash-encoder.md) | ✅ Done |
| 03 | [Neuromorphic Core](./03-neuromorphic-core.md) | ✅ Done |
| 04 | [Config System](./04-config-system.md) | ✅ Done (PR #14) |
| 05 | [FlyHash Optimization](./05-flyhash-optimization.md) | ✅ Done (PR #17) |
| 06 | [Healthcheck](./06-healthcheck.md) | ✅ Done (PR #18) |
| 07 | [Stochastic Consolidation](./07-stochastic-consolidation.md) | ✅ Done (PR #20) |
| 08 | [Docker Compose](./08-docker-compose.md) | ✅ Done (PR #21) |

### Phase 1 — Product Loop

| # | Feature | Status | Priority |
|---|---------|--------|----------|
| 09 | [Benchmarks](./09-benchmarks.md) | 🔴 Not Started | P0 |
| 10 | [Structured Logging](./10-structured-logging.md) | 🔴 Not Started | P0 |

### Phase 2 — Synthetic Hippocampus

| # | Feature | Status | Priority |
|---|---------|--------|----------|
| 11 | [Attractor Dynamics](./11-attractor-dynamics.md) | 🔴 Not Started | P1 |
| 12 | [Temporal Binding](./12-temporal-binding.md) | 🔴 Not Started | P1 |
| 13 | [Persistence](./13-persistence.md) | 🔴 Not Started | P1 |

### Phase 3 — Hardware Migration

| # | Feature | Status | Priority |
|---|---------|--------|----------|
| 14 | [Lava Process Integration](./14-lava-process-integration.md) | 🔴 Not Started | P2 |

## Recommended Execution Order

### Phase 1: Product Loop
1. ~~**04-config-system**~~ — ✅ Done
2. ~~**05-flyhash-optimization**~~ — ✅ Done (8x memory reduction)
3. ~~**06-healthcheck**~~ — ✅ Done (gRPC health check)
4. ~~**07-stochastic-consolidation**~~ — ✅ Done (Patent claim enabler)
5. ~~**08-docker-compose**~~ — ✅ Done (One-command run)
6. **10-structured-logging** — Observability infrastructure
7. **09-benchmarks** — Prove value vs baselines

### Phase 2: Synthetic Hippocampus
8. **11-attractor-dynamics** — Pattern completion
9. **12-temporal-binding** — Sequence memory
10. **13-persistence** — Production readiness

### Phase 3: Hardware Migration
11. **14-lava-process-integration** — Intel Loihi 2 deployment

## Definition of Done (All Features)

- [ ] Tests added/updated
- [ ] README updated if user-facing
- [ ] Determinism preserved (seeded)
- [ ] Logging for operations
- [ ] Code review with `gemini-3-pro-high`
- [ ] CI green before merge
