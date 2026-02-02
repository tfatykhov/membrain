# Membrain Feature Specifications

This folder contains detailed specifications for all Membrain features.

## Feature Status

### Completed (in codebase)

| # | Feature | Status |
|---|---------|--------|
| 01 | [gRPC A2A Interface](./01-grpc-a2a-interface.md) | ✅ Done |
| 02 | [FlyHash Encoder](./02-flyhash-encoder.md) | ✅ Done |
| 03 | [Neuromorphic Core](./03-neuromorphic-core.md) | ✅ Done |
| 05 | [Config System](./05-config-system.md) | ✅ Done (PR #14) |
| 06 | [FlyHash Optimization](./06-flyhash-optimization.md) | ✅ Done (PR #17) |
| 07 | [Healthcheck](./07-healthcheck.md) | ✅ Done (PR #18) |

### Phase 1 — Product Loop

| # | Feature | Status | Priority |
|---|---------|--------|----------|
| 08 | [Stochastic Consolidation](./08-stochastic-consolidation.md) | 🔴 Not Started | **CRITICAL** |
| 09 | [Docker Compose](./09-docker-compose.md) | 🟡 Needs Verify | P0 |
| 10 | [Benchmarks](./10-benchmarks.md) | 🔴 Not Started | P0 |

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
1. ~~**05-config-system**~~ — ✅ Done
2. ~~**06-flyhash-optimization**~~ — ✅ Done (8x memory reduction)
3. ~~**07-healthcheck**~~ — ✅ Done (gRPC health check)
4. **08-stochastic-consolidation** — **CRITICAL** (Patent claim enabler)
5. **09-docker-compose** — One-command run
6. **10-benchmarks** — Prove value vs baselines

### Phase 2: Synthetic Hippocampus
7. **11-attractor-dynamics** — Pattern completion
8. **12-temporal-binding** — Sequence memory
9. **13-persistence** — Production readiness

### Phase 3: Hardware Migration
10. **14-lava-process-integration** — Intel Loihi 2 deployment

## Critical Feature: Stochastic Consolidation

**Feature 08** is marked as **CRITICAL** because it:
- Enables the core patent claim for "Attractor Dynamics"
- Provides biological plausibility (mimics hippocampal consolidation)
- Distinguishes Membrain from simple vector search
- Must be implemented before benchmarks can prove the SNN advantage

## Definition of Done (All Features)

- [ ] Tests added/updated
- [ ] README updated if user-facing
- [ ] Determinism preserved (seeded)
- [ ] Logging for operations
- [ ] Code review with `gemini-3-pro-high`
- [ ] CI green before merge
