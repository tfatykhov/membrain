# Membrain Feature Specifications

This folder contains detailed specifications for all Membrain features.

## Feature Status

### Completed (in codebase)

| # | Feature | Status |
|---|---------|--------|
| 01 | [gRPC A2A Interface](./01-grpc-a2a-interface.md) | ✅ Done (PR #5) |
| 02 | [FlyHash Encoder](./02-flyhash-encoder.md) | ✅ Done |
| 03 | [Neuromorphic Core](./03-neuromorphic-core.md) | ✅ Done |

### P0 — Product Loop

| # | Feature | Status | Priority |
|---|---------|--------|----------|
| 04 | [Lava Process Integration](./04-lava-process-integration.md) | 🔴 Not Started | P0 |
| 05 | [Config System](./05-config-system.md) | 🟡 Partial | P0 |
| 06 | [FlyHash Optimization](./06-flyhash-optimization.md) | 🔴 Not Started | P0 |
| 07 | [Healthcheck](./07-healthcheck.md) | 🟡 Partial | P0 |
| 08 | [Docker Compose](./08-docker-compose.md) | 🟡 Needs Verify | P0 |
| 09 | [Benchmarks](./09-benchmarks.md) | 🔴 Not Started | P0 |

### P1 — Synthetic Hippocampus

| # | Feature | Status | Priority |
|---|---------|--------|----------|
| 10 | [Attractor Dynamics](./10-attractor-dynamics.md) | 🔴 Not Started | P1 |
| 11 | [Temporal Binding](./11-temporal-binding.md) | 🔴 Not Started | P1 |
| 12 | [Persistence](./12-persistence.md) | 🔴 Not Started | P1 |

## Recommended Execution Order

### Phase 1: Product Loop
1. **05-config-system** — Foundational (config dataclass)
2. **06-flyhash-optimization** — Memory reduction (8x)
3. **07-healthcheck** — gRPC health check
4. **08-docker-compose** — One-command run
5. **09-benchmarks** — Prove value vs baselines

### Phase 2: Synthetic Hippocampus
6. **10-attractor-dynamics** — Key differentiator
7. **11-temporal-binding** — Sequence memory
8. **12-persistence** — Production readiness

## Issues Found

1. **FlyHash uses ~245 MB** for default config → fix with int8
2. **Config scattered** → centralize in dataclass
3. **Healthcheck is socket-only** → needs gRPC validation
4. **No baseline comparisons** → can't prove SNN advantage
5. **Recall isn't pattern completion** → needs attractor dynamics

## Definition of Done (All Features)

- [ ] Tests added/updated
- [ ] README updated if user-facing
- [ ] Determinism preserved (seeded)
- [ ] Logging for operations
- [ ] Code review with `gemini-3-pro-high`
- [ ] CI green before merge
