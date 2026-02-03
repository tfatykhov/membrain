# Membrain Feature Specifications

This folder contains detailed specifications for all Membrain features.

## Feature Status

### Core Infrastructure (Complete)

| # | Feature | Status | PR |
|---|---------|--------|-----|
| 01 | [gRPC A2A Interface](./01-grpc-a2a-interface.md) | ✅ Done | — |
| 02 | [FlyHash Encoder](./02-flyhash-encoder.md) | ✅ Done | — |
| 03 | [Neuromorphic Core](./03-neuromorphic-core.md) | ✅ Done | — |
| 04 | [Config System](./04-config-system.md) | ✅ Done | #14 |
| 05 | [FlyHash Optimization](./05-flyhash-optimization.md) | ✅ Done | #17 |
| 06 | [Healthcheck](./06-healthcheck.md) | ✅ Done | #18 |
| 07 | [Stochastic Consolidation](./07-stochastic-consolidation.md) | ✅ Done | #20 |
| 08 | [Docker Compose](./08-docker-compose.md) | ✅ Done | #21 |

### Phase 1 — Product Loop (Complete)

| # | Feature | Status | PR |
|---|---------|--------|-----|
| 09 | [Benchmarks](./09-benchmarks.md) | ✅ Done | #28 |
| 10 | [Structured Logging](./10-structured-logging.md) | ✅ Done | #23 |

### Phase 2 — Synthetic Hippocampus

| # | Feature | Status | Priority | PR |
|---|---------|--------|----------|-----|
| 11 | [Attractor Dynamics](./11-attractor-dynamics.md) | ✅ Done | P1 | #20 |
| 15 | [Noise-Robust Recall](./15-noise-robust-recall.md) | 🟡 In Progress | P0 | — |
| 12 | [Temporal Binding](./12-temporal-binding.md) | 🔴 Not Started | P1 | — |
| 13 | [Persistence](./13-persistence.md) | 🔴 Not Started | P1 | — |

### Phase 3 — Hardware Migration

| # | Feature | Status | Priority |
|---|---------|--------|----------|
| 14 | [Lava Process Integration](./14-lava-process-integration.md) | 🔴 Not Started | P2 |

---

## Current Focus

**Feature 15: Noise-Robust Recall** — Making Membrain exceed baselines at noisy recall.

Phased approach:
1. **Phase 1:** Attractor query denoising in recall path
2. **Phase 2:** Pre-seeding for training density  
3. **Phase 3:** Revisit neuron-space comparison

See [15-noise-robust-recall.md](./15-noise-robust-recall.md) for details.

---

## Execution History

### Completed
1. ~~**04-config-system**~~ — Environment + pydantic config
2. ~~**05-flyhash-optimization**~~ — int8 quantization (8x memory reduction)
3. ~~**06-healthcheck**~~ — gRPC health check endpoint
4. ~~**07-stochastic-consolidation**~~ — Noise injection + attractor settling
5. ~~**08-docker-compose**~~ — One-command deployment
6. ~~**10-structured-logging**~~ — JSON logging with request tracking
7. ~~**09-benchmarks**~~ — Noise robustness benchmarks vs baselines
8. ~~**11-attractor-dynamics**~~ — Hopfield-style pattern completion

### In Progress
- **15-noise-robust-recall** — Leveraging attractor for query denoising

### Upcoming
- **12-temporal-binding** — Sequence memory
- **13-persistence** — Durable storage
- **14-lava-process-integration** — Intel Loihi 2 deployment

---

## Supplementary Docs

| File | Purpose |
|------|---------|
| [11-attractor-dynamics-minsky.md](./11-attractor-dynamics-minsky.md) | Minsky Society of Mind mapping for attractor design |

---

## Definition of Done (All Features)

- [ ] Tests added/updated
- [ ] README updated if user-facing
- [ ] Determinism preserved (seeded)
- [ ] Logging for operations
- [ ] Code review with `gemini-3-pro-high`
- [ ] CI green before merge
