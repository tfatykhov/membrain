# Membrain Feature Specifications

This folder contains detailed specifications for each planned PR, organized by priority.

## Status Overview

| PR | Name | Priority | Status | Notes |
|----|------|----------|--------|-------|
| PR-001 | Proto Generation | P0 | ✅ Done | Merged in previous work |
| PR-002 | gRPC Server Skeleton | P0 | ✅ Done | Merged as PR #5 |
| PR-003 | Wire Remember/Recall | P0 | ✅ Done | Merged as PR #5 |
| PR-004 | Config System | P0 | 🟡 Partial | Needs dataclass + validation |
| PR-005 | FlyHash Optimization | P0 | 🔴 Not Started | Memory footprint reduction |
| PR-006 | Consolidate + Healthcheck | P0 | 🟡 Partial | Consolidate done, healthcheck needs fix |
| PR-007 | Docker Compose | P0 | 🟡 Partial | Needs verification |
| PR-008 | Benchmarks | P0 | 🔴 Not Started | Integration tests + baselines |
| PR-009 | Attractor Dynamics | P1 | 🔴 Not Started | True pattern completion |
| PR-010 | Temporal Binding | P1 | 🔴 Not Started | Sequence memory |
| PR-011 | Persistence | P1 | 🔴 Not Started | Snapshots + versioning |

## Recommended Execution Order

### Phase 1: Product Loop (P0)
1. **PR-004** — Config System (foundational)
2. **PR-005** — FlyHash Optimization (required for scale)
3. **PR-006** — Healthcheck fix (operability)
4. **PR-007** — Docker Compose verification (usability)
5. **PR-008** — Benchmarks (validation)

### Phase 2: Synthetic Hippocampus (P1)
6. **PR-009** — Attractor Dynamics (key differentiator)
7. **PR-010** — Temporal Binding (sequence memory)
8. **PR-011** — Persistence (production readiness)

## Issues Found During Analysis

### 1. FlyHash Memory Usage (PR-005)
Current implementation uses float64 projection matrix: ~245 MB for default config.
**Recommendation**: Use int8 {-1, +1} projection (8x reduction).

### 2. Config Scattered (PR-004)
Environment variables parsed in multiple places; no validation.
**Recommendation**: Centralize in config.py dataclass.

### 3. Docker Healthcheck (PR-006)
Current healthcheck only checks socket, not gRPC.
**Recommendation**: Create health_check.py that calls Ping RPC.

### 4. No Baseline Comparisons (PR-008)
No evidence Membrain SNN outperforms simple cosine similarity.
**Recommendation**: Add benchmark harness with baselines before claiming advantage.

### 5. Recall Not True Pattern Completion (PR-009)
Current recall is similarity lookup, not attractor dynamics.
**Risk**: "Synthetic hippocampus" claim not defensible without fix.

## Spec Files

- [PR-004: Config System](./PR-004-config-system.md)
- [PR-005: FlyHash Optimization](./PR-005-flyhash-memory-optimization.md)
- [PR-006: Consolidate + Healthcheck](./PR-006-consolidate-healthcheck.md)
- [PR-007: Docker Compose](./PR-007-docker-compose.md)
- [PR-008: Benchmarks](./PR-008-integration-benchmarks.md)
- [PR-009: Attractor Dynamics](./PR-009-attractor-dynamics.md)
- [PR-010: Temporal Binding](./PR-010-temporal-binding.md)
- [PR-011: Persistence](./PR-011-persistence.md)

## Definition of Done (All PRs)

- [ ] Tests added/updated for new behavior
- [ ] README updates if user-facing surface changes
- [ ] Determinism preserved (seeded)
- [ ] Logging for operational events
- [ ] Proto changes include version notes
- [ ] Code review with `gemini-3-pro-high`
- [ ] CI green before merge
