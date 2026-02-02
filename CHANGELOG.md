# Changelog

All notable changes to Membrain will be documented in this file.

## [0.3.0] - 2026-02-02

### Changed
- **BREAKING**: `consolidate()` now returns `tuple[int, int]` (steps_to_converge, pruned_count)
- **BREAKING**: `SleepSignal` proto message redesigned for stochastic consolidation
- **BREAKING**: `Consolidate` RPC now returns `ConsolidateResponse` instead of `Ack`

### Added
- **Feature 09**: Docker Compose support for one-command startup (`docker compose up`)
- **Feature 08**: Stochastic consolidation with Gaussian white noise injection (attractor dynamics)
- **Feature 07**: gRPC Health Checking Protocol support
- Convergence detection loop (settles when state difference < threshold)
- New config params: `noise_scale`, `max_consolidation_steps`, `convergence_threshold`
- `seed` parameter in `BiCameralMemory` for reproducible consolidation
- 8 new consolidation tests covering noise resilience and spurious state rejection

## [0.2.0] - 2026-02-02

### Changed
- **BREAKING**: FlyHash encoder now uses int8 {-1, +1} projection matrix instead of sparse float32 {0, 1}
  - Memory reduced from ~245 MB to ~30 MB for default config (8x improvement)
  - Sparse codes will differ from v0.1.0 — any persisted codes need re-encoding
- Removed `connection_probability` parameter (no longer applicable with dense binary projection)

### Added
- `FlyHash.memory_bytes` property to check projection memory usage
- Memory efficiency tests in test suite

## [0.1.0] - 2026-02-01

### Added
- Initial release
- FlyHash encoder with locality-sensitive hashing
- BiCameralMemory SNN-based associative memory
- gRPC A2A interface (Remember, Recall, Consolidate, Ping)
- Token authentication with constant-time comparison
- MembrainConfig centralized configuration
- Python 3.11+ support
