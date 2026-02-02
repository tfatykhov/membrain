# Changelog

All notable changes to Membrain will be documented in this file.

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
