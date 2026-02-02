# Membrain Benchmark Log

Tracking benchmark results across versions to measure progress.

## Summary Table

| Date | Commit | Version | Phase | Hit@1 (0%) | Hit@1 (20%) | Hit@1 (40%) | Notes |
|------|--------|---------|-------|------------|-------------|-------------|-------|
| 2026-02-02 | 1622ae3 | 0.4.0 | baseline | 1.00 | 1.00 | 1.00 | Gaussian too easy, need clustered |

## Benchmark Protocol

1. **Synthetic Dataset:** Gaussian, n=100, dim=128, seed=42
2. **Noise Levels:** 0%, 5%, 10%, 20%, 30%, 40%, 50%
3. **Metrics:** hit@1, hit@5, MRR, latency (avg, p99), memory, throughput
4. **Baselines:** CosineBaseline, FAISSFlatBaseline (when available)

## Result Files

Each benchmark run saves:
- `benchmarks/YYYY-MM-DD-{phase}.json` — Full results with metadata
- `benchmarks/YYYY-MM-DD-{phase}.csv` — Raw CSV output

## Phases

- **baseline** — Before Feature 11 (Attractor Dynamics)
- **attractor** — After Feature 11
- **stdp** — After Feature 12 (Temporal Binding)
- **sleep** — After Feature 13 (Consolidation)

---

## Detailed Results

### Baseline (Pre-Attractor) — 2026-02-02

**Commit:** 1622ae3 | **Version:** 0.4.0

**CosineBaseline Results:**
| Noise | Hit@1 | Hit@5 | MRR | Latency (ms) |
|-------|-------|-------|-----|--------------|
| 0% | 1.00 | 1.00 | 1.00 | 0.28 |
| 10% | 1.00 | 1.00 | 1.00 | 0.43 |
| 20% | 1.00 | 1.00 | 1.00 | 0.62 |
| 30% | 1.00 | 1.00 | 1.00 | 0.39 |
| 40% | 1.00 | 1.00 | 1.00 | 0.54 |
| 50% | 1.00 | 1.00 | 1.00 | 0.48 |
| 60% | 1.00 | 1.00 | 1.00 | 0.20 |
| 70% | 1.00 | 1.00 | 1.00 | 0.45 |

**Issues Found:**
1. ⚠️ Synthetic Gaussian dataset is too easy — 100% recall even at 70% noise
2. ❌ FAISSFlatBaseline has init bug (missing `dim` arg)
3. ⏸️ Membrain server not running (baseline-only)

**Action Items:**
- [ ] Add `--dataset clustered` option to bench_noise.py
- [ ] Fix FAISSFlatBaseline initialization
- [ ] Run full benchmark with Membrain server
