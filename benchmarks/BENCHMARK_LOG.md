# Membrain Benchmark Log

Tracking benchmark results across versions to measure progress.

## Summary Table

| Date | Commit | Version | Phase | Hit@1 (0%) | Hit@1 (20%) | Hit@1 (40%) | Notes |
|------|--------|---------|-------|------------|-------------|-------------|-------|
| 2026-02-02 | 1622ae3 | 0.4.0 | baseline | 1.00 | 0.84 | 0.36 | Clustered dataset, cosine only |

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

**Commit:** 1622ae3 | **Version:** 0.4.0 | **Dataset:** Clustered (10 clusters, std=0.1)

**CosineBaseline Results:**
| Noise | Hit@1 | Hit@5 | MRR | Latency (ms) |
|-------|-------|-------|-----|--------------|
| 0% | 1.00 | 1.00 | 1.00 | 0.23 |
| 10% | 0.99 | 1.00 | 0.995 | 0.19 |
| 20% | 0.84 | 0.99 | 0.897 | 0.17 |
| 30% | 0.59 | 0.90 | 0.711 | 0.32 |
| 40% | 0.36 | 0.72 | 0.504 | 0.25 |
| 50% | 0.24 | 0.53 | 0.360 | 0.20 |

**Key Insight:** Clustered dataset creates realistic challenge. Hit@1 degrades significantly at 30%+ noise — **this is where attractor dynamics should shine.**

**Thesis to Prove:** Membrain should maintain hit@1 > 0.59 at 30% noise (beat similarity search).
