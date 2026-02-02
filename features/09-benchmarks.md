# Feature 09 — Integration Tests + Benchmark Harness

## Status: 🟡 In Progress (Phase 1 Complete)

## Objective

Provide measurable evidence (noise robustness, hit@k) and baseline comparisons.

---

## Requirements

### A. Noise Robustness Tests

`tests/integration/test_noise_robustness.py`:

```python
@pytest.mark.parametrize("noise_level", [0.0, 0.05, 0.10, 0.20, 0.30])
def test_hit_at_1(stored_memories, noise_level):
    """Test hit@1 accuracy at various noise levels."""
    # Store 100 items, query with noise, measure accuracy
    pass
```

### B. Baseline Comparators

`bench/baselines.py`:

1. **CosineBaseline** — Simple cosine similarity on raw vectors
2. **FlyHashJaccardBaseline** — FlyHash + Jaccard (no SNN)

### C. Benchmark Harness

`bench/bench_noise.py`:

```bash
python bench/bench_noise.py --output results.csv
```

Outputs:
| method | noise_level | hit_at_1 | hit_at_5 | avg_latency_ms |
|--------|-------------|----------|----------|----------------|
| cosine | 0.00 | 1.00 | 1.00 | 0.05 |
| membrain_snn | 0.00 | 0.95 | 0.99 | 2.50 |
| ... | ... | ... | ... | ... |

---

## Files / Modules

| File | Action |
|------|--------|
| `tests/integration/__init__.py` | **Create** |
| `tests/integration/test_noise_robustness.py` | **Create** |
| `bench/__init__.py` | **Create** |
| `bench/baselines.py` | **Create** |
| `bench/bench_noise.py` | **Create** |
| `bench/README.md` | **Create** |

---

## Expected Results

| Method | Hit@1 (0% noise) | Hit@1 (20% noise) |
|--------|------------------|-------------------|
| Cosine | 100% | ~85% |
| FlyHash+Jaccard | ~98% | ~65% |
| Membrain SNN | ~95% | ~75% |

**Goal:** Show SNN advantage at high noise levels.

---

## Acceptance Criteria

- [ ] Results reproducible with seeded randomness
- [ ] Benchmark outputs CSV with all metrics
- [ ] At least one measurable advantage shown vs baseline
- [ ] Integration tests pass in CI
